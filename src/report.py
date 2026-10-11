"""Structured detection report (manual ch.9.2) - JSON + single-file HTML.

The report answers "why did the system reach this verdict", not just the score.
Fields: input identity / verdict / model evidence / forensic evidence /
localization / attention explanation / uncertainty / traceability.

Terminology discipline (manual ch.3.5): the L3 tamper localization and the
classifier attention explanation live in *separate* top-level fields and in
separate HTML sections. An attention map only says which regions the classifier
relied on; it is never presented as - and must never be read as - a tamper or
synthetic-region localization.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

REPORT_VERSION = "1.1.0"


@dataclass
class DetectionReport:
    input_hash: str = ""
    input_type: str = "image"
    resolution: str = ""
    duration_s: float | None = None
    verdict: str = "review"              # normal | review | suspect
    verdict_cn: str = "待复核"
    confidence: float = 0.0
    fused_score: float = 0.0
    model_scores: dict = field(default_factory=dict)
    disagreement: float = 0.0
    forensic_clues: list = field(default_factory=list)   # [{name, status, detail}]
    localization: dict = field(default_factory=dict)     # L3 tamper localization
    explanation: dict = field(default_factory=dict)      # classifier attention explanation
    uncertainty: list = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)

    VERDICT_CN = {"normal": "正常", "review": "待复核", "suspect": "疑似生成/篡改"}

    def finalize(self):
        self.verdict_cn = self.VERDICT_CN.get(self.verdict, self.verdict)
        return self

    def to_json(self) -> str:
        self.finalize()
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# traceability: model weight / threshold hashing (manual ch.9.2, 16.x)
# ---------------------------------------------------------------------------
HASH_EXT = (".joblib", ".pt", ".pth", ".bin", ".safetensors", ".json")
HASH_CHUNK = 1 << 20
_FILE_HASH_CACHE: dict = {}


def sha256_file(path: str) -> str:
    """SHA-256 of a file, cached on (absolute path, size, mtime_ns).

    The cache matters: the same frozen weights are hashed for every sample of an
    evaluation run, so the hash of a 45 MB checkpoint is only paid once.
    """
    ap = os.path.abspath(str(path))
    st = os.stat(ap)
    key = (ap, st.st_size, st.st_mtime_ns)
    hit = _FILE_HASH_CACHE.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with open(ap, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASH_CACHE[key] = digest
    return digest


def sha256_of_obj(obj) -> str:
    """SHA-256 over canonical JSON - identifies a parameter set, not a file."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_hash_table(paths: dict, scan_dirs: bool = True) -> dict:
    """``{file_name: "sha256:<hex>"}`` for the frozen weights and thresholds.

    ``paths`` maps a logical name to a configured artifact. Configured files that
    exist are keyed by their file name; ones that are absent/unusable stay visible
    as ``不可用：…`` so no audit-relevant key is silently dropped. With
    ``scan_dirs`` the containing directories are additionally scanned one level
    deep for :data:`HASH_EXT` artifacts, so a weight that is present but not part
    of the current run is still auditable.
    """
    out: dict = {}
    dirs = set()
    for name, p in dict(paths).items():
        try:
            if not p:
                out[name] = "不可用：未配置"
                continue
            if os.path.isdir(str(p)):
                dirs.add(os.path.abspath(str(p)))
                out[name] = "不可用：目录（未按文件哈希）"
                continue
            if not os.path.isfile(str(p)):
                out[name] = "不可用：文件不存在"
                continue
            out[os.path.basename(str(p))] = "sha256:" + sha256_file(str(p))
            dirs.add(os.path.dirname(os.path.abspath(str(p))))
        except Exception as e:                      # never let traceability break a verdict
            out[name] = f"不可用：{e}"
    if scan_dirs:
        for d in sorted(dirs):
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for fn in names:
                fp = os.path.join(d, fn)
                try:
                    if not os.path.isfile(fp) or not fn.lower().endswith(HASH_EXT):
                        continue
                    if fn in out:
                        continue
                    out[fn] = "sha256:" + sha256_file(fp)
                except Exception as e:
                    out[fn] = f"不可用：{e}"
    return out


def thresholds_sha256(fusion) -> str:
    """SHA-256 over the frozen decision parameters (thresholds/weights/temperatures).

    Thresholds live in ``fusion.json``, whose own file hash is already recorded;
    hashing the *values* as well pins the decision boundary independently of the
    serialization, so a re-formatted file cannot silently change the audit trail.
    """
    try:
        payload = {
            "thr_low": round(float(fusion.thr_low), 10),
            "thr_high": round(float(fusion.thr_high), 10),
            "disagree_review": round(float(getattr(fusion, "disagree_review", 1.25)), 10),
            "weights": {str(k): round(float(v), 10) for k, v in dict(fusion.weights).items()},
            "temperatures": {str(k): round(float(getattr(v, "temperature", v)), 10)
                             for k, v in dict(getattr(fusion, "scalers", {})).items()},
        }
        return "sha256:" + sha256_of_obj(payload)
    except Exception as e:
        return f"不可用：{e}"


def heatmap_to_b64_png(heatmap: np.ndarray, overlay_bgr: np.ndarray | None = None,
                       alpha: float = 0.55) -> str:
    """Encode a float heatmap [0,1] (optionally overlaid on the image) as b64 PNG.

    With an overlay the *overlay's* resolution wins: a coarse map (e.g. the 3x3 tile
    attribution) is resampled down to it instead of being upscaled, which keeps the
    report small without discarding any information the map actually carries.
    """
    import cv2
    hm = np.clip(heatmap, 0, 1).astype(np.float32)
    if overlay_bgr is not None and overlay_bgr.shape[:2] != hm.shape[:2]:
        hm = cv2.resize(hm, (overlay_bgr.shape[1], overlay_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    hm8 = (hm * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    if overlay_bgr is not None:
        color = cv2.addWeighted(overlay_bgr, 1 - alpha, color, alpha, 0)
    ok, buf = cv2.imencode(".png", color)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def save_html(report: DetectionReport, path: str, image_b64: str | None = None):
    """Self-contained single-file HTML report."""
    report.finalize()
    r = report

    def esc(x):
        if isinstance(x, dict):
            return "<br>".join(f"{esc(k)} = {esc(v)}" for k, v in x.items()) or "不可用"
        if isinstance(x, (list, tuple)):
            return esc(", ".join(str(i) for i in x))
        if x is None:
            return "不可用"
        return html.escape(str(x))

    def num(x, fmt="{:.4f}"):
        try:
            return fmt.format(float(x))
        except (TypeError, ValueError):
            return esc(x)

    verdict_color = {"normal": "#2e7d32", "review": "#f9a825", "suspect": "#c62828"}[r.verdict]
    clues_rows = "".join(
        f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('status',''))}</td>"
        f"<td>{esc(c.get('detail',''))}</td></tr>" for c in r.forensic_clues)
    model_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{num(v)}</td></tr>" for k, v in r.model_scores.items())
    lat_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{num(v, '{:.0f}')} ms</td></tr>" for k, v in r.latency_ms.items())
    ver_rows = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in r.versions.items())
    img_tag = (f'<img src="data:image/png;base64,{image_b64}" '
               f'style="max-width:360px;border:1px solid #ccc">') if image_b64 else "<i>不可用</i>"

    # ---- L3 tamper localization panel ----
    loc = r.localization or {}
    loc_img = (f'<img src="data:image/png;base64,{loc.get("heatmap_png_b64","")}" '
               f'style="max-width:360px;border:1px solid #ccc">') if loc.get("heatmap_png_b64") else ""
    loc_status = "可用" if loc_img else "不可用"
    loc_reason = esc(loc.get("reason") or "定位分支未产出图像")
    loc_note = f'<p class="note">不可用说明：{loc_reason}</p>' if not loc_img else ""
    loc_boxes = loc.get("boxes") or []
    loc_rows = "".join(
        f"<tr><td>{i+1}</td><td>({int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])})</td>"
        f"<td>{num(b[4], '{:.3f}')}</td></tr>" for i, b in enumerate(loc_boxes) if len(b) >= 5)
    loc_table = (f'<table><tr><th>#</th><th>x,y,w,h</th><th>异常强度</th></tr>{loc_rows}</table>'
                 if loc_rows else "<p>无可报告异常区域</p>")

    # ---- classifier attention explanation panel (≠ localization) ----
    ex = r.explanation or {}
    ex_img = (f'<img src="data:image/png;base64,{ex.get("heatmap_png_b64","")}" '
              f'style="max-width:360px;border:1px solid #ccc">') if ex.get("heatmap_png_b64") else ""
    ex_status = esc(ex.get("status") or ("可用" if ex_img else "不可用"))
    ex_reason = esc(ex.get("reason") or "关注区域解释未产出图像")
    ex_note = f'<p class="note">不可用说明：{ex_reason}</p>' if not ex_img else ""
    gradcam = ex.get("gradcam") or {}
    tiles = ex.get("tiles") or []
    tile_rows = "".join(
        f"<tr><td>{int(t.get('index', i)) + 1}</td>"
        f"<td>({esc(t.get('row'))},{esc(t.get('col'))})</td>"
        f"<td>{esc(t.get('bbox_xywh'))}</td>"
        f"<td>{num(t.get('delta_logit'))}</td><td>{num(t.get('weight'), '{:.3f}')}</td></tr>"
        for i, t in enumerate(tiles))
    tile_table = (f"<table><tr><th>块#</th><th>网格(row,col)</th><th>x,y,w,h</th>"
                  f"<th>贡献Δ(logit)</th><th>归一化权重</th></tr>{tile_rows}</table>"
                  if tile_rows else "")

    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>检测报告 - {esc(r.input_hash[:12])}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:4px 10px;font-size:13px}}
.badge{{display:inline-block;padding:6px 18px;color:#fff;border-radius:4px;font-size:18px;background:{verdict_color}}}
.grid{{display:flex;gap:24px;flex-wrap:wrap}}
.panel{{border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin-top:8px}}
.panel.loc{{border-left:6px solid #1565c0}}
.panel.ex{{border-left:6px solid #6a1b9a}}
.note{{font-size:12px;color:#666}}
.warn{{font-size:12px;color:#b00020}}
</style></head><body>
<h1>AI 生成材料反欺诈检测报告</h1>
<div class="badge">{esc(r.verdict_cn)}</div>
<p>置信度（校准后）:<b>{num(r.confidence, '{:.3f}')}</b> ｜ 融合分数: {num(r.fused_score, '{:.3f}')} ｜ 模型分歧度: {num(r.disagreement, '{:.3f}')}</p>
<h2>输入标识</h2>
<table><tr><th>SHA-256</th><td>{esc(r.input_hash)}</td></tr>
<tr><th>类型</th><td>{esc(r.input_type)}</td><th>分辨率/时长</th><td>{esc(r.resolution)}{(' / %.1fs' % r.duration_s) if r.duration_s else ''}</td></tr></table>
<div class="grid"><div><h2>输入图像</h2>{img_tag}</div></div>

<h2>A. 伪造区域定位（L3）</h2>
<div class="panel loc">
<p>性质：<b>篡改/合成区域定位</b>（取证异常图，零权重 ELA + 噪声不一致）｜ 状态：<b>{loc_status}</b>｜ 方法：{esc(loc.get('kind') or '—')}</p>
{loc_img}
{loc_note}
{loc_table}
<p class="note">此图为区域定位证据；它与下方 B 节（关注区域解释）是两类不同的输出，不可互相替代。</p>
</div>

<h2>B. 分类器关注区域解释（块级）｜ 不等于伪造区域定位</h2>
<div class="panel ex">
<p>性质：<b>关注区域解释</b>（只说明分类器打分时依赖哪些图像块）｜ 状态：<b>{ex_status}</b>｜ 方法：{esc(ex.get('method') or '—')}</p>
<p>作用头：{esc(ex.get('head') or '—')}｜ 标签：{esc(ex.get('label_cn') or '—')}</p>
{ex_img}
{ex_note}
{tile_table}
<p class="note">Grad-CAM 路径：{esc(gradcam.get('status') or '不可用')}｜ {esc(gradcam.get('reason') or '')}</p>
<p class="warn">{esc(ex.get('disclaimer') or '关注区域解释 ≠ 伪造区域定位。')}</p>
</div>

<h2>模型证据</h2><table><tr><th>模型</th><th>校准后分数</th></tr>{model_rows}</table>
<h2>L1 取证线索</h2><table><tr><th>线索</th><th>状态</th><th>说明</th></tr>{clues_rows}</table>
<h2>不确定性与局限</h2><ul>{''.join(f'<li>{esc(u)}</li>' for u in r.uncertainty) or '<li>无</li>'}</ul>
<h2>耗时</h2><table>{lat_rows}</table>
<h2>可追溯性</h2><table>{ver_rows}</table>
<p style="color:#888;font-size:12px">报告版本 {REPORT_VERSION} ｜ 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 本报告结论为技术判定（材料层面），不构成业务欺诈定性。</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
