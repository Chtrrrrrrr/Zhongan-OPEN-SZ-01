"""Structured detection report (manual ch.9.2) - JSON + single-file HTML.

The report answers "why did the system reach this verdict", not just the score.
Fields: input identity / verdict / model evidence / forensic evidence /
localization / uncertainty / traceability.
"""
from __future__ import annotations

import base64
import html
import io
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

REPORT_VERSION = "1.0.0"


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
    localization: dict = field(default_factory=dict)     # {heatmap_png_b64, boxes, kind}
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


def heatmap_to_b64_png(heatmap: np.ndarray, overlay_bgr: np.ndarray | None = None,
                       alpha: float = 0.55) -> str:
    """Encode a float heatmap [0,1] (optionally overlaid on the image) as b64 PNG."""
    import cv2
    hm = np.clip(heatmap, 0, 1)
    hm8 = (hm * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    if overlay_bgr is not None:
        base = overlay_bgr
        if base.shape[:2] != color.shape[:2]:
            base = cv2.resize(base, (color.shape[1], color.shape[0]))
        color = cv2.addWeighted(base, 1 - alpha, color, alpha, 0)
    ok, buf = cv2.imencode(".png", color)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def save_html(report: DetectionReport, path: str, image_b64: str | None = None):
    """Self-contained single-file HTML report."""
    report.finalize()
    r = report

    def esc(x):
        return html.escape(str(x))

    verdict_color = {"normal": "#2e7d32", "review": "#f9a825", "suspect": "#c62828"}[r.verdict]
    clues_rows = "".join(
        f"<tr><td>{esc(c.get('name',''))}</td><td>{esc(c.get('status',''))}</td>"
        f"<td>{esc(c.get('detail',''))}</td></tr>" for c in r.forensic_clues)
    model_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:.4f}</td></tr>" for k, v in r.model_scores.items())
    lat_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:.0f} ms</td></tr>" for k, v in r.latency_ms.items())
    img_tag = f'<img src="data:image/png;base64,{image_b64}" style="max-width:360px;border:1px solid #ccc">' if image_b64 else ""
    loc = r.localization or {}
    loc_tag = (f'<img src="data:image/png;base64,{loc.get("heatmap_png_b64","")}" '
               f'style="max-width:360px;border:1px solid #ccc">') if loc.get("heatmap_png_b64") else "<i>无定位图</i>"

    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>检测报告 - {esc(r.input_hash[:12])}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:4px 10px;font-size:13px}}
.badge{{display:inline-block;padding:6px 18px;color:#fff;border-radius:4px;font-size:18px;background:{verdict_color}}}
.grid{{display:flex;gap:24px;flex-wrap:wrap}}
</style></head><body>
<h1>AI 生成材料反欺诈检测报告</h1>
<div class="badge">{esc(r.verdict_cn)}</div>
<p>置信度（校准后）:<b>{r.confidence:.3f}</b> ｜ 融合分数: {r.fused_score:.3f} ｜ 模型分歧度: {r.disagreement:.3f}</p>
<h2>输入标识</h2>
<table><tr><th>SHA-256</th><td>{esc(r.input_hash)}</td></tr>
<tr><th>类型</th><td>{esc(r.input_type)}</td><th>分辨率/时长</th><td>{esc(r.resolution)}{(' / %.1fs' % r.duration_s) if r.duration_s else ''}</td></tr></table>
<div class="grid"><div><h2>输入图像</h2>{img_tag}</div>
<div><h2>伪造区域定位（L3）</h2>{loc_tag}</div></div>
<h2>模型证据</h2><table><tr><th>模型</th><th>校准后分数</th></tr>{model_rows}</table>
<h2>L1 取证线索</h2><table><tr><th>线索</th><th>状态</th><th>说明</th></tr>{clues_rows}</table>
<h2>不确定性与局限</h2><ul>{''.join(f'<li>{esc(u)}</li>' for u in r.uncertainty) or '<li>无</li>'}</ul>
<h2>耗时</h2><table>{lat_rows}</table>
<h2>可追溯性</h2><table>{''.join(f'<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>' for k, v in r.versions.items())}</table>
<p style="color:#888;font-size:12px">报告版本 {REPORT_VERSION} ｜ 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 本报告结论为技术判定（材料层面），不构成业务欺诈定性。</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
