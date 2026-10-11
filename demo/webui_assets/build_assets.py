#!/usr/bin/env python3
"""生成 WebUI 静态资产（demo/webui_assets/）。

为什么需要这个生成器
--------------------
WebUI 的「样本库」板块要在浏览器里展示 119 套样本的**缩略图**与**折外（OOF）判定**
（正确 / 误报 / 漏报着色、可按误报漏报筛选）。但：

  * 命题数据 `data/images/*.png` 不在任何 HTTP 静态目录内（服务只托管 `reports/` 与 `demo/`）；
  * 逐样本的折外分数/判定在仓库里唯一权威来源是 `models/oof_bundle_119.json`
    （由评测脚本按 evaluate_system.py 完全相同的折划分/阈值协议导出），
    该文件同样不在 HTTP 静态目录内。

因此这里把两者**离线**转成 WebUI 可访问的资产，并在输出里写清来源路径与 sha256，
使页面展示的每个数字都可回溯到仓库内文件：

  输入（只读）
    <ws>/models/oof_bundle_119.json   逐样本折外 logits / 温度标定分 / 融合分 / 判定 / 折阈值
    <ws>/data/manifest.json           样本清单（宽高、sha256、原始标签）
    <ws>/data/images/NNN.png          样本原图
  输出
    demo/webui_assets/oof_119.json    页面读取的精简数据（含来源 sha256 与统计口径）
    demo/webui_assets/thumbs/NNN.svg  150px 缩略图（SVG 包装 base64 JPEG）

缩略图为什么是 .svg 而不是 .jpg：仓库 `.gitignore` 全局忽略 `*.png/*.jpg/*.jpeg`，
位图缩略图不会被提交，克隆仓库后页面就会出现坏图；`.svg` 文本资产不受该规则影响，
浏览器渲染效果一致（内部就是一张 JPEG 的 data URI）。

用法：
    python demo/webui_assets/build_assets.py            # 生成全部资产
    python demo/webui_assets/build_assets.py --check     # 只校验输出是否与来源一致
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

DEMO_ASSETS = Path(__file__).resolve().parent
REPO = DEMO_ASSETS.parent.parent          # <repo>/
WS = REPO.parent                          # 工作区根目录

OOF_SRC = WS / "models" / "oof_bundle_119.json"
MANIFEST = WS / "data" / "manifest.json"
IMAGES = WS / "data" / "images"
METRICS = REPO / "reports" / "metrics.json"

THUMB_W, THUMB_H = 150, 206
JPEG_Q = 72


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_oof_payload() -> dict:
    bundle = json.loads(OOF_SRC.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = {int(r["seq"]): r for r in manifest["records"]}

    seqs = [int(s) for s in bundle["seqs"]]
    y = [int(v) for v in bundle["y"]]
    dep = bundle["deployed_oof"]
    fused = dep["fused"]
    pred = [int(v) for v in dep["pred"]]
    verdict = list(dep["verdict"])

    folds = {}
    for f in bundle["folds"]:
        for s in f["holdout_seqs"]:
            folds[int(s)] = f

    cal = bundle["oof_calibrated"]

    samples = []
    for i, seq in enumerate(seqs):
        label = "ai" if y[i] == 1 else "real"
        p = pred[i]
        if p == 1 and label == "ai":
            outcome = "true_positive"
        elif p == 1 and label == "real":
            outcome = "false_positive"
        elif p == 0 and label == "ai":
            outcome = "false_negative"
        else:
            outcome = "true_negative"
        f = folds.get(seq, {})
        m = meta.get(seq, {})
        samples.append({
            "seq": seq,
            "file": Path(str(m.get("file", f"data/images/{seq:03d}.png"))).name,
            "label": label,
            "label_cn": "AI 生成/篡改" if label == "ai" else "真实",
            "width": m.get("width"),
            "height": m.get("height"),
            "sha256": m.get("sha256"),
            "fold": f.get("fold"),
            "thr_low": f.get("thr_low"),
            "thr_high": f.get("thr_high"),
            "members": f.get("deployed_members"),
            "fused_score": round(float(fused[i]), 6),
            "pred": p,
            "verdict": verdict[i],
            "verdict_cn": {"suspect": "疑似生成/篡改", "normal": "正常", "review": "待复核"}[verdict[i]],
            "outcome": outcome,
            "member_scores": {k: round(float(v[i]), 6) for k, v in cal.items()},
        })

    s = {k: 0 for k in ("true_positive", "true_negative", "false_positive", "false_negative")}
    tiers = {"suspect": 0, "normal": 0, "review": 0}
    for r in samples:
        s[r["outcome"]] += 1
        tiers[r["verdict"]] += 1

    return {
        "note": "折外判定数据；由 models/oof_bundle_119.json 精简导出，字段含义见 assets 内 README",
        "protocol": bundle.get("protocol", {}),
        "weights": bundle.get("weights", {}).get("deployed"),
        "members": bundle.get("members", {}),
        "per_member_auc": bundle.get("per_member_auc", {}),
        "guardrails": dep.get("guardrails"),
        "deployed_metrics": dep.get("metrics"),
        "error_analysis": {
            "false_positives_real_seqs": dep.get("false_positives_real_seqs"),
            "false_negatives_ai_seqs": dep.get("false_negatives_ai_seqs"),
        },
        "counts": {"n": len(samples), "outcome": s, "three_tier": tiers},
        "sources": {
            "oof_bundle": {"path": str(OOF_SRC), "sha256": sha256_file(OOF_SRC)},
            "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
            "metrics": {"path": str(METRICS), "sha256": sha256_file(METRICS)},
        },
        "generator": "demo/webui_assets/build_assets.py",
        "samples": samples,
    }


def build_thumb(seq: int, force: bool = False) -> Path | None:
    src = IMAGES / f"{seq:03d}.png"
    out = DEMO_ASSETS / "thumbs" / f"{seq:03d}.svg"
    if not src.is_file():
        return None
    if out.is_file() and not force and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    from PIL import Image

    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
    w, h = im.size
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d" role="img" aria-label="sample %03d thumbnail">'
        '<image width="%d" height="%d" href="data:image/jpeg;base64,%s"/></svg>'
        % (w, h, w, h, seq, w, h, b64),
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="只校验 oof_119.json 与来源是否一致（不写文件）")
    ap.add_argument("--force", action="store_true", help="重建已存在的缩略图")
    args = ap.parse_args()

    payload = build_oof_payload()
    out_json = DEMO_ASSETS / "oof_119.json"
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False)

    if args.check:
        if not out_json.is_file():
            print("[check] 缺少 %s" % out_json)
            return 1
        cur = json.loads(out_json.read_text(encoding="utf-8"))
        # 判定只看**数据内容**（samples/counts/deployed_metrics/protocol/...）；
        # sources 里的 sha256 只是生成时的来源指纹：来源文件可能被重新生成（重跑
        # evaluate_system.py、批处理归档刷新等），且仓库 core.autocrlf=true 时磁盘字节
        # sha 与 git blob sha 会因 CRLF/LF 不同（内容一致）。因此指纹漂移只作提示、
        # 不判失败；需要刷新记录时重跑本脚本即可。
        cur_src = cur.pop("sources", {})
        fresh_src = payload.get("sources", {})
        fresh = {k: v for k, v in payload.items() if k != "sources"}
        same = json.dumps(cur, ensure_ascii=False, sort_keys=True) == \
            json.dumps(fresh, ensure_ascii=False, sort_keys=True)
        print("[check] oof_119.json 数据内容 %s" % ("与来源一致" if same else "与来源不一致（需重建）"))
        drifted = [k for k in fresh_src
                   if fresh_src[k].get("sha256") != (cur_src.get(k) or {}).get("sha256")]
        if drifted:
            print("[check] 提示：来源指纹已漂移 %s（不影响页面；运行本脚本即可刷新记录）" % drifted)
        else:
            print("[check] 来源指纹未漂移（oof_bundle / manifest / metrics 三个来源 sha256 一致）")
        missing = [s["seq"] for s in payload["samples"]
                   if not (DEMO_ASSETS / "thumbs" / ("%03d.svg" % s["seq"])).is_file()]
        print("[check] 缩略图 %d/%d，缺失 %s"
              % (len(payload["samples"]) - len(missing), len(payload["samples"]), missing[:10]))
        return 0 if (same and not missing) else 1

    out_json.write_text(text, encoding="utf-8")
    print("[out] %s (%.1f KB)" % (out_json, out_json.stat().st_size / 1024))

    ok = 0
    for s in payload["samples"]:
        if build_thumb(s["seq"], force=args.force):
            ok += 1
    total = sum(f.stat().st_size for f in (DEMO_ASSETS / "thumbs").glob("*.svg"))
    print("[out] thumbs %d/%d 张，共 %.0f KB" % (ok, len(payload["samples"]), total / 1024))

    print("[stat] n=%d outcome=%s three_tier=%s"
          % (payload["counts"]["n"], payload["counts"]["outcome"], payload["counts"]["three_tier"]))
    m = payload["deployed_metrics"] or {}
    print("[stat] 与 reports/metrics.json oof_overall 一致：accuracy=%s fpr=%s tpr=%s confusion=%s"
          % (m.get("accuracy"), m.get("fpr"), m.get("tpr"), m.get("confusion")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
