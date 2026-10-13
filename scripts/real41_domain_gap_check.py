"""Verify the suspected IMAGE-HEAD DOMAIN GAP on web-sourced real claim photos.

repo/reports/testset_real_41_eval.md claims the frozen image heads flag these 41
genuine photos at FPR 92.7%. This script re-measures that directly through the
production detector with the repo/ frozen artifacts, and characterises how these
photos differ from the brief's own real class.

Why it matters: the video channel's low AUC came from this gap, not from the video
path. If the gap is real, adding web-sourced real photos as negatives is the fix
that helps BOTH the image head and the video channel.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(r"C:\Files\Deepseek Harness\Zhongan-OPEN-SZ-01")
REPO = WS / "repo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from detector import AIGCDetector, PipelineConfig  # noqa: E402
from models_path import resolve_models_dir  # noqa: E402


def resolve_clip(p: Path) -> Path:
    if (p / "config.json").is_file():
        return p
    if p.is_dir():
        for d in sorted([x for x in p.iterdir() if x.is_dir()], reverse=True):
            if (d / "config.json").is_file():
                return d
    return p


def main():
    md, src = resolve_models_dir(workspace=WS)
    print(f"[models] {md} (source={src})")
    clip = resolve_clip(WS / "hf_cache" / "hub"
                        / "models--openai--clip-vit-large-patch14" / "snapshots")
    cfg = PipelineConfig(
        clip_dir=str(clip),
        clip_probe=str(WS / "models" / "clip_probe_norm.pt"),
        forensic_lr=str(md / "forensic_lr.joblib"),
        clip_domain_lr=str(WS / "models" / "clip_domain_lr.joblib"),
        combined_lr=str(md / "combined_lr.joblib"),
        feature_spec=str(md / "feature_spec.json"),
        fusion_json=str(md / "fusion.json"),
        device="cuda",
    )
    det = AIGCDetector(cfg)
    print(f"[fusion] thr_low={det.fusion.thr_low:.4f} thr_high={det.fusion.thr_high:.6f} "
          f"weights={det.fusion.weights}")

    # warm up (one-time model load must not be charged to the batch)
    det.detect_image(str(WS / "data" / "images" / "001.png"))

    rows = []
    for p in sorted((REPO / "testset" / "real").glob("*.jpg")):
        t = time.perf_counter()
        rep, _ = det.detect_image(str(p))
        ms = (time.perf_counter() - t) * 1000
        rows.append({"file": p.name, "verdict": rep.verdict,
                     "confidence": rep.confidence, "fused_score": rep.fused_score,
                     "model_scores": rep.model_scores, "ms": round(ms, 1)})
    print(f"\n=== 41 web-sourced REAL claim photos (all ground truth = real) ===")
    n = len(rows)
    for v in ("suspect", "review", "normal"):
        k = sum(1 for r in rows if r["verdict"] == v)
        print(f"  {v:8s} {k:3d}/{n}  ({k/n:.1%})")
    sus = sum(1 for r in rows if r["verdict"] == "suspect")
    print(f"\n  FPR (suspect counted as AI) = {sus}/{n} = {sus/n:.4f}")
    conf = np.array([r["confidence"] for r in rows])
    print(f"  confidence: median={np.median(conf):.4f} p90={np.percentile(conf,90):.4f} "
          f"max={conf.max():.4f}")

    # per-member view: which head drives it?
    print("\n  === per-member calibrated scores (mean over the 41) ===")
    members = sorted({m for r in rows for m in r["model_scores"]})
    for m in members:
        v = np.array([r["model_scores"].get(m, np.nan) for r in rows], dtype=float)
        print(f"    {m:14s} mean={np.nanmean(v):.4f} median={np.nanmedian(v):.4f}")

    out = WS / "work" / "improve" / "real41_domain_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "models_dir": str(md), "source": src,
        "thr_low": det.fusion.thr_low, "thr_high": det.fusion.thr_high,
        "weights": det.fusion.weights,
        "n": n, "suspect": sus, "fpr": sus / n,
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("suspect", "review", "normal")},
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
