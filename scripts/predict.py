#!/usr/bin/env python3
"""Single-image / single-video CLI prediction -> structured report.

Usage:
  python scripts/predict.py --image path/to/img.png --out reports/
  python scripts/predict.py --video path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import AIGCDetector, PipelineConfig   # noqa: E402
from report import save_html                        # noqa: E402


def resolve_clip_dir(base: str) -> str:
    p = Path(base)
    if p.is_dir() and not (p / "config.json").exists():
        snaps = [d for d in p.iterdir() if d.is_dir()]
        if snaps:
            return str(snaps[0])
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--clip-dir", default="models/hf_cache/models--openai--clip-vit-large-patch14/snapshots")
    ap.add_argument("--clip-probe", default="models/clip_probe_norm.pt")
    ap.add_argument("--l1-model", default="models/l1_lr.joblib")
    ap.add_argument("--forensic-lr", default="models/forensic_lr.joblib")
    ap.add_argument("--clip-domain-lr", default="models/clip_domain_lr.joblib")
    ap.add_argument("--combined-lr", default="models/combined_lr.joblib")
    ap.add_argument("--fusion", default="models/fusion.json")
    ap.add_argument("--npr", default="", help="NPR weights; empty by default (ablated out, see docs/results.md 9)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    def _opt(p):
        return p if p and Path(p).exists() else None

    cfg = PipelineConfig(
        clip_dir=resolve_clip_dir(args.clip_dir) if Path(args.clip_dir).exists() else None,
        clip_probe=_opt(args.clip_probe),
        l1_lr_model=_opt(args.l1_model),
        forensic_lr=_opt(args.forensic_lr),
        clip_domain_lr=_opt(args.clip_domain_lr),
        combined_lr=_opt(args.combined_lr),
        fusion_json=_opt(args.fusion),
        npr_weights=_opt(args.npr),
        device=args.device,
    )
    det = AIGCDetector(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        rep, conf = det.detect_image(args.image)
        print(json.dumps(json.loads(rep.to_json()), ensure_ascii=False, indent=1))
        stem = Path(args.image).stem
        rep.save_json(str(out_dir / f"{stem}.json"))
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        save_html(rep, str(out_dir / f"{stem}.html"), image_b64=b64)
        print(f"[save] {out_dir / (stem + '.json')} / {out_dir / (stem + '.html')}")
    elif args.video:
        rep = det.detect_video(args.video)
        print(json.dumps(json.loads(rep.to_json()), ensure_ascii=False, indent=1))
        stem = Path(args.video).stem
        rep.save_json(str(out_dir / f"{stem}.json"))
        print(f"[save] {out_dir / (stem + '.json')}")
    else:
        ap.error("provide --image or --video")


if __name__ == "__main__":
    main()
