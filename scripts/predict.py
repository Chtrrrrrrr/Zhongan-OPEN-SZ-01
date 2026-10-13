#!/usr/bin/env python3
"""Single-image / single-video CLI prediction -> structured report.

Artifact paths are resolved through `scripts/models_path.py`: the frozen repository
copy (`<repo>/models`) wins, and only artifacts the frozen copy does not ship
(`clip_probe_norm.pt`, `clip_domain_lr.joblib`, `l1_lr.joblib`, `hf_cache/`) fall
back to `<workspace>/models` with `source=workspace-fallback` printed.

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
WS = ROOT.parent                                     # workspace root (legacy copy)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import models_path as mp                            # noqa: E402
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
    ap.add_argument("--models-dir", default=None,
                    help="directory holding the deployment artifacts; default resolution "
                         "prefers <repo>/models (the frozen copy), with per-file fallback to "
                         "<workspace>/models - see scripts/models_path.py")
    ap.add_argument("--clip-dir", default=None,
                    help="CLIP-L/14 snapshot dir (default: resolved hf_cache under the models dir)")
    ap.add_argument("--clip-probe", default=None)
    ap.add_argument("--l1-model", default=None)
    ap.add_argument("--forensic-lr", default=None)
    ap.add_argument("--clip-domain-lr", default=None)
    ap.add_argument("--combined-lr", default=None)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--npr", default="", help="NPR weights; empty by default (ablated out, see docs/results.md 9)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # ---- one place decides which artifact copy is loaded ---------------------
    md, md_source = mp.resolve_models_dir(args.models_dir, workspace=WS)
    print(mp.describe(md, md_source))
    print(mp.assert_expected_fusion(md))

    def art(name: str, explicit: str | None):
        if explicit:
            return Path(explicit)
        p, src = mp.resolve_artifact(md, name, workspace=WS)
        if src != "frozen":
            print(f"[models] {name}: {p}  (source={src})")
        return p

    clip_probe = art("clip_probe_norm.pt", args.clip_probe)
    l1_model = art("l1_lr.joblib", args.l1_model)
    forensic_lr = art("forensic_lr.joblib", args.forensic_lr)
    clip_domain_lr = art("clip_domain_lr.joblib", args.clip_domain_lr)
    combined_lr = art("combined_lr.joblib", args.combined_lr)
    fusion_json = art("fusion.json", args.fusion)
    clip_dir = (Path(args.clip_dir) if args.clip_dir
                else art("hf_cache/models--openai--clip-vit-large-patch14/snapshots", None))

    def _opt(p):
        return str(p) if p and Path(p).exists() else None

    cfg = PipelineConfig(
        clip_dir=resolve_clip_dir(str(clip_dir)) if Path(clip_dir).exists() else None,
        clip_probe=_opt(clip_probe),
        l1_lr_model=_opt(l1_model),
        forensic_lr=_opt(forensic_lr),
        clip_domain_lr=_opt(clip_domain_lr),
        combined_lr=_opt(combined_lr),
        fusion_json=_opt(fusion_json),
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
