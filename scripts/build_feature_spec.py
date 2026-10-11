#!/usr/bin/env python3
"""Freeze the deployed feature spec and rebuild the heads' inputs from cached blocks.

The tile-CLIP evidence sweep (scripts/tile_clip_sweep.py) measured every candidate
tile layout and aggregation under one protocol. This script turns the chosen cell into
the frozen artifact set the service loads:

  models/feature_spec.json     block layout (scales, aggregation, patch tokens)
  models/clip_tiles_spec.npy   aggregated tile block for the 119 samples
  models/clip_patch.npy        patch-token statistics (written by
                               scripts/extract_clip_patch_features.py)

Tile blocks are recombined from the per-scale caches produced by the sweep, so no
backbone pass is repeated here.

Usage
  python scripts/build_feature_spec.py --cell C7 --aggregation meanmax --patch
  python scripts/build_feature_spec.py --cell C0 --aggregation legacy      # archived layout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tile_clip_sweep import AGGS, CONFIGS, aggregate_multi, load_tiles   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="C7", help="tile configuration from the sweep table")
    ap.add_argument("--aggregation", default="meanmax")
    ap.add_argument("--patch", action="store_true", help="include CLIP patch-token statistics")
    ap.add_argument("--patch-max-side", type=int, default=1600)
    ap.add_argument("--cache", default="models/tile_cache")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    md = ROOT / args.models_dir
    md.mkdir(parents=True, exist_ok=True)
    if args.cell == "C0" and args.aggregation in ("legacy", "base"):
        agg = np.load(md / "clip_tiles.npy")                     # archived artifact
        scales = [{"grid": 3, "overlap": 0.25, "max_side": 1600}]
        print(f"[tiles] reusing archived clip_tiles.npy {agg.shape}")
    else:
        if args.cell not in CONFIGS:
            raise SystemExit(f"unknown cell {args.cell}; see tile_clip_sweep.py --list")
        agg = aggregate_multi(load_tiles(ROOT / args.cache, args.cell), args.aggregation)
        scales = [{"grid": g, "overlap": o, "max_side": m} for (g, o, m) in CONFIGS[args.cell]]
        print(f"[tiles] {args.cell} agg={args.aggregation} -> {agg.shape}")

    np.save(md / "clip_tiles_spec.npy", agg.astype(np.float32))

    patch_ok = False
    if args.patch:
        p = md / "clip_patch.npy"
        if not p.exists():
            raise SystemExit(f"{p} missing; run scripts/extract_clip_patch_features.py first")
        patch_ok = True
        print(f"[patch] {np.load(p).shape} from {p.name}")

    from feature_spec import FeatureSpec
    spec = FeatureSpec(scales=scales, aggregation=args.aggregation,
                       patch_tokens=patch_ok, patch_max_side=args.patch_max_side,
                       name=args.name or f"{args.cell}-{args.aggregation}"
                                         + ("+patch" if patch_ok else ""))
    spec.save(md / "feature_spec.json")
    print(f"[spec] {json.dumps(spec.to_dict(), ensure_ascii=False)}")
    print(f"[spec] forensic dim = {spec.forensic_dim()}, combined dim = "
          f"{spec.forensic_dim() + 1024 + agg.shape[1]}")
    print(f"[save] {md/'feature_spec.json'} and {md/'clip_tiles_spec.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
