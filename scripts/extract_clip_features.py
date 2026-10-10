#!/usr/bin/env python3
"""Batch-extract CLIP features (global pooled + native-resolution tiles).

Outputs (default dir models/):
  clip_feats.npy   - [N, D]      global 224px pooled CLIP feature
  clip_tiles.npy   - [N, 3D]     mean/max/std over 3x3 overlapping native-res tiles

The tiled view is what makes rendered documents separable from photographed
paper: it keeps native text/table detail that a single 224px resize destroys.

Usage: python scripts/extract_clip_features.py --data-dir data --out-dir models
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l0_input import parse_image                       # noqa: E402
from l2_clip import CLIPFeatureExtractor               # noqa: E402
from l2_clip_tiles import extract_tiled_features       # noqa: E402

CLIP_REPO = "openai/clip-vit-large-patch14"


def resolve_clip(name: str) -> str:
    p = Path(name)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p)
    for root in (ROOT / "models" / "hf_cache", ROOT.parent / "models" / "hf_cache",
                 Path.home() / ".cache" / "huggingface" / "hub"):
        snaps = root / ("models--" + CLIP_REPO.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            for d in sorted(snaps.iterdir(), reverse=True):
                if (d / "config.json").is_file():
                    return str(d)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--clip-dir", default="models/hf_cache/models--openai--clip-vit-large-patch14/snapshots")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))

    ext = CLIPFeatureExtractor(resolve_clip(args.clip_dir), device=args.device, input_size=224)
    glob_feats, tile_feats = [], []
    t0 = time.perf_counter()
    for i, s in enumerate(seqs):
        p = parse_image(str(data_dir / "images" / f"{s:03d}.png"), keep_full=True)
        if not p.ok:
            raise SystemExit(f"cannot parse {s}: {p.error}")
        glob_feats.append(ext(p.bgr))
        if not args.no_tiles:
            tile_feats.append(extract_tiled_features(ext, p.bgr_full))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(seqs)}", flush=True)

    g = np.stack(glob_feats)
    np.save(out_dir / "clip_feats.npy", g)
    suffix = "" if len(seqs) != 119 else "_119"
    np.save(out_dir / f"clip_feats{suffix}.npy", g)
    print(f"[clip] global {g.shape} in {time.perf_counter()-t0:.1f}s")
    if tile_feats:
        t = np.stack(tile_feats)
        np.save(out_dir / "clip_tiles.npy", t)
        np.save(out_dir / f"clip_tiles{suffix}.npy", t)
        print(f"[clip] tiles  {t.shape}")


if __name__ == "__main__":
    main()
