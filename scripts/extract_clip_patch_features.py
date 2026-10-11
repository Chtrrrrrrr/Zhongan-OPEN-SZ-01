#!/usr/bin/env python3
"""Patch-token anomaly statistics from the frozen CLIP ViT - an independent clue family.

Motivation (manual ch.3.5 / 8.4, and the error analysis in reports/verify_rounds.json):
the deployed evidence is global-pooled CLIP plus tile-level CLIP. Both reduce a patch
grid to a pooled vector, so a *localised* inconsistency - the signature of inpainting,
splicing, or a locally re-rendered region - is averaged away. The frozen ViT already
computes one token per 16x16 patch; this script keeps the patch grid and turns it into
training-free statistics:

  for each patch i:  cos_i = <t_i, mean_j t_j> / (|t_i| |mean_j t_j|)
  features: mean / std / min / max / p05 / p95 of cos, the share of patches below
            mean-2*std, the spatial autocorrelation of the cos map at lag 1, the
            quadrant contrast between the 4 image quadrants, and the same statistics on
            the residual norm |t_i - mean| relative to |t_i|

Nothing here is trained: the features are deterministic functions of the frozen
backbone, so they cannot overfit the 119 samples.

Outputs (default models/):
  clip_patch.npy        [N, K] patch-token statistics
  clip_patch_names.json feature names

Usage: python scripts/extract_clip_patch_features.py --data-dir data --out-dir models
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l0_input import parse_image                                  # noqa: E402
from l2_clip import CLIPFeatureExtractor                          # noqa: E402
from tile_clip_sweep import resolve_clip                          # noqa: E402

FEATURE_NAMES: list[str] = [
    "pt_cosMean", "pt_cosStd", "pt_cosMin", "pt_cosMax",
    "pt_cosP05", "pt_cosP95", "pt_cosLowFrac", "pt_cosHighFrac",
    "pt_cosAutocorr1", "pt_cosQuadrantSpread",
    "pt_resMean", "pt_resStd", "pt_resMax", "pt_resLowFrac",
    "pt_normMean", "pt_normStd",
]


def patch_stats(extractor: CLIPFeatureExtractor, bgr: np.ndarray) -> np.ndarray:
    """[T, D] patch tokens -> K training-free statistics."""
    x = extractor.preprocess(bgr).unsqueeze(0).to(extractor.device)
    with torch.no_grad():
        out = extractor.model(x, output_hidden_states=True)
    h = out.hidden_states[-1][0]                      # [T, D] (includes CLS)
    t = h[1:]                                         # drop CLS -> patch grid
    t = torch.nn.functional.normalize(t.float(), dim=-1)
    mu = t.mean(dim=0)
    mu = torch.nn.functional.normalize(mu, dim=-1, eps=1e-8)
    cos = (t @ mu)                                   # [T] cosine to the patch mean
    res = (t - mu).norm(dim=-1)                      # [T] residual norm
    norm = t.norm(dim=-1)
    c = cos.detach().cpu().numpy().astype(np.float64)
    r = res.detach().cpu().numpy().astype(np.float64)
    nrm = norm.detach().cpu().numpy().astype(np.float64)

    T = int(np.sqrt(c.size))
    grid = c[: T * T].reshape(T, T) if T * T == c.size else None

    thr_lo = c.mean() - 2 * c.std()
    thr_hi = c.mean() + 2 * c.std()
    feats = [c.mean(), c.std(), c.min(), c.max(),
             float(np.percentile(c, 5)), float(np.percentile(c, 95)),
             float((c < thr_lo).mean()), float((c > thr_hi).mean())]
    if grid is not None and T >= 4:
        a = grid - grid.mean()
        denom = float((a * a).sum()) + 1e-12
        ac1 = float((a[:, :-1] * a[:, 1:]).sum() / denom)
        half = T // 2
        quads = [grid[:half, :half].mean(), grid[:half, half:].mean(),
                 grid[half:, :half].mean(), grid[half:, half:].mean()]
        feats += [ac1, float(np.std(quads))]
    else:
        feats += [0.0, 0.0]
    r_lo = r.mean() - 2 * r.std()
    feats += [r.mean(), r.std(), r.max(), float((r > r.mean() + 2 * r.std()).mean()),
              nrm.mean(), nrm.std()]
    return np.array(feats, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--clip-dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="pre-scale bound for the patch pass (native detail, bounded cost)")
    args = ap.parse_args()

    import cv2
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))

    ext = CLIPFeatureExtractor(resolve_clip(args.clip_dir), device=args.device, input_size=224)
    rows = []
    t0 = time.perf_counter()
    for i, s in enumerate(seqs):
        p = parse_image(str(data_dir / "images" / f"{s:03d}.png"), keep_full=True)
        if not p.ok:
            raise SystemExit(f"cannot parse {s}: {p.error}")
        img = p.bgr_full
        h, w = img.shape[:2]
        if max(h, w) > args.max_side:
            sc = args.max_side / max(h, w)
            img = cv2.resize(img, (max(8, int(w * sc)), max(8, int(h * sc))),
                             interpolation=cv2.INTER_AREA)
        # the ViT sees a 224 square, so the patch grid is 16x16 for whatever we feed it
        rows.append(patch_stats(ext, img))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(seqs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    A = np.stack(rows)
    np.save(out_dir / "clip_patch.npy", A)
    (out_dir / "clip_patch_names.json").write_text(
        json.dumps(FEATURE_NAMES, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[patch] {A.shape} in {time.perf_counter()-t0:.1f}s -> {out_dir/'clip_patch.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
