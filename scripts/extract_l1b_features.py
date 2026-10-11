#!/usr/bin/env python3
"""Batch-extract L1 + L1b forensic features for all samples.

Outputs:
  features.csv              - 47-dim L1 features (compat with the pilot baseline)
  <models>/l1b_feats.npy    - 20-dim camera-vs-render features (+ feature_names json)

Usage: python scripts/extract_l1b_features.py --data-dir data --out-dir models
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l0_input import parse_image                                    # noqa: E402
from l1_forensics import extract_l1_features, FEATURE_NAMES         # noqa: E402
from l1b_domain import extract_l1b_features, FEATURE_NAMES_B, feature_vector_b  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # older label files carry the path under "file", the current one under
            # "filename" - accept both instead of failing on the header spelling
            rel = r.get("file") or r.get("filename") or ""
            rows.append((int(r["seq"]), rel if rel.endswith(".png") else r["filename"],
                         r["label"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_vecs, l1b_vecs = [], []
    with open(args.features, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "label"] + FEATURE_NAMES)
        for i, (seq, fn, label) in enumerate(rows):
            path = fn if Path(fn).exists() else str(data_dir / "images" / f"{seq:03d}.png")
            p = parse_image(path, keep_full=True)
            if not p.ok:
                print(f"[WARN] {path}: {p.error}")
                continue
            f1 = extract_l1_features(p)
            fb = extract_l1b_features(p)
            w.writerow([seq, label] + [f"{f1[k]:.6g}" for k in FEATURE_NAMES])
            l1_vecs.append([f1[k] for k in FEATURE_NAMES])
            l1b_vecs.append(feature_vector_b(fb))
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(rows)}")

    np.save(out_dir / "l1_feats.npy", np.array(l1_vecs, dtype=np.float64))
    np.save(out_dir / "l1b_feats.npy", np.array(l1b_vecs, dtype=np.float64))
    # the evaluator uses the 119-sample filename by convention
    if len(l1b_vecs) == 119:
        np.save(out_dir / "l1b_feats_119.npy", np.array(l1b_vecs, dtype=np.float64))
    (out_dir / "l1b_feature_names.json").write_text(
        __import__("json").dumps(FEATURE_NAMES_B, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {args.features} + {out_dir}/l1b_feats.npy ({len(l1b_vecs)} samples)")


if __name__ == "__main__":
    main()
