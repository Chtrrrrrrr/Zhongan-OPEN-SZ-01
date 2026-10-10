#!/usr/bin/env python3
"""Batch L1 feature extraction over the dataset -> features.csv"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from l0_input import parse_image          # noqa: E402
from l1_forensics import extract_l1_features, FEATURE_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    data = Path(args.data_dir)
    rows = []
    with open(data / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["seq"]), r["filename"], r["label"]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "label"] + FEATURE_NAMES)
        for i, (seq, fn, label) in enumerate(rows):
            p = parse_image(fn, keep_full=False)
            if not p.ok:
                print(f"[WARN] {fn}: {p.error}")
                continue
            feat = extract_l1_features(p)
            w.writerow([seq, label] + [f"{feat[k]:.6g}" for k in FEATURE_NAMES])
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(rows)}")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
