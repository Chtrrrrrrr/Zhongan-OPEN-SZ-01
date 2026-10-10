#!/usr/bin/env python3
"""Error analysis for the 119 competition samples (manual ch.14.2).

Produces an out-of-fold score table plus the forensic profile of the hardest
samples (highest-scoring REAL, lowest-scoring AI), which is what drives the next
feature/model iteration instead of blind model stacking.

Usage: python scripts/diagnose_errors.py [--csv reports/error_analysis.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l1b_domain import FEATURE_NAMES_B   # noqa: E402
from l1_forensics import FEATURE_NAMES   # noqa: E402


def load_all(ws: Path):
    seqs, y = [], []
    with open(ws / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)

    rows = []
    with open(ws / "features.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append([float(v) for v in row[2:]])
    X1 = np.array(rows)
    feat_names = header[2:]

    parts = [X1]
    names = list(feat_names)
    for fn, nm in ((ws / "models" / "l1b_feats_119.npy", FEATURE_NAMES_B),
                   (ws / "models" / "clip_feats_119.npy", None),
                   (ws / "models" / "clip_tiles_119.npy", None)):
        if fn.exists():
            arr = np.load(fn)
            parts.append(arr)
            names += list(nm) if nm else [f"{fn.stem}_{i}" for i in range(arr.shape[1])]
    Z = np.concatenate(parts, axis=1)
    return seqs, y, Z, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT.parent))
    ap.add_argument("--csv", default="reports/error_analysis.csv")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    ws = Path(args.workspace)

    seqs, y, Z, names = load_all(ws)
    print(f"[data] {len(y)} samples, feature dim {Z.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(C=1.0, max_iter=4000).fit(sc.transform(Z[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(Z[va]))[:, 1]

    from sklearn.metrics import roc_auc_score
    print(f"[score] OOF AUC = {roc_auc_score(y, oof):.4f}")

    real = sorted([(seqs[i], float(oof[i])) for i in range(len(y)) if y[i] == 0],
                  key=lambda t: -t[1])
    ai = sorted([(seqs[i], float(oof[i])) for i in range(len(y)) if y[i] == 1],
                key=lambda t: t[1])

    man_path = ws / "data" / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    recs = man.get("records", man) if isinstance(man, dict) else man
    bym = {int(r["seq"]): r for r in recs} if isinstance(recs, list) else {}

    def profile(seq, p, label):
        i = seqs.index(seq)
        m = bym.get(seq, {})
        idx = {n: k for k, n in enumerate(names)}
        d = {
            "seq": seq, "label": label, "oof_p": round(p, 4),
            "file": m.get("file", f"data/images/{seq:03d}.png"),
            "orig_wh": f"{int(Z[i][idx['ow']])}x{int(Z[i][idx['oh']])}",
        }
        for k in ("nidat", "hasSrgb", "hasExif", "hasItxt", "blockRatio", "msResRatio"):
            if k in idx:
                d[k] = round(float(Z[i][idx[k]]), 4)
        for k in ("b_flatNoise", "b_nearWhiteFrac", "b_noiseRatio", "b_illumGradient",
                  "b_sharpCV", "b_grayLevelCount", "b_colorfulness"):
            if k in idx:
                d[k] = round(float(Z[i][idx[k]]), 4)
        return d

    print(f"\n=== hardest REAL samples (highest AI score) ===")
    for s, p in real[:args.top]:
        print("  ", profile(s, p, "REAL"))
    print(f"\n=== hardest AI samples (lowest AI score) ===")
    for s, p in ai[:args.top]:
        print("  ", profile(s, p, "AI"))

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "label", "oof_score", "difficulty"])
        for s, p in real:
            w.writerow([s, "real", f"{p:.6f}", "hard" if p > 0.5 else ""])
        for s, p in ai:
            w.writerow([s, "ai", f"{p:.6f}", "hard" if p < 0.5 else ""])
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
