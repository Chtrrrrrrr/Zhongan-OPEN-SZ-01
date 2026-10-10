#!/usr/bin/env python3
"""Feature-set ablation (manual ch.8.2 / 12.1).

Question: the pilot baseline's strongest single features (IDAT count, sRGB chunk,
file size) are PNG *encoder* fingerprints. The manual classifies them as fragile
auxiliary evidence that a single re-encode destroys. Do they help or hurt the
operating point we actually care about (low-FPR region)?

Variants
  full         : L1(47) + L1b(20) + CLIP global + CLIP tiles
  no_container : drop the 9 container features (nidat/hasSrgb/hasExif/hasItxt/
                 hasGama/hasChrm/bytes/ow/oh) from every head
  pixel_only   : L1 pixel features only (no container) + L1b, no CLIP

Reports OOF AUC and the ROC frontier (TPR at FPR = 1/2/3/5/10%), which is the
metric that decides the accuracy/FPR trade-off.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONTAINER_IDX = list(range(37, 46))   # nidat..oh in the 47-dim L1 vector


def frontier(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    out = {}
    for t in (0.01, 0.02, 0.03, 0.05, 0.10):
        i = max(0, min(int(np.searchsorted(fpr, t, side="right")) - 1, len(tpr) - 1))
        out[f"tpr@fpr={t:.2f}"] = round(float(tpr[i]), 4)
    return out


def best_acc(y, s, max_fpr):
    best = 0.0
    for t in np.unique(s):
        p = (s >= t).astype(int)
        if p[y == 0].mean() <= max_fpr:
            best = max(best, float((p == y).mean()))
    return round(best, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="reports/feature_ablation.json")
    args = ap.parse_args()
    ws = ROOT.parent
    md = ws / args.models_dir

    seqs, y = [], []
    with open(ws / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)

    X1 = np.loadtxt(ws / args.features, delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(md / "l1b_feats_119.npy")
    Fg = np.load(md / "clip_feats_119.npy")
    Ft = np.load(md / "clip_tiles_119.npy")
    keep = [i for i in range(X1.shape[1]) if i not in CONTAINER_IDX]

    variants = {
        "full": np.concatenate([X1, Xb, Fg, Ft], 1),
        "no_container": np.concatenate([X1[:, keep], Xb, Fg, Ft], 1),
        "pixel_only": np.concatenate([X1[:, keep], Xb], 1),
        "container_plus_l1b": np.concatenate([X1[:, CONTAINER_IDX], Xb], 1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, Z in variants.items():
        oof = np.zeros(len(y))
        for tr, va in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Z[tr]), y[tr])
            p = clf.predict_proba(sc.transform(Z[va]))[:, 1]
            oof[va] = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        auc = float(roc_auc_score(y, oof))
        fr = frontier(y, oof)
        res = {"dim": int(Z.shape[1]), "oof_auc": round(auc, 4), "frontier": fr,
               "best_acc@fpr<=0.03": best_acc(y, oof, 0.03),
               "best_acc@fpr<=0.032": best_acc(y, oof, 0.0318),
               "best_acc@fpr<=0.05": best_acc(y, oof, 0.05)}
        results[name] = res
        print(f"[{name:20s}] dim={Z.shape[1]:5d} AUC={auc:.4f} frontier={fr} "
              f"bestACC@FPR3%={res['best_acc@fpr<=0.03']} "
              f"@3.2%={res['best_acc@fpr<=0.032']} @5%={res['best_acc@fpr<=0.05']}")

    out = Path(ws / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "out-of-fold, 5-fold stratified, random_state=42",
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
