#!/usr/bin/env python3
"""Fusion-rule ablation (manual ch.12.1).

Compares evidence-combination rules under the SAME out-of-fold protocol, with
thresholds always taken from the training-fold REAL scores only:

  combined_only : flag if the joint [L1+L1b+CLIP+tiles] head clears its threshold
  forensic_only : flag if the [L1+L1b] forensic head clears its threshold
  consensus     : flag only if BOTH heads clear their thresholds
                  (manual 6.2 R2: tighten the high tier to "score above threshold
                   AND at least two independent lines of evidence agree")
  avg_logit     : mean of the two heads' logits

Outputs reports/fusion_ablation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def thr_real(scores, real_idx, target_fpr):
    s = np.sort(scores[real_idx])[::-1]
    n = len(s)
    a = int(np.floor(target_fpr * n))
    if n == 0:
        return 0.0
    if a <= 0:
        return float(s[0]) + 1e-9
    if a >= n:
        return float(s[-1]) - 1e-9
    return float((s[a - 1] + s[a]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="reports/fusion_ablation.json")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    ws = ROOT.parent
    data_dir = ws / args.data_dir
    models_dir = ws / args.models_dir

    seqs, y = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)
    n = len(y)

    X1 = np.loadtxt(ws / args.features, delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(models_dir / "l1b_feats_119.npy")
    Fg = np.load(models_dir / "clip_feats_119.npy")
    Ft = np.load(models_dir / "clip_tiles_119.npy")
    XF = np.concatenate([X1, Xb], 1)
    XC = np.concatenate([X1, Xb, Fg, Ft], 1)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    of = np.zeros(n)
    oc = np.zeros(n)
    for tr, va in skf.split(XF, y):
        for X, out in ((XF, of), (XC, oc)):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            out[va] = logit(clf.predict_proba(sc.transform(X[va]))[:, 1])

    print(f"[auc] forensic={roc_auc_score(y, of):.4f}  combined={roc_auc_score(y, oc):.4f}")
    print("[hard cases]  seq label  forensic  combined")
    for s in (57, 58, 60, 67, 82, 29, 3, 9, 53):
        i = seqs.index(s)
        print(f"   {s:4d}  {'AI  ' if y[i] else 'REAL'}  {of[i]:8.2f}  {oc[i]:8.2f}")

    results = []
    for rule in ("combined_only", "forensic_only", "consensus", "avg_logit"):
        for tfpr in (0.02, 0.03, 0.05):
            pred = np.zeros(n, dtype=int)
            for tr, va in skf.split(XF, y):
                ri = tr[y[tr] == 0]
                tc, tf = thr_real(oc, ri, tfpr), thr_real(of, ri, tfpr)
                for i in va:
                    if rule == "combined_only":
                        pred[i] = 1 if oc[i] >= tc else 0
                    elif rule == "forensic_only":
                        pred[i] = 1 if of[i] >= tf else 0
                    elif rule == "consensus":
                        pred[i] = 1 if (oc[i] >= tc and of[i] >= tf) else 0
                    else:
                        pred[i] = 1 if (oc[i] + of[i]) / 2 >= (tc + tf) / 2 else 0
            row = {
                "rule": rule, "target_fpr": tfpr,
                "accuracy": round(float((pred == y).mean()), 4),
                "fpr": round(float(pred[y == 0].mean()), 4),
                "tpr": round(float(pred[y == 1].mean()), 4),
                "fp": int(pred[y == 0].sum()), "fn": int((1 - pred[y == 1]).sum()),
                "fp_seqs": [int(seqs[i]) for i in range(n) if y[i] == 0 and pred[i] == 1],
            }
            results.append(row)
            print(f"[{rule:14s} fpr<={tfpr}] ACC={row['accuracy']:.4f} FPR={row['fpr']:.4f} "
                  f"TPR={row['tpr']:.4f} FP={row['fp']} FN={row['fn']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "out-of-fold protocol; thresholds from training-fold real scores only",
        "oof_auc": {"forensic": round(float(roc_auc_score(y, of)), 4),
                    "combined": round(float(roc_auc_score(y, oc)), 4)},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
