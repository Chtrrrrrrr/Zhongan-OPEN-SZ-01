#!/usr/bin/env python3
"""L1 baseline: L2-regularized logistic regression, stratified 5-fold x 5 seeds.

Reproduces the pilot experiment (manual ch.11.1): expected avg AUC ~0.916,
best CV accuracy ~85.7%. Also reports per-feature univariate AUC ranking.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from l1_forensics import FEATURE_NAMES  # noqa: E402


def load(path):
    seqs, y, X = [], [], []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            seqs.append(int(row[0]))
            y.append(1 if row[1] == "ai" else 0)
            X.append([float(v) for v in row[2:]])
    return np.array(seqs), np.array(y), np.array(X), header[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    args = ap.parse_args()

    seqs, y, X, names = load(args.features)
    print(f"samples={len(y)}  ai={y.sum()}  real={(1 - y).sum()}  dims={X.shape[1]}")

    aucs, accs = [], []
    for seed in range(5):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_acc = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
            acc = accuracy_score(y[te], prob >= 0.5)
            accs.append(acc)
            fold_acc.append(acc)
        print(f"seed {seed}: fold acc = {['%.3f' % a for a in fold_acc]}")

    aucs, accs = np.array(aucs), np.array(accs)
    print(f"\nmean AUC = {aucs.mean():.3f} (+- {aucs.std():.3f})")
    print(f"mean ACC = {accs.mean():.3f}   best fold ACC = {accs.max():.3f}")

    # univariate AUC per feature
    uni = []
    for j, n in enumerate(names):
        x = X[:, j]
        if np.std(x) < 1e-12:
            continue
        a = roc_auc_score(y, x)
        uni.append((max(a, 1 - a), n))
    uni.sort(reverse=True)
    print("\ntop-10 univariate AUC:")
    for a, n in uni[:10]:
        print(f"  {n:14s} {a:.3f}")


if __name__ == "__main__":
    main()
