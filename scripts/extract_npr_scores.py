#!/usr/bin/env python3
"""Score every sample with the NPR branch and report its standalone value.

Produces models/npr_scores_119.npy (logits) plus a short ablation summary:
  - NPR OOF AUC on the 119
  - error-overlap with the existing members (for the manual's retention rule)

Retention rule (manual ch.12.1): keep a new member only if it adds stable
discriminative power AND its errors are not highly overlapping with the others.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT.parent))
    ap.add_argument("--npr", default="models/npr.pt")
    ap.add_argument("--out", default="reports/npr_ablation.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ws = Path(args.workspace)
    md = ws / "models"

    npr_path = md / "npr.pt"
    if not npr_path.exists():
        raise SystemExit(f"NPR weights not found: {npr_path} (run train_npr.py)")

    seqs, y = [], []
    with open(ws / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)

    from l0_input import parse_image
    from l2_npr import NPRFeatureExtractor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    ext = NPRFeatureExtractor(str(npr_path), args.device)
    print("[npr] scoring 119 samples ...")
    scores = []
    for s in seqs:
        p = parse_image(str(ws / "data" / "images" / f"{s:03d}.png"), keep_full=False)
        scores.append(ext.logit(p.bgr))
    scores = np.array(scores)
    np.save(md / "npr_scores_119.npy", scores)
    auc = float(roc_auc_score(y, scores))
    print(f"[npr] standalone AUC on the 119 = {auc:.4f}")

    # pairwise error overlap vs existing members (at their own OOF thresholds)
    result = {"npr_standalone_auc": round(auc, 4)}
    try:
        X1 = np.loadtxt(ws / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
        Xb = np.load(md / "l1b_feats_119.npy")
        Fg = np.load(md / "clip_feats_119.npy")
        Ft = np.load(md / "clip_tiles_119.npy")
        XF = np.concatenate([X1, Xb], 1)
        XC = np.concatenate([X1, Xb, Fg, Ft], 1)

        def oof_logit(Z):
            out = np.zeros(len(y))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr, va in skf.split(Z, y):
                sc = StandardScaler().fit(Z[tr])
                clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Z[tr]), y[tr])
                p = clf.predict_proba(sc.transform(Z[va]))[:, 1]
                out[va] = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
            return out

        comb = oof_logit(XC)
        fore = oof_logit(XF)

        # does adding NPR improve the OOF AUC of the combined score?
        def rank(a):
            return np.argsort(np.argsort(a)) / len(a)

        for w in (0.0, 0.1, 0.2, 0.3):
            fused = (1 - w) * rank(comb) + w * rank(scores)
            result[f"combined_rank_plus_npr_w{w}"] = round(float(roc_auc_score(y, fused)), 4)
            # error sets at a fixed 3% FPR operating point (global, descriptive only)
            thr = np.quantile(fused[y == 0], 1 - 0.03)
            pred = (fused >= thr).astype(int)
            result[f"acc@fpr3_w{w}"] = round(float((pred == y).mean()), 4)
            result[f"fp_w{w}"] = int(pred[y == 0].sum())
            result[f"fn_w{w}"] = int((1 - pred[y == 1]).sum())

        result["corr_npr_combined"] = round(float(np.corrcoef(scores, comb)[0, 1]), 4)
        result["corr_npr_forensic"] = round(float(np.corrcoef(scores, fore)[0, 1]), 4)
    except Exception as e:
        result["ablation_error"] = str(e)

    for k, v in result.items():
        print(f"  {k}: {v}")
    out = ws / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
