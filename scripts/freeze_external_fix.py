#!/usr/bin/env python3
"""Freeze the defect fix: refit the decision heads on official + external material.

Background: docs/known-issue-scene-photo-fpr.md. The competition "real" class is nearly
all phone photos of paper, so the head flagged real claim SCENE photos 100%. This script
freezes the remedy measured by scripts/train_with_external_reals.py:

  training rows = official 119 (weight `--official-weight`) + external labelled corpus
                  (weight 1), external material chosen by `--corpus-mode`
  threshold     = the same rule as before (order statistic over the official out-of-fold
                  real scores at the false-positive budget), so the competition operating
                  point keeps its meaning; `--threshold pooled` additionally shifts it to
                  cover external reals, and the artifact records which one shipped

Artifacts written to --models-dir:
  forensic_lr.joblib, combined_lr.joblib, fusion.json   (same paths as before)
  fix_metrics.json                                       (what changed and by how much)

Usage
  python scripts/freeze_external_fix.py --corpus models/external_corpus \
      --official-weight 300 --corpus-mode both --rounds 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_spec import FeatureSpec                                        # noqa: E402
from tile_clip_sweep import _fit_temperature, _fused, _fpr_threshold        # noqa: E402
from train_with_external_reals import (WEIGHTS, load_corpus, load_official,  # noqa: E402
                                       run_variant, summarize, _fit_head_w)


def thresholds_from(result, y_o, y_e, fp_budget):
    """Official-only threshold and official+external pooled threshold (both logits)."""
    all_thr = np.array(result["member_thresholds"])
    thr_official = float(np.median(all_thr))
    pooled = np.concatenate([result["oof"][:, y_o == 0].mean(axis=0),
                             result["external_fused"][y_e == 0]])
    s = np.sort(pooled)[::-1]
    k = max(0, min(fp_budget, len(s) - 1))
    thr_pooled = float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)
    return thr_official, thr_pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--corpus", default="models/external_corpus")
    ap.add_argument("--corpus-mode", default="both", choices=("both", "reals", "fakes"))
    ap.add_argument("--official-weight", type=float, default=300.0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--threshold", default="official", choices=("official", "pooled"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import joblib
    from sklearn.model_selection import StratifiedKFold

    md = ROOT / args.models_dir
    spec = FeatureSpec.load(md / "feature_spec.json")
    Xf_o, Xc_o = load_official(md, spec)
    y_o = []
    with open(Path(args.data_dir) / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y_o.append(1 if r["label"] == "ai" else 0)
    y_o = np.array(y_o)
    Xf_e, Xc_e, y_e, folders = load_corpus(Path(args.corpus), spec)
    if args.corpus_mode == "reals":
        sel = y_e == 0
    elif args.corpus_mode == "fakes":
        sel = y_e == 1
    else:
        sel = np.ones(len(y_e), dtype=bool)
    Xf_e, Xc_e, y_e = Xf_e[sel], Xc_e[sel], y_e[sel]
    print(f"[official] n={len(y_o)} (ai={int((y_o==1).sum())})")
    print(f"[corpus]   n={len(y_e)} (real={int((y_e==0).sum())} fake={int((y_e==1).sum())}) "
          f"mode={args.corpus_mode} official_weight={args.official_weight}")

    # ---- 1. honest verification of the frozen configuration ----
    res = run_variant("freeze", Xf_o, Xc_o, y_o, Xf_e, Xc_e, y_e, args.rounds,
                      args.n_folds, args.fp_budget, 42, args.corpus_mode,
                      sample_weight=args.official_weight)
    res["thr_official"], res["thr_pooled"] = thresholds_from(res, y_o, y_e, args.fp_budget)
    summary = summarize(res, y_o, y_e, args.fp_budget, args.rounds)
    print(f"[verify] official OOF acc={summary['official_oof']['accuracy_mean']} "
          f"[{summary['official_oof']['accuracy_min']},{summary['official_oof']['accuracy_max']}] "
          f"fpr={summary['official_oof']['fpr_mean']} tpr={summary['official_oof']['tpr_mean']}")
    print(f"[verify] external real FPR: @official={summary['external_real_fpr_at_official_threshold']:.2%} "
          f"@pooled={summary['external_real_fpr_at_pooled_threshold']:.2%} (n={summary['external_real_n']})")
    print(f"[verify] official at pooled threshold: "
          f"{json.dumps(summary['official_oof_at_pooled_threshold'])}")

    # ---- 2. refit on everything for deployment ----
    Xf_all = np.concatenate([Xf_o, Xf_e], axis=0)
    Xc_all = np.concatenate([Xc_o, Xc_e], axis=0)
    y_all = np.concatenate([y_o, y_e])
    w_all = np.concatenate([np.full(len(y_o), float(args.official_weight)),
                            np.ones(len(y_e))])
    head_f = _fit_head_w(Xf_all, y_all, w_all)
    head_c = _fit_head_w(Xc_all, y_all, w_all)

    # temperatures + low/high thresholds on the official inner-CV scale (unchanged rule)
    inner_splits = max(3, args.n_folds - 1)
    inner = {"forensic": np.zeros(len(y_o)), "combined": np.zeros(len(y_o))}
    iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=777)
    for itr, iva in iskf.split(Xc_o, y_o):
        hf = _fit_head_w(Xf_o[itr], y_o[itr])
        hc = _fit_head_w(Xc_o[itr], y_o[itr])
        inner["forensic"][iva] = hf[1].decision_function(hf[0].transform(Xf_o[iva]))
        inner["combined"][iva] = hc[1].decision_function(hc[0].transform(Xc_o[iva]))
    temps = {m: _fit_temperature(inner[m], y_o) for m in inner}
    fused_official = _fused(inner, temps, WEIGHTS)
    thr_official_logit = _fpr_threshold(fused_official[y_o == 0], args.fp_budget)
    thr_low_logit = float(np.median(fused_official[y_o == 0]))

    def to_prob(z):
        return float(1.0 / (1.0 + np.exp(-z)))

    thr_high_p = to_prob(res["thr_pooled"] if args.threshold == "pooled" else thr_official_logit)
    thr_low_p = to_prob(thr_low_logit)
    if thr_low_p >= thr_high_p:
        thr_low_p = float(np.percentile([to_prob(v) for v in fused_official[y_o == 0]], 25))

    joblib.dump({"scaler": head_f[0], "clf": head_f[1],
                 "features": f"{spec.name}: forensic (+external corpus, w={args.official_weight})"},
                md / "forensic_lr.joblib")
    joblib.dump({"scaler": head_c[0], "clf": head_c[1],
                 "features": f"{spec.name}: forensic|clip_global|tiles (+external corpus, "
                             f"w={args.official_weight})"},
                md / "combined_lr.joblib")
    fusion = json.loads((md / "fusion.json").read_text(encoding="utf-8"))
    fusion.update({
        "weights": {m: WEIGHTS[m] for m in ("combined", "forensic")},
        "thr_low": round(thr_low_p, 6),
        "thr_high": round(thr_high_p, 6),
        "threshold_space": "probability",
        "temperatures": temps,
        "training_corpus": {
            "official": int(len(y_o)), "official_weight": args.official_weight,
            "external_real": int((y_e == 0).sum()), "external_fake": int((y_e == 1).sum()),
            "corpus_mode": args.corpus_mode, "corpus_dir": str(args.corpus),
        },
        "calibration": {
            "protocol": "threshold = order statistic over the official nested-CV out-of-fold "
                        "real scores at the false-positive budget"
                        + (" (pooled with external reals)" if args.threshold == "pooled" else ""),
            "fp_budget": args.fp_budget,
            "official_threshold_logit": round(thr_official_logit, 6),
            "pooled_threshold_logit": round(res["thr_pooled"], 6),
            "shipped": args.threshold,
        },
    })
    (md / "fusion.json").write_text(json.dumps(fusion, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    out = Path(args.out) if args.out else md / "fix_metrics.json"
    payload = {
        "report": "external_fix_freeze",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue": "docs/known-issue-scene-photo-fpr.md",
        "configuration": {
            "corpus_mode": args.corpus_mode, "official_weight": args.official_weight,
            "threshold_shipped": args.threshold, "rounds": args.rounds,
            "n_folds": args.n_folds, "fp_budget": args.fp_budget,
        },
        "verification": summary,
        "shipped_thresholds": {"low": round(thr_low_p, 6), "high": round(thr_high_p, 6),
                               "high_logit": round(res["thr_pooled"] if args.threshold == "pooled"
                                                   else thr_official_logit, 6)},
        "temperatures": temps,
        "deployed_heads": {
            "rows": int(len(y_all)),
            "official_weight": args.official_weight,
            "official_rows": int(len(y_o)), "external_rows": int(len(y_e)),
        },
        "caveats": [
            "External material is used for TRAINING only; every accuracy number above is an "
            "out-of-fold prediction on the 119 competition samples.",
            "The external corpus includes Synthbuster real photos paired with its generated "
            "counterparts; it is not a claim-material dataset.",
            "Two real populations with different score levels cannot share one exact operating "
            "point, so the artifact records both candidate thresholds and which one shipped.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[freeze] thr_low={thr_low_p:.6f} thr_high={thr_high_p:.6f} "
          f"(logit {res['thr_pooled'] if args.threshold=='pooled' else thr_official_logit:.4f}, "
          f"space={args.threshold}) temps={ {k: round(v,3) for k,v in temps.items()} }")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
