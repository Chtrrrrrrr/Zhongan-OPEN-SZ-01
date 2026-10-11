#!/usr/bin/env python3
"""Choose the fusion weight set and official up-weight for the external-material fix.

The defect (docs/known-issue-scene-photo-fpr.md) is only half fixed by adding external
material: the `combined` member learns that real scene photos are real, but the cheap
`forensic` member keeps firing positive on them (median logit +4.4 on the 41 real claim
photos). Since scene-photo safety is a business requirement, the fusion weights have to
be chosen with BOTH populations in view:

  * official 119 out-of-fold accuracy / FPR  (the competition metric, must not regress)
  * claim-photo false-positive rate on the 41 network-collected real claim photos
    (the business metric that was 100% before)

Both are reported per configuration; nothing is selected on the official held-out folds
beyond what the standard protocol already does (heads fit on training folds only,
threshold from the nested inner CV).

Usage
  python scripts/tune_fusion_for_reals.py --corpus models/external_corpus \
      --official-weights 100,300,1000 --weight-sets w60_25,w70_30,w80_20,combined_only
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

from feature_spec import FeatureSpec                                       # noqa: E402
from train_with_external_reals import (load_corpus, load_official,          # noqa: E402
                                       run_variant, summarize)

WEIGHT_SETS = {
    "w60_25": {"combined": 0.60, "forensic": 0.25},      # archived deployment weights
    "w70_30": {"combined": 0.70, "forensic": 0.30},
    "w80_20": {"combined": 0.80, "forensic": 0.20},
    "w90_10": {"combined": 0.90, "forensic": 0.10},
    "combined_only": {"combined": 1.0},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--corpus", default="models/external_corpus")
    ap.add_argument("--corpus-mode", default="both")
    ap.add_argument("--official-weights", default="100,300,1000")
    ap.add_argument("--weight-sets", default="w60_25,w70_30,w80_20,w90_10,combined_only")
    ap.add_argument("--claim-folder", default="real",
                    help="corpus folder holding the real claim photos (business negatives)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--out", default="reports/fusion_tuning_reals.json")
    args = ap.parse_args()

    md = ROOT / args.models_dir
    spec = FeatureSpec.load(md / "feature_spec.json")
    Xf_o, Xc_o = load_official(md, spec)
    y_o = []
    with open(Path(args.data_dir) / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y_o.append(1 if r["label"] == "ai" else 0)
    y_o = np.array(y_o)
    Xf_e, Xc_e, y_e, folders = load_corpus(Path(args.corpus), spec)
    claim = folders == args.claim_folder
    print(f"[official] n={len(y_o)}  [corpus] n={len(y_e)} "
          f"(real={int((y_e==0).sum())} fake={int((y_e==1).sum())})  "
          f"[claim photos] n={int(claim.sum())}")

    rows = []
    for ws_name in [s.strip() for s in args.weight_sets.split(",") if s.strip()]:
        if ws_name not in WEIGHT_SETS:
            raise SystemExit(f"unknown weight set {ws_name!r}; choose from {list(WEIGHT_SETS)}")
        weights = WEIGHT_SETS[ws_name]
        for ow in [float(x) for x in args.official_weights.split(",") if x.strip()]:
            t0 = time.perf_counter()
            res = run_variant(f"{ws_name}_ow{int(ow)}", Xf_o, Xc_o, y_o, Xf_e, Xc_e, y_e,
                              args.rounds, args.n_folds, args.fp_budget, 42,
                              args.corpus_mode, sample_weight=ow, weights=weights)
            all_thr = np.array(res["member_thresholds"])
            res["thr_official"] = float(np.median(all_thr))
            pooled = np.concatenate([res["oof"][:, y_o == 0].mean(axis=0),
                                     res["external_fused"][y_e == 0]])
            s = np.sort(pooled)[::-1]
            k = max(0, min(args.fp_budget, len(s) - 1))
            res["thr_pooled"] = float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)
            summary = summarize(res, y_o, y_e, args.fp_budget, args.rounds)
            thr = res["thr_official"]
            claim_fused = res["external_fused"][claim]
            row = {
                "weight_set": ws_name, "weights": weights, "official_weight": ow,
                "official_acc_mean": summary["official_oof"]["accuracy_mean"],
                "official_acc_min": summary["official_oof"]["accuracy_min"],
                "official_acc_max": summary["official_oof"]["accuracy_max"],
                "official_fpr_mean": summary["official_oof"]["fpr_mean"],
                "official_tpr_mean": summary["official_oof"]["tpr_mean"],
                "claim_fpr_at_official_threshold": round(float((claim_fused >= thr).mean()), 4),
                "claim_flagged": int((claim_fused >= thr).sum()),
                "claim_n": int(claim.sum()),
                "external_real_fpr": summary["external_real_fpr_at_official_threshold"],
                "external_auc": summary["external_auc"],
                "ceiling_at_budget": summary["accuracy_ceiling_at_budget"],
                "threshold_official": round(thr, 6),
                "threshold_pooled": round(res["thr_pooled"], 6),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(f"  {ws_name:14s} ow={int(ow):5d} | official acc={row['official_acc_mean']:.4f} "
                  f"[{row['official_acc_min']:.4f},{row['official_acc_max']:.4f}] "
                  f"fpr={row['official_fpr_mean']:.4f} tpr={row['official_tpr_mean']:.4f} | "
                  f"claim FPR={row['claim_fpr_at_official_threshold']:.1%} "
                  f"({row['claim_flagged']}/{row['claim_n']}) | "
                  f"ext real FPR={row['external_real_fpr']:.2%} auc={row['external_auc']}",
                  flush=True)

    rows.sort(key=lambda r: (-r["official_acc_mean"], r["claim_fpr_at_official_threshold"]))
    payload = {
        "report": "fusion_tuning_reals",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue": "docs/known-issue-scene-photo-fpr.md",
        "protocol": {
            "evaluation": "official 119 out-of-fold accuracy; claim-photo FPR is a holdout "
                          "measurement under the official threshold",
            "rounds": args.rounds, "n_folds": args.n_folds, "fp_budget": args.fp_budget,
            "corpus_mode": args.corpus_mode,
        },
        "rows": rows,
        "best_by_official_accuracy": rows[0] if rows else None,
    }
    ranked_by_safety = sorted(rows, key=lambda r: (r["claim_fpr_at_official_threshold"],
                                                   -r["official_acc_mean"]))
    payload["best_by_claim_safety"] = ranked_by_safety[0] if ranked_by_safety else None
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[best official accuracy] {json.dumps(payload['best_by_official_accuracy'])}")
    print(f"[best claim safety]     {json.dumps(payload['best_by_claim_safety'])}")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
