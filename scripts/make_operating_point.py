"""Emit the honest operating-point evidence file into the repo.

Locks in three independently reproducible operating points on the archived
out-of-fold scores, identifies which one the repository's own stated rule
(fp_budget=2 on pooled pure-real scores) actually implies, and reports the
brief-mandated 7:3 resampling.

This file exists because audit-1 (work/audit/audit-1.md D1) showed the DEPLOYED
threshold realises 92.44% out-of-fold accuracy while the project publishes 95.94%.
Reproducible by:  python work/improve/operating_point.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Portable: repo/ = parent of this file's directory (scripts/ -> repo/). Do NOT hardcode an
# absolute machine path here -- a reviewer clones to an arbitrary location and the script
# must still work (and must write inside THEIR clone, not someone else's working copy).
REPO = Path(__file__).resolve().parent.parent
WS = REPO.parent

dep = json.loads((REPO / "models" / "deployment_metrics.json").read_text(encoding="utf-8"))
S = np.asarray(dep["_oof_fused_scores"], dtype=float)
seqs = np.asarray(dep["_oof_seqs"], dtype=int)
if seqs.ndim > 1:
    seqs = seqs[0]
y = np.array([1 if s <= 56 else 0 for s in seqs])
R, N = S.shape
N_AI, N_REAL = int(y.sum()), int((1 - y).sum())


def per_round(thr):
    acc, fpr, tpr, fps, fns, accs = [], [], [], [], [], []
    for r in range(R):
        pred = (S[r] >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        accs.append(round((tp + tn) / N, 4)); acc.append((tp + tn) / N)
        fpr.append(fp / N_REAL); tpr.append(tp / N_AI); fps.append(fp); fns.append(fn)
    return {"accuracy_mean": round(float(np.mean(acc)), 4), "accuracy_per_round": accs,
            "fpr_mean": round(float(np.mean(fpr)), 4),
            "tpr_mean": round(float(np.mean(tpr)), 4),
            "fp_per_round": fps, "fn_per_round": fns}


def pooled(thr):
    p = S.mean(axis=0)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"accuracy": round((tp + tn) / N, 4), "fpr": round(fp / N_REAL, 4),
            "fpr_count": f"{fp}/{N_REAL}", "tpr": round(tp / N_AI, 4),
            "fp_seqs": [int(seqs[i]) for i in np.where((pred == 1) & (y == 0))[0]],
            "fn_seqs": [int(seqs[i]) for i in np.where((pred == 0) & (y == 1))[0]]}


pooled_scores = S.mean(axis=0)
real_desc = np.sort(pooled_scores[y == 0])[::-1]
# IMPORTANT: keep full float64 precision. Rounding this to 6 dp (e.g. -0.424307)
# makes it GREATER than the 3rd largest pooled real score (-0.4243068333333333) and
# therefore admits THREE positives, silently changing the metric. audit-2 N5 caught this.
thr_rule = float(real_desc[2]) + 1e-12

POINTS = {
    "deployed_fusion_json": 1.561879,
    "per_fold_median_logit": 0.956386,
    "rule_implied_pooled_2fp": thr_rule,
}

res = {}
for name, thr in POINTS.items():
    res[name] = {"threshold_logit": float(thr),
                 "threshold_logit_repr": repr(float(thr)),
                 "threshold_probability": float(1 / (1 + np.exp(-thr))),
                 "per_round": per_round(thr), "pooled": pooled(thr)}

# brief-mandated 7:3 resampling
rng = np.random.default_rng(11)
real_idx = np.where(y == 0)[0]; ai_idx = np.where(y == 1)[0]
sim = {}
for name, thr in POINTS.items():
    accs, fprs = [], []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(real_idx, 70, replace=True),
                              rng.choice(ai_idx, 30, replace=True)])
        yy = y[idx]
        pred = (pooled_scores[idx] >= thr).astype(int)
        accs.append((pred == yy).mean()); fprs.append(pred[yy == 0].mean())
    accs = np.array(accs); fprs = np.array(fprs)
    sim[name] = {"accuracy_mean": round(float(accs.mean()), 4),
                 "accuracy_ci95": [round(float(np.percentile(accs, 2.5)), 4),
                                   round(float(np.percentile(accs, 97.5)), 4)],
                 "fpr_mean": round(float(fprs.mean()), 4),
                 "p_accuracy_ge_095": round(float((accs >= 0.95).mean()), 3),
                 "p_fpr_le_003": round(float((fprs <= 0.03).mean()), 3)}

payload = {
    "report": "operating_point_analysis",
    "purpose": "Establish which reproducible operating point the repository's own "
               "fp_budget=2 rule implies, and how far the DEPLOYED frozen threshold is "
               "from it. Written in response to audit-1 D1.",
    "reproduce": "python scripts/make_operating_point.py",
    "data_source": "repo/models/deployment_metrics.json::_oof_fused_scores "
                   "(6 rounds x 119 out-of-fold fused LOGITS) + _oof_seqs",
    "n_samples": N, "class_balance": {"ai": N_AI, "real": N_REAL},
    "ratio_note": "the brief specifies ~7:3 (real:fake); the shipped set is 47:53 "
                  "(56 ai / 63 real), so both are reported",
    "threshold_space": "logits; probability = sigmoid(logit). Thresholds are stored at "
                       "full float64 precision -- rounding the rule-implied threshold to "
                       "6 dp (-0.424307) makes it LARGER than the 3rd largest pooled real "
                       "score (-0.4243068333333333) and admits 3 positives instead of 2.",
    "fp_budget": 2,
    "rule_definition": "admit at most 2 real samples -> threshold must be strictly below "
                       "the 3rd largest pooled pure-real fused score, i.e. "
                       "thr in (4th_largest, 3rd_largest]; implementing it as "
                       "3rd_largest + 1e-12 puts the admitted count at exactly 2.",
    "operating_points": res,
    "official_ratio_7_3_resampling": {
        "draws": 2000, "n_real": 70, "n_fake": 30, "seed": 11, "results": sim},
    "finding": {
        "headline_claim": "README/results.md publish 95.94% accuracy / 2.12% FPR",
        "deployed_realisation": res["deployed_fusion_json"]["per_round"]["accuracy_mean"],
        "rule_implied_pooled": res["rule_implied_pooled_2fp"]["pooled"]["accuracy"],
        "conclusion": "The DEPLOYED frozen threshold (thr_high=0.826623) is NOT the point "
                      "the repository's own stated rule (fp_budget=2 on pooled pure-real "
                      "scores) implies. The rule implies -0.424307, which realises 97.48% "
                      "pooled / 96.08% per-round accuracy at the SAME FP count (2/63) and "
                      "strictly dominates the deployed threshold. The published 95.94% "
                      "matches neither and is not reproducible from repo/models/.",
        "fpr_requirement": "NOTE: 2/63 = 3.17% still EXCEEDS the required <=3%. On this "
                           "set satisfying <=3% requires at most 1 FP (1.59%), and every "
                           "threshold reaching 0 FP destroys recall (accuracy ~53%), so "
                           "<=3% is not met by any reproducible operating point.",
    },
    "caveat": "The pooled score is a diagnostic aggregate, not a deployable per-item "
              "rule. Per-round application of the rule-implied threshold yields FP counts "
              "of 2-5, i.e. the bound holds in aggregate, not uniformly per round.",
}

out = REPO / "reports" / "operating_point_analysis.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[save] {out}")
print(f"  deployed per-round acc = {res['deployed_fusion_json']['per_round']['accuracy_mean']}")
print(f"  rule-implied pooled acc = {res['rule_implied_pooled_2fp']['pooled']['accuracy']}")
print(f"  rule-implied per-round acc = {res['rule_implied_pooled_2fp']['per_round']['accuracy_mean']}")
print(f"  7:3 P(acc>=.95): deployed={sim['deployed_fusion_json']['p_accuracy_ge_095']} "
      f"rule={sim['rule_implied_pooled_2fp']['p_accuracy_ge_095']}")
