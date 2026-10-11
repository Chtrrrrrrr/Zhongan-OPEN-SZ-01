#!/usr/bin/env python3
"""Confidence-calibration analysis (manual ch.13.1 置信度校准, ch.15.1 护栏与操作点指标).

Evidence produced -> reports/calibration.json :

  1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) with
     10 and 15 buckets, both equal-width (uniform, the standard definition) and
     equal-mass (quantile) binning, plus full reliability-diagram bin data
     (bin range / count / mean confidence / empirical positive rate / gap).
  2. The same quantities before and after temperature scaling, at three stages:
       raw        sigmoid(member logit), no temperature at all
       before     per-member temperature scaling only (this is the confidence the
                  deployed pipeline reports today, fitted on inner-CV training logits)
       after      member temperature + one global temperature on the fused logit,
                  fitted out-of-fold (4/5 of the folds fit, the 5th is scored)
     Brier score and log loss accompany every stage.
  3. Guardrail / operating-point metrics required by manual ch.15.1:
       TPR@FPR=1%   (primary criterion, ch.3.2) - interpolated ROC value and the
                    empirically attainable value (counts are integral: with 63 real
                    samples one false positive already costs 1/63 = 1.59%)
       FPR@TPR=95%
       Precision / Recall at the frozen deployed threshold
     reported for the deployed per-fold path, the frozen fixed-weight table and the
     five-route analysis table.
  4. Guardrail curves over an FPR grid and a TPR grid.
  5. Per-member calibration diagnostics (AUC, ECE before/after, Brier, log loss,
     fitted temperature per fold).

The out-of-fold score matrix comes from scripts/metrics_common.py, which reproduces
scripts/evaluate_system.py fold for fold; the deployed operating point is verified
against reports/metrics.json (consistency_check section).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
TPR_GRID = [0.50, 0.75, 0.90, 0.95, 0.99]
ECE_BOOTSTRAP = 2000


def fold_index(bundle) -> np.ndarray:
    """Per-sample outer-fold index (0-based)."""
    n = len(bundle["y"])
    out = np.zeros(n, dtype=int)
    for k, f in enumerate(bundle["folds"]):
        for s in f["holdout_seqs"]:
            out[int(s) - 1] = k
    return out


def perfold_fuse(bundle, a, source: dict, table: str) -> np.ndarray:
    """Fuse per fold.

    table="deployed"  -> the fold's admitted member set (clip_external only when its
                         training-fold AUC passed the admission rule)
    table=<name>      -> fixed member set from metrics_common weight tables
    """
    n = len(a["y"])
    out = np.zeros(n)
    for f in bundle["folds"]:
        idx = [int(s) - 1 for s in f["holdout_seqs"]]
        if table == "deployed":
            members = list(f["deployed_members"])
            wts = {m: mc.DEPLOYED_WEIGHTS[m] for m in members}
        else:
            wts = mc.WEIGHT_TABLES[table]
            members = [m for m in mc.MEMBER_ORDER if m in wts]
        probs = {m: np.asarray(source[m])[idx] for m in members}
        out[idx] = mc.fuse_probabilities(probs, wts, members)[0]
    return out


def fit_final_temperature(deployed_logit: np.ndarray, y: np.ndarray,
                         fold_idx: np.ndarray) -> dict:
    """Out-of-fold global temperature on the fused logit.

    For every outer fold the temperature is fitted on the other folds and applied to
    the held-out fold, so the reported ECE is not the ECE of a parameter fitted on
    the same samples it is scored on. The all-data fit is reported for reference only.
    """
    temps, pred = [], np.zeros(len(y))
    for k in np.unique(fold_idx):
        te = fold_idx == k
        tr = ~te
        sc = mc.TemperatureScaler().fit(deployed_logit[tr], y[tr])
        temps.append(round(float(sc.temperature), 4))
        pred[te] = sc.transform_proba(deployed_logit[te])
    insample = mc.TemperatureScaler().fit(deployed_logit, y)
    return {"per_fold": temps,
            "median": round(float(np.median(temps)), 4),
            "mean": round(float(np.mean(temps)), 4),
            "min": round(float(np.min(temps)), 4),
            "max": round(float(np.max(temps)), 4),
            "in_sample_reference": round(float(insample.temperature), 4),
            "note": "T>1 means the fused confidence was over-confident; T<1 the opposite",
            "oof_probability": pred}


def ece_bootstrap_ci(y, p, n_bins=10, draws=ECE_BOOTSTRAP, seed=17):
    """Percentile CI of ECE over bootstrap resamples (sampling noise of a 119-sample set)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    vals = []
    for _ in range(draws):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(mc.ece(y[idx], p[idx], n_bins)["ece"])
    if not vals:
        return None
    return {"bins": n_bins, "draws": len(vals),
            "ece_ci95": [round(float(np.percentile(vals, 2.5)), 4),
                         round(float(np.percentile(vals, 97.5)), 4)],
            "ece_std": round(float(np.std(vals)), 4)}


def guardrails(y, score, frozen_threshold=None):
    out = {
        "tpr@fpr=1%": mc.tpr_at_fpr(y, score, 0.01),
        # count granularity reference: 1/63, the smallest achievable non-zero FPR here
        "tpr@fpr=1.59%(1/63)": mc.tpr_at_fpr(y, score, 1 / 63),
        "fpr@tpr=95%": mc.fpr_at_tpr(y, score, 0.95),
        "tpr_curve_vs_fpr": {f"fpr={t:.3f}": mc.tpr_at_fpr(y, score, t)["tpr_interp"]
                             for t in FPR_GRID},
        "fpr_curve_vs_tpr": {f"tpr={t:.2f}": mc.fpr_at_tpr(y, score, t)["fpr"]
                             for t in TPR_GRID},
        "auc": mc.auc(y, score),
    }
    if frozen_threshold is not None:
        pred = (np.asarray(score) >= frozen_threshold).astype(int)
        out["fixed_threshold_operating_point"] = {
            "threshold": round(float(frozen_threshold), 6),
            "source": "models/fusion.json thr_high (frozen deployed threshold)",
            "note": "descriptive: one global frozen threshold applied to the whole OOF "
                    "score vector. The deployed pipeline derives a threshold per fold "
                    "from its training-fold real scores, which is why its confusion "
                    "matrix differs slightly from this row.",
            **mc.metrics_at(y, pred)}
    out["convention_note"] = (
        "tpr_interp linearly interpolates the ROC curve at the requested FPR; "
        "tpr_empirical is the best attainable TPR whose measured FPR is <= the target "
        "and therefore matches the step-function convention used by "
        "reports/metrics.json -> roc_frontier.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/calibration.json")
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    y, seqs, fold_idx = a["y"], a["seqs"], fold_index(bundle)

    fused_cfg = mc.load_deployed_config(ws / "models")
    print(f"[calib] frozen fusion config: {fused_cfg['source']} "
          f"weights={fused_cfg['weights']} thr_high={fused_cfg['thr_high']}", flush=True)

    raw_source = {m: mc.sigmoid(a["oof_logits"][m]) for m in mc.MEMBER_ORDER}
    cal_source = a["oof_calibrated"]

    deployed_raw = perfold_fuse(bundle, a, raw_source, "deployed")
    deployed_before = a["deployed_fused"]          # the deployed per-fold path itself
    recomputed = perfold_fuse(bundle, a, cal_source, "deployed")
    max_dev = float(np.max(np.abs(recomputed - deployed_before)))
    print(f"[calib] deployed fused recomputation max deviation = {max_dev:.2e}", flush=True)

    d_logit = mc.logit(deployed_before)
    final_temp = fit_final_temperature(d_logit, y, fold_idx)
    deployed_after = np.asarray(final_temp.pop("oof_probability"))

    # frozen fixed-weight table (models/fusion.json) over all folds: no per-fold
    # admission, used to separate "which members" from "how they are calibrated"
    stages = {}
    for name, p in (("raw", deployed_raw), ("before_member_temperature", deployed_before),
                    ("after_member_and_final_temperature", deployed_after)):
        entry = {
            "definition": {
                "raw": "sigmoid(fused logit) with no temperature at all",
                "before_member_temperature": "per-member temperature scaling only "
                                             "(the confidence the deployed pipeline reports)",
                "after_member_and_final_temperature": "per-member temperature + one global "
                                                      "temperature on the fused logit, "
                                                      "fitted out-of-fold",
            }[name],
            "ece_uniform_10": mc.ece(y, p, 10, "uniform")["ece"],
            "ece_uniform_15": mc.ece(y, p, 15, "uniform")["ece"],
            "ece_quantile_10": mc.ece(y, p, 10, "quantile")["ece"],
            "ece_quantile_15": mc.ece(y, p, 15, "quantile")["ece"],
            "mce_uniform_10": mc.ece(y, p, 10, "uniform")["mce"],
            "mce_uniform_15": mc.ece(y, p, 15, "uniform")["mce"],
            "brier": mc.brier(y, p),
            "log_loss": mc.log_loss(y, p),
            "mean_confidence": round(float(np.mean(p)), 4),
            "base_rate_positive": round(float(np.mean(y)), 4),
            "guardrails": guardrails(y, p),
        }
        entry["ece_ci95"] = {f"uniform_{b}": ece_bootstrap_ci(y, p, b) for b in (10, 15)}
        stages[name] = entry
        print(f"[calib] {name:38s} ECE10={entry['ece_uniform_10']:.4f} "
              f"ECE15={entry['ece_uniform_15']:.4f} Brier={entry['brier']:.4f} "
              f"logloss={entry['log_loss']:.4f}", flush=True)

    # alternative fusion tables (analysis configurations, not deployed)
    alt = {}
    for table in ("nominal_3route", "extended_5route", "equal_5route"):
        wts = mc.WEIGHT_TABLES[table]
        pr = perfold_fuse(bundle, a, cal_source, table)
        alt[table] = {
            "weights": wts,
            "members": [m for m in mc.MEMBER_ORDER if m in wts],
            "scope": "analysis configuration - not the deployed fusion",
            "ece_uniform_10": mc.ece(y, pr, 10)["ece"],
            "ece_uniform_15": mc.ece(y, pr, 15)["ece"],
            "brier": mc.brier(y, pr), "log_loss": mc.log_loss(y, pr),
            "guardrails": guardrails(y, pr),
        }
        print(f"[calib] {table:16s} AUC={alt[table]['guardrails']['auc']:.4f} "
              f"ECE10={alt[table]['ece_uniform_10']:.4f} "
              f"TPR@FPR=1%={alt[table]['guardrails']['tpr@fpr=1%']['tpr_interp']:.4f}",
              flush=True)

    # per-member diagnostics
    per_member = {}
    for m in mc.MEMBER_ORDER:
        raw_m = mc.sigmoid(a["oof_logits"][m])
        cal_m = a["oof_calibrated"][m]
        per_member[m] = {
            "definition": bundle["members"][m],
            "frozen_weight": fused_cfg["weights"].get(m, 0.0),
            "auc_oof": mc.auc(y, a["oof_logits"][m]),
            "ece_uniform_10_raw": mc.ece(y, raw_m, 10)["ece"],
            "ece_uniform_10_calibrated": mc.ece(y, cal_m, 10)["ece"],
            "ece_uniform_15_calibrated": mc.ece(y, cal_m, 15)["ece"],
            "brier_calibrated": mc.brier(y, cal_m),
            "log_loss_calibrated": mc.log_loss(y, cal_m),
            "temperature_per_fold": bundle["temperatures_per_fold"][m],
            "temperature_median": round(float(np.median(bundle["temperatures_per_fold"][m])), 4),
            "in_deployed_fusion": m in fused_cfg["weights"],
            "guardrails": guardrails(y, cal_m),
        }

    # reliability diagram data (uniform 10 and 15 for both stages)
    reliability = {
        "before": {f"uniform_{b}": mc.ece(y, deployed_before, b, "uniform") for b in (10, 15)},
        "after": {f"uniform_{b}": mc.ece(y, deployed_after, b, "uniform") for b in (10, 15)},
        "reading": "each bucket lists [lo, hi), sample count, mean predicted confidence, "
                   "empirical positive rate and their absolute gap; a well-calibrated "
                   "detector has gap ~= 0 in every bucket",
    }

    # consistency check against the shipped headline evidence
    ref = ws / "reports" / "metrics.json"
    consistency = {"reference": str(ref), "checked": ref.exists()}
    if ref.exists():
        refj = json.loads(ref.read_text(encoding="utf-8"))
        got = mc.metrics_at(y, a["deployed_pred"])
        want = refj.get("oof_overall", {})
        consistency["deployed_oof_reproduced"] = got
        consistency["reports_metrics_json_oof_overall"] = {
            k: want.get(k) for k in ("accuracy", "fpr", "tpr", "precision", "confusion",
                                     "three_tier")}
        consistency["matches"] = bool(
            got["accuracy"] == want.get("accuracy") and got["fpr"] == want.get("fpr")
            and got["tpr"] == want.get("tpr") and got["precision"] == want.get("precision")
            and got["confusion"] == want.get("confusion"))
        consistency["recomputation_max_abs_deviation"] = round(max_dev, 9)
        print(f"[calib] consistency with reports/metrics.json: {consistency['matches']}",
              flush=True)

    g_before = guardrails(y, deployed_before, fused_cfg["thr_high"])
    g_raw = guardrails(y, deployed_raw)
    g_after = guardrails(y, deployed_after, fused_cfg["thr_high"])

    payload = {
        "report": "calibration_analysis",
        "manual_sections": ["13.1 阈值标定与置信度校准", "15.1 护栏与操作点指标",
                            "3.2 误报硬护栏"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/calibration_analysis.py",
        "data": {
            "n_samples": len(y),
            "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
            "source": "out-of-fold scores from scripts/metrics_common.py "
                      "(same folds/seed/features as scripts/evaluate_system.py)",
            "protocol": bundle["protocol"],
        },
        "operating_point": {
            "score_definition": "fused probability = sigmoid(weighted mean of member "
                                "logits after per-member temperature scaling), "
                                "src/fusion.py FusionEngine.fuse",
            "deployed_frozen_weights": fused_cfg["weights"],
            "deployed_weights_source": fused_cfg["source"],
            "deployed_thresholds": {"thr_low": fused_cfg["thr_low"],
                                    "thr_high": fused_cfg["thr_high"]},
            "guardrails": g_before,
            "guardrails_raw_before_calibration": g_raw,
            "guardrails_after_final_temperature": g_after,
            "alternative_fusion_tables": alt,
        },
        "calibration": {
            "stages": stages,
            "final_temperature": final_temp,
            "headline_delta_ece": {
                # literal "before/after temperature scaling": raw member logits vs the
                # fully calibrated score
                "ece_uniform_10_raw_minus_after": round(
                    stages["raw"]["ece_uniform_10"]
                    - stages["after_member_and_final_temperature"]["ece_uniform_10"], 4),
                "ece_uniform_15_raw_minus_after": round(
                    stages["raw"]["ece_uniform_15"]
                    - stages["after_member_and_final_temperature"]["ece_uniform_15"], 4),
                # deployed confidence (member temperature only) vs the extra final temperature
                "ece_uniform_10_before_minus_after": round(
                    stages["before_member_temperature"]["ece_uniform_10"]
                    - stages["after_member_and_final_temperature"]["ece_uniform_10"], 4),
                "ece_uniform_15_before_minus_after": round(
                    stages["before_member_temperature"]["ece_uniform_15"]
                    - stages["after_member_and_final_temperature"]["ece_uniform_15"], 4),
                "brier_before_minus_after": round(
                    stages["before_member_temperature"]["brier"]
                    - stages["after_member_and_final_temperature"]["brier"], 4),
                "sign_convention": "positive => the 'after' stage is better calibrated",
            },
            "interpretation": {
                "final_temperature_improves_ece_vs_raw": bool(
                    stages["after_member_and_final_temperature"]["ece_uniform_10"]
                    < stages["raw"]["ece_uniform_10"]),
                "final_temperature_improves_ece_vs_member_temperature": bool(
                    stages["after_member_and_final_temperature"]["ece_uniform_10"]
                    < stages["before_member_temperature"]["ece_uniform_10"]),
                "note": "the bootstrap CIs of the two stages overlap, so on 119 samples "
                        "neither ordering is statistically established; the per-member "
                        "table shows where temperature scaling is clearly necessary "
                        "(the frozen clip_external / npr channels are severely "
                        "over-confident before scaling)",
            },
            "per_member": per_member,
            "reliability_diagram": reliability,
        },
        "consistency_check": consistency,
        "headline": {
            "n_samples": int(len(y)),
            "n_ai": int((y == 1).sum()),
            "n_real": int((y == 0).sum()),
            "fused_auc": mc.auc(y, deployed_before),
            "tpr_at_fpr_1pct_interp": g_before["tpr@fpr=1%"]["tpr_interp"],
            "tpr_at_fpr_1pct_empirical": g_before["tpr@fpr=1%"]["tpr_empirical"],
            "fpr_at_tpr_95pct": g_before["fpr@tpr=95%"]["fpr"],
            "precision_at_frozen_threshold": (
                g_before["fixed_threshold_operating_point"]["precision"]),
            "recall_at_frozen_threshold": (
                g_before["fixed_threshold_operating_point"]["tpr"]),
            "ece10_before": stages["before_member_temperature"]["ece_uniform_10"],
            "ece10_after": stages["after_member_and_final_temperature"]["ece_uniform_10"],
            "ece15_before": stages["before_member_temperature"]["ece_uniform_15"],
            "ece15_after": stages["after_member_and_final_temperature"]["ece_uniform_15"],
            "ece10_raw_no_temperature": stages["raw"]["ece_uniform_10"],
            "brier_before": stages["before_member_temperature"]["brier"],
            "brier_after": stages["after_member_and_final_temperature"]["brier"],
            "final_temperature_median": final_temp["median"],
            "reproduces_reports_metrics_json": consistency.get("matches"),
        },
        "limitations": [
            "The 119 samples are the only labelled domain data; the OOF split is not an "
            "independent test set and the class ratio (56 AI / 63 real) differs from the "
            "stated 7:3 evaluation ratio.",
            "TPR@FPR=1% is granularity-limited: with 63 real samples, one false positive "
            "costs 1.59% FPR, so the exact 1% point is only reachable by interpolation "
            "(tpr_interp); tpr_empirical is the attainable value at FPR<=1%.",
            "ECE on 119 samples is noisy; bootstrap 95% CIs are reported next to every "
            "ECE value and the intervals overlap between the before/after stages.",
            "The final temperature is fitted out-of-fold, but on the same 119 samples "
            "used for evaluation - the estimate is therefore optimistic in absolute "
            "terms while still being an unbiased comparison of the two stages.",
        ],
    }
    mc.write_json(out_path, payload)

    print(f"\n[calib] deployed guardrails: "
          f"TPR@FPR=1%={payload['operating_point']['guardrails']['tpr@fpr=1%']['tpr_interp']} "
          f"FPR@TPR=95%={payload['operating_point']['guardrails']['fpr@tpr=95%']['fpr']}")
    print(f"[calib] ECE10 before={stages['before_member_temperature']['ece_uniform_10']} "
          f"after={stages['after_member_and_final_temperature']['ece_uniform_10']}")


if __name__ == "__main__":
    main()
