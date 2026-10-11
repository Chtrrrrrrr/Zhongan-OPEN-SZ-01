#!/usr/bin/env python3
"""Threshold calibration, bootstrap stability and threshold sensitivity (manual ch.13.1).

Evidence produced -> reports/threshold_stability.json :

  1. Calibration set description (size, source distribution, proxy render/capture mix).
  2. Point estimate of every reported threshold: the order statistic admitting at most
     `fp_budget` real false positives (the deployed rule), the strict 1% variant, the
     low-tier (median) threshold, and the empirical-FPR reading of each.
  3. Bootstrap (default 2000 resamples, >= 1000 required by the manual) of the pure-real
     calibration scores: median, mean, standard deviation and 95% percentile interval of
     the threshold, plus the bootstrap distribution of the empirical FPR of the frozen
     threshold.
  4. Threshold sensitivity: how far the threshold drifts when
       - the target FPR moves (with the local slope in score units per 1% FPR),
       - the score distribution shifts location (additive drift),
       - the score distribution changes spread (multiplicative drift),
       - the calibration-set size shrinks (20 / 30 / 40 / 50 out of 63).
  5. Repeated-calibration fluctuation: 500 random calibration/hold-out halvings of the
     real scores, reporting the spread of the calibrated threshold and the empirical FPR
     measured on the held-out half (manual 表 13-1 step 6: verify FPR on an independent
     pure-real hold-out).
  6. Explicit granularity caveat: with 63 real samples one false positive is 1.59% FPR,
     so a 1% target is not reachable and a 3% target means exactly 2 false positives.

The scores come from scripts/metrics_common.py (out-of-fold, identical folds to
scripts/evaluate_system.py). They are a *proxy* pure-real calibration set: the 119
competition samples are the only labelled domain data available, so no fully
independent real calibration set exists in this workspace. That limitation is stated
in the report itself and matters when the bootstrap interval width is read.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

TARGET_FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
LOCATION_SHIFTS = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
SCALE_FACTORS = [0.80, 0.90, 1.0, 1.10, 1.20]
SUBSAMPLE_SIZES = [20, 30, 40, 50, 63]
BOOTSTRAP_DRAWS = 2000
SUBSAMPLE_DRAWS = 500
HALVING_DRAWS = 500


def boot_stat(values, fn, draws=BOOTSTRAP_DRAWS, seed=101):
    """Bootstrap percentile summary of `fn` applied to resamples of `values`."""
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=np.float64)
    out = np.array([fn(rng.choice(v, len(v), replace=True)) for _ in range(draws)],
                   dtype=np.float64)
    return {
        "draws": draws,
        "median": round(float(np.median(out)), 6),
        "mean": round(float(out.mean()), 6),
        "std": round(float(out.std(ddof=1)), 6),
        "ci95_percentile": [round(float(np.percentile(out, 2.5)), 6),
                            round(float(np.percentile(out, 97.5)), 6)],
        "width_ci95": round(float(np.percentile(out, 97.5) - np.percentile(out, 2.5)), 6),
        "min": round(float(out.min()), 6),
        "max": round(float(out.max()), 6),
    }


def empirical_fpr(real_scores, threshold):
    s = np.asarray(real_scores, dtype=np.float64)
    return float((s >= threshold).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--fp-budget", type=int, default=mc.FP_BUDGET_DEFAULT)
    ap.add_argument("--boot-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/threshold_stability.json")
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    y = a["y"]
    fused_cfg = mc.load_deployed_config(ws / "models")

    # ---------------- calibration set -------------------------------------------------
    real_deployed = a["deployed_fused"][y == 0]
    real_extended, _ = mc.fuse_probabilities(
        {m: a["oof_calibrated"][m] for m in mc.MEMBER_ORDER}, mc.EXTENDED_WEIGHTS)
    real_extended = real_extended[y == 0]
    n_real = len(real_deployed)

    l1b = np.load(ws / "models" / "l1b_feats_119.npy")
    nw = l1b[y == 0, 0]
    fz = l1b[y == 0, 3]
    calib_set = {
        "n_real": int(n_real),
        "n_ai": int((y == 1).sum()),
        "source": "out-of-fold fused scores of the real samples in the 119-sample "
                  "competition set (scripts/metrics_common.py, deployed per-fold path)",
        "score_statistics": {
            "min": round(float(real_deployed.min()), 6),
            "p25": round(float(np.percentile(real_deployed, 25)), 6),
            "median": round(float(np.median(real_deployed)), 6),
            "p75": round(float(np.percentile(real_deployed, 75)), 6),
            "p90": round(float(np.percentile(real_deployed, 90)), 6),
            "p95": round(float(np.percentile(real_deployed, 95)), 6),
            "max": round(float(real_deployed.max()), 6),
            "mean": round(float(real_deployed.mean()), 6),
            "std": round(float(real_deployed.std(ddof=1)), 6),
            "iqr": round(float(np.percentile(real_deployed, 75)
                               - np.percentile(real_deployed, 25)), 6),
        },
        "proxy_source_mix": {
            "note": "proxy labels derived from L1b features; not human annotation "
                    "(see reports/stratified_analysis.json for the rule)",
            "digital_render_proxy": int(((nw >= 0.05) & (fz <= 0.495)).sum()),
            "camera_capture_proxy": int(((nw < 0.05) & (fz >= 0.557)).sum()),
            "ambiguous": int((~(((nw >= 0.05) & (fz <= 0.495))
                                | ((nw < 0.05) & (fz >= 0.557)))).sum()),
        },
        "independence": "THE 119 SAMPLES ARE THE ONLY LABELLED DOMAIN DATA: real scores are "
                        "out-of-fold but not drawn from an independent pure-real "
                        "calibration corpus, so every interval below underestimates the "
                        "true uncertainty of a deployment threshold",
    }
    print(f"[thr] calibration set n_real={n_real} median={calib_set['score_statistics']['median']} "
          f"max={calib_set['score_statistics']['max']}", flush=True)

    # ---------------- point estimates -------------------------------------------------
    def point(scores, target_fpr, fp_budget=None):
        thr = mc.fpr_capped_threshold(scores, target_fpr, fp_budget)
        return {"threshold": round(float(thr), 6),
                "rule": (f"order statistic admitting at most {fp_budget} false positive(s)"
                         if fp_budget is not None else
                         f"order statistic admitting at most floor({target_fpr}*n_real) "
                         f"false positives"),
                "empirical_fpr_on_calibration_set": round(empirical_fpr(scores, thr), 4),
                "allowed_fp": int(np.floor(target_fpr * len(scores))) if fp_budget is None
                else int(fp_budget),
                "achieved_fp": int((np.asarray(scores) >= thr).sum())}

    points = {
        "high_tier_deployed_rule_fp_budget_2": point(real_deployed, args.target_fpr,
                                                     args.fp_budget),
        "high_tier_target_1pct_fp_budget_0": point(real_deployed, 0.01, 0),
        "high_tier_quantile_97pct": {
            "threshold": round(float(np.percentile(real_deployed, 97)), 6),
            "rule": "empirical 97th percentile (initial estimate only; the manual forbids "
                    "treating '99th percentile == FPR 1%' as an identity)",
            "empirical_fpr_on_calibration_set": round(
                empirical_fpr(real_deployed, np.percentile(real_deployed, 97)), 4),
        },
        "low_tier_median": {
            "threshold": round(float(np.median(real_deployed)), 6),
            "rule": "median of the pure-real calibration scores (the deployed thr_low rule)",
            "empirical_fpr_on_calibration_set": round(
                empirical_fpr(real_deployed, np.median(real_deployed)), 4),
        },
        "frozen_deployed_thresholds": {
            "thr_high": fused_cfg["thr_high"], "thr_low": fused_cfg["thr_low"],
            "source": fused_cfg["source"],
            "empirical_fpr_at_thr_high": round(
                empirical_fpr(real_deployed, fused_cfg["thr_high"]), 4),
            "note": "the value actually shipped in models/fusion.json, derived by "
                    "evaluate_system.py on the same protocol",
        },
    }
    print(f"[thr] deployed-rule threshold = "
          f"{points['high_tier_deployed_rule_fp_budget_2']['threshold']} "
          f"(frozen {fused_cfg['thr_high']})", flush=True)

    # ---------------- bootstrap -------------------------------------------------------
    boot = {}
    for name, target, budget in (("absolute_budget_2_deployed_rule", args.target_fpr,
                                  args.fp_budget),
                                 ("absolute_budget_0_strict_1pct", 0.01, 0),
                                 ("scaled_budget_floor_3pct_of_n", 0.03,
                                  int(np.floor(0.03 * n_real)))):
        boot[name] = {
            "rule": f"resample the {n_real} real scores with replacement, recompute the "
                    f"order statistic with a false-positive budget of {budget}",
            "point_estimate": round(float(mc.fpr_capped_threshold(real_deployed, target,
                                                                 budget)), 6),
            "bootstrap": boot_stat(real_deployed,
                                   lambda s, b=budget: mc.fpr_capped_threshold(s, target, b),
                                   draws=args.boot_draws, seed=103 + budget),
        }
        b = boot[name]["bootstrap"]
        print(f"[thr] bootstrap {name}: median={b['median']} ci95={b['ci95_percentile']} "
              f"width={b['width_ci95']}", flush=True)

    # bootstrap of the empirical FPR produced by the frozen threshold
    fpr_boot = boot_stat(
        (real_deployed >= fused_cfg["thr_high"]).astype(float),
        lambda s: float(np.mean(s)), draws=args.boot_draws, seed=211)
    # bootstrap over the *scores* is the relevant one for a threshold read at a fixed level
    thr_fixed = float(fused_cfg["thr_high"])
    fpr_at_frozen = boot_stat(real_deployed,
                              lambda s: empirical_fpr(s, thr_fixed),
                              draws=args.boot_draws, seed=212)
    threshold_distribution = {
        "empirical_fpr_of_frozen_threshold": {
            "frozen_threshold": round(thr_fixed, 6),
            "empirical_fpr_calibration_set": round(empirical_fpr(real_deployed, thr_fixed), 4),
            "bootstrap_ci95": fpr_at_frozen["ci95_percentile"],
            "bootstrap_mean": fpr_at_frozen["mean"],
            "bootstrap_std": fpr_at_frozen["std"],
            "target_fpr": 0.03,
            "verdict": ("95% CI upper bound above the 3% acceptance level"
                        if fpr_at_frozen["ci95_percentile"][1] > 0.03
                        else "95% CI stays at or below the 3% acceptance level"),
        },
        "fraction_of_bootstrap_draws_with_zero_fp": {
            "rule": "share of resamples of the real calibration scores in which the frozen "
                    "threshold admits no false positive at all",
            "value": round(float(np.mean(
                [empirical_fpr(rng2 := np.random.RandomState(300 + i).choice(
                    real_deployed, n_real, replace=True), thr_fixed) == 0
                 for i in range(500)])), 4),
        },
    }

    # ---------------- sensitivity -----------------------------------------------------
    target_curve = {}
    for t in TARGET_FPR_GRID:
        thr = float(mc.fpr_capped_threshold(real_deployed, t))
        target_curve[f"target_fpr={t:.3f}"] = {
            "threshold": round(thr, 6),
            "allowed_fp": int(np.floor(t * n_real)),
            "empirical_fpr": round(empirical_fpr(real_deployed, thr), 4),
        }
    slope_1pct = round(float(
        (float(mc.fpr_capped_threshold(real_deployed, 0.02))
         - float(mc.fpr_capped_threshold(real_deployed, 0.01))) / 0.01), 6)
    slope_3pct = round(float(
        (float(mc.fpr_capped_threshold(real_deployed, 0.04))
         - float(mc.fpr_capped_threshold(real_deployed, 0.03))) / 0.01), 6)

    iqr = float(np.percentile(real_deployed, 75) - np.percentile(real_deployed, 25))
    med = float(np.median(real_deployed))
    location = []
    for d in LOCATION_SHIFTS:
        shifted = real_deployed + d
        thr = float(mc.fpr_capped_threshold(shifted, args.target_fpr, args.fp_budget))
        base = points["high_tier_deployed_rule_fp_budget_2"]["threshold"]
        location.append({
            "shift": d, "threshold": round(thr, 6),
            "delta_threshold": round(thr - base, 6),
            "drift_ratio": round((thr - base) / d, 4) if d else None,
            "delta_in_iqr_units": round(d / iqr, 4) if iqr else None,
        })
    scale = []
    for s in SCALE_FACTORS:
        scaled = med + (real_deployed - med) * s
        thr = float(mc.fpr_capped_threshold(scaled, args.target_fpr, args.fp_budget))
        base = points["high_tier_deployed_rule_fp_budget_2"]["threshold"]
        scale.append({"factor": s, "threshold": round(thr, 6),
                      "delta_threshold": round(thr - base, 6)})
    subsample = {}
    rng = np.random.RandomState(404)
    for n in SUBSAMPLE_SIZES:
        draws = np.array([float(mc.fpr_capped_threshold(
            real_deployed[rng.choice(n_real, n, replace=False)], args.target_fpr,
            min(args.fp_budget, max(0, n - 1)))) for _ in range(SUBSAMPLE_DRAWS)])
        subsample[f"n={n}"] = {
            "threshold_mean": round(float(draws.mean()), 6),
            "threshold_std": round(float(draws.std(ddof=1)), 6),
            "threshold_ci95": [round(float(np.percentile(draws, 2.5)), 6),
                               round(float(np.percentile(draws, 97.5)), 6)],
            "draws": SUBSAMPLE_DRAWS,
        }
    loc_spread = [abs(r["delta_threshold"]) for r in location if r["shift"]]
    sensitivity = {
        "target_fpr_curve": target_curve,
        "target_fpr_slope_score_units_per_1pct_fpr": {
            "at_1pct": slope_1pct, "at_3pct": slope_3pct,
            "sign_convention": "negative = a looser (higher) FPR target lowers the threshold",
            "reading": "how far the threshold moves per percentage point of target FPR; "
                       "a steep slope means the operating point is fragile to FPR "
                       "re-negotiation, and a large magnitude here means the real-score "
                       "distribution has a wide gap where the threshold must sit",
        },
        "location_shift": {
            "rule": "add a constant to every real calibration score, then recompute the "
                    "threshold (a distribution whose quantiles drift upward/downward)",
            "iqr": round(iqr, 6),
            "table": location,
            "max_abs_delta_threshold": round(max(loc_spread), 6) if loc_spread else None,
            "interpretation": "the order-statistic threshold tracks a pure location shift "
                              "one-for-one (drift_ratio ~1); what changes the *false "
                              "positive count* is the shape, not the shift",
        },
        "scale_change": {
            "rule": "stretch the real scores around their median by a factor, then "
                    "recompute the threshold",
            "table": scale,
            "max_abs_delta_threshold": round(max(abs(r["delta_threshold"]) for r in scale), 6),
        },
        "calibration_set_size": {
            "rule": "subsample the real calibration scores without replacement and "
                    "recompute the threshold (how much a smaller calibration set moves it)",
            "table": subsample,
            "reading": "the threshold spread grows as the calibration set shrinks - the "
                       "reason the manual asks for a calibration-set size to be reported "
                       "next to every threshold",
        },
    }
    print(f"[thr] sensitivity: slope@1%={slope_1pct} slope@3%={slope_3pct} "
          f"max|Δthr| under shift={sensitivity['location_shift']['max_abs_delta_threshold']}",
          flush=True)

    # ---------------- repeated calibration / hold-out FPR -----------------------------
    # Two budget policies are run side by side, because the deployed rule uses an
    # ABSOLUTE budget (2 false positives per evaluation batch) whereas a scaled budget
    # (floor(target * n)) is the transferable analogue on a half-sized calibration set.
    rng = np.random.RandomState(909)
    policies = {"absolute_budget_2": lambda n_cal: args.fp_budget,
                "scaled_budget_floor_3pct": lambda n_cal: int(np.floor(args.target_fpr * n_cal))}
    repeated = {
        "protocol": f"{HALVING_DRAWS} random half/half splits of the {n_real} real scores: "
                    f"the threshold is calibrated on one half and the empirical FPR is "
                    f"measured on the other half (manual 表 13-1 step 6); both halves are "
                    f"out-of-fold scores of real samples",
        "policies": {},
    }
    for pname, budget_of in policies.items():
        cal_thr, hold_fpr, hold_fpr_frozen = [], [], []
        for _ in range(HALVING_DRAWS):
            perm = rng.permutation(n_real)
            calib, hold = real_deployed[perm[:n_real // 2]], real_deployed[perm[n_real // 2:]]
            b = min(budget_of(len(calib)), len(calib) - 1)
            t = float(mc.fpr_capped_threshold(calib, args.target_fpr, b))
            cal_thr.append(t)
            hold_fpr.append(empirical_fpr(hold, t))
            hold_fpr_frozen.append(empirical_fpr(hold, thr_fixed))
        cal_thr, hold_fpr = np.array(cal_thr), np.array(hold_fpr)
        repeated["policies"][pname] = {
            "budget_rule": ("2 false positives regardless of calibration size (deployed rule)"
                            if pname.startswith("absolute")
                            else "floor(3% * calibration-set size)"),
            "calibrated_threshold": {
                "mean": round(float(cal_thr.mean()), 6),
                "std": round(float(cal_thr.std(ddof=1)), 6),
                "ci95": [round(float(np.percentile(cal_thr, 2.5)), 6),
                         round(float(np.percentile(cal_thr, 97.5)), 6)],
                "full_set_threshold": points["high_tier_deployed_rule_fp_budget_2"]["threshold"],
            },
            "holdout_empirical_fpr": {
                "mean": round(float(hold_fpr.mean()), 4),
                "median": round(float(np.median(hold_fpr)), 4),
                "ci95": [round(float(np.percentile(hold_fpr, 2.5)), 4),
                         round(float(np.percentile(hold_fpr, 97.5)), 4)],
                "share_above_3pct": round(float((hold_fpr > 0.03).mean()), 4),
            },
            "holdout_empirical_fpr_of_frozen_threshold": {
                "mean": round(float(np.mean(hold_fpr_frozen)), 4),
                "ci95": [round(float(np.percentile(hold_fpr_frozen, 2.5)), 4),
                         round(float(np.percentile(hold_fpr_frozen, 97.5)), 4)],
                "share_above_3pct": round(float((np.array(hold_fpr_frozen) > 0.03).mean()), 4),
            },
        }
        r = repeated["policies"][pname]
        print(f"[thr] repeated calibration {pname}: thr std={r['calibrated_threshold']['std']} "
              f"holdout FPR mean={r['holdout_empirical_fpr']['mean']} "
              f"ci95={r['holdout_empirical_fpr']['ci95']}", flush=True)
    repeated["reading"] = (
        "half-size calibration halves have a different real-score composition, so the "
        "calibrated threshold moves; the hold-out FPR spread is the honest statement of "
        "how well a 3% target transfers. The absolute-budget policy allows 2 false "
        "positives on a 31-sample half (6.5%), which is why its hold-out FPR reads higher "
        "than the scaled-budget policy - both are reported rather than picking the "
        "flattering one.")

    payload = {
        "report": "threshold_stability",
        "manual_sections": ["13.1 阈值标定与置信度校准", "表 13-1 步骤 3-6",
                            "表 13-2 阈值标定记录", "17.1 多次标定的波动"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/threshold_stability.py",
        "calibration_set": calib_set,
        "point_estimates": points,
        "bootstrap": boot,
        "frozen_threshold_fpr": threshold_distribution,
        "sensitivity": sensitivity,
        "repeated_calibration": repeated,
        "granularity": {
            "n_real": int(n_real),
            "one_false_positive_equals_fpr": round(1 / n_real, 4),
            "two_false_positives_equal_fpr": round(2 / n_real, 4),
            "note": "the 1% guardrail cannot be resolved exactly at this sample size; the "
                    "3% acceptance level corresponds to exactly 2 false positives out of 63",
        },
        "headline": {
            "calibration_set_n_real": int(n_real),
            "threshold_deployed_rule": points["high_tier_deployed_rule_fp_budget_2"]["threshold"],
            "threshold_frozen_shipped": fused_cfg["thr_high"],
            "threshold_bootstrap_median": boot["absolute_budget_2_deployed_rule"]["bootstrap"]["median"],
            "threshold_bootstrap_ci95": boot["absolute_budget_2_deployed_rule"]["bootstrap"]["ci95_percentile"],
            "threshold_bootstrap_width": boot["absolute_budget_2_deployed_rule"]["bootstrap"]["width_ci95"],
            "bootstrap_draws": boot["absolute_budget_2_deployed_rule"]["bootstrap"]["draws"],
            "empirical_fpr_frozen_threshold": threshold_distribution[
                "empirical_fpr_of_frozen_threshold"]["empirical_fpr_calibration_set"],
            "empirical_fpr_frozen_threshold_ci95": threshold_distribution[
                "empirical_fpr_of_frozen_threshold"]["bootstrap_ci95"],
            "holdout_fpr_mean_scaled_budget": repeated["policies"][
                "scaled_budget_floor_3pct"]["holdout_empirical_fpr"]["mean"],
            "holdout_fpr_mean_absolute_budget": repeated["policies"][
                "absolute_budget_2"]["holdout_empirical_fpr"]["mean"],
            "sensitivity_max_threshold_drift_location_shift": sensitivity["location_shift"][
                "max_abs_delta_threshold"],
            "sensitivity_max_threshold_drift_scale_change": sensitivity["scale_change"][
                "max_abs_delta_threshold"],
            "sensitivity_slope_at_3pct": slope_3pct,
        },
        "limitations": [
            "The calibration set is the out-of-fold real scores of the 119 competition "
            "samples, not an independent pure-real corpus (none exists in this workspace); "
            "bootstrap intervals therefore describe sampling noise within this set only.",
            "Real samples dominate the calibration set with digital/render-like material; "
            "the difficult negatives (low light, glare, blur, creases, screenshots) named "
            "in manual ch.13.1 are not separately available.",
            "The threshold rule (order statistic with an absolute false-positive budget) is "
            "the deployed rule; other rules (e.g. fitting the score distribution) would "
            "produce different sensitivities.",
        ],
    }
    mc.write_json(out_path, payload)


if __name__ == "__main__":
    main()
