#!/usr/bin/env python3
"""Multi-member combination and error-correlation experiment (manual ch.12.1).

Evidence produced -> reports/member_analysis.json :

  1. Single-route capability table (manual 表 12-1): AUC, accuracy at an own
     FPR<=3% operating point, TPR@FPR=1%, achieved FPR - for all five score routes.
  2. Score correlation: Pearson and Spearman matrices (with p-values) over the
     out-of-fold scores.
  3. Decision disagreement: pairwise inconsistency rate and per-sample unanimity,
     under two operating-point conventions
       own_fpr1pct        each member thresholded at its own FPR<=1% order statistic
       frozen_common      every member compared against the frozen deployed
                          thr_high (models/fusion.json)
  4. False-alarm (FP set) and miss (FN set) overlap: intersection-over-union (IoU)
     and overlap coefficient for every pair, under both conventions - the manual's
     "误报重叠 / 漏报重叠" requirement.
  5. Marginal value of adding a route: at a common real-data false-positive budget,
     how many errors of the incumbent fusion does the added route fix, and how many
     new errors does it introduce (manual: "它是否真的修正了其他路已经犯下的错误？").
  6. Stepwise incremental gain (manual 表 12-2) and the leave-one-out curve:
     ΔTPR@FPR=1% after removing each route, plus the full Δ curve over an FPR grid,
     for the deployed table, the five-route analysis table and an equal-weight control.
  7. Pre-registered decision rule for keeping a route (declared here, in the source,
     before the numbers are read):
       |ΔTPR@FPR=1%| < 0.02  ->  the route contributes less than roughly one
       corrected AI sample (1/56 = 1.79 pp) -> candidate for removal
     plus an overlap gate: mean pairwise FP-set IoU >= 0.5 with every other route.

The out-of-fold score matrix comes from scripts/metrics_common.py and reproduces
scripts/evaluate_system.py fold for fold.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

# --- pre-registered retention rule (declared before reading the results) ---------
PRESET = {
    "delta_tpr_at_fpr1_min": 0.02,
    "delta_rationale": "1/56 AI samples = 1.79 pp, so 0.02 is 'less than one corrected "
                       "AI sample'; the manual (ch.12.1) asks for a threshold fixed "
                       "before the experiment",
    "primary_fpr_level": 0.05,
    "primary_fpr_rationale": "the 1% guardrail is not resolvable with 63 real samples "
                             "(0.01 < 1/63 = 1.59%), so the retention decision is taken "
                             "at FPR=5%, where the ROC curve has several real points; the "
                             "1% Δ is still reported for completeness",
    "fp_overlap_iou_max": 0.5,
    "overlap_rationale": "mean pairwise FP-set IoU >= 0.5 against every other route means "
                         "the second route mostly repeats the first one's false alarms",
    "declared_before_measurement": True,
}

FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
CONVENTIONS = ["own_fpr1pct", "frozen_common"]


def pairwise_matrix(values_by_member, members, fn):
    mat = {}
    for a in members:
        mat[a] = {}
        for b in members:
            mat[a][b] = fn(values_by_member[a], values_by_member[b])
    return mat


def overlap_stats(set_a, set_b):
    a = set(int(x) for x in set_a)
    b = set(int(x) for x in set_b)
    inter = len(a & b)
    union = len(a | b)
    return {
        "size_a": len(a), "size_b": len(b), "intersection": inter, "union": union,
        "iou": round(inter / union, 4) if union else None,
        "overlap_coefficient": round(inter / min(len(a), len(b)), 4) if min(len(a), len(b)) else None,
    }


def correlation_matrices(scores, members):
    from scipy.stats import pearsonr, spearmanr
    pearson, spearman, detail = {}, {}, {}
    for a in members:
        pearson[a], spearman[a] = {}, {}
        for b in members:
            r = pearsonr(scores[a], scores[b])
            rho = spearmanr(scores[a], scores[b])
            pearson[a][b] = round(float(r[0]), 4)
            spearman[a][b] = round(float(rho[0]), 4)
            if a < b:
                detail[f"{a}|{b}"] = {"pearson": round(float(r[0]), 4),
                                      "pearson_p": float(f"{r[1]:.3g}"),
                                      "spearman": round(float(rho[0]), 4),
                                      "spearman_p": float(f"{rho[1]:.3g}")}
    return pearson, spearman, detail


def thresholds_per_member(y, scores, members, target_fpr, fp_budget):
    thr = {}
    for m in members:
        real = np.asarray(scores[m])[y == 0]
        thr[m] = round(mc.fpr_capped_threshold(real, target_fpr, fp_budget), 6)
    return thr


def decisions(scores, members, thresholds):
    return {m: (np.asarray(scores[m]) >= thresholds[m]).astype(int) for m in members}


def disagreement_analysis(y, dec, members):
    n = len(y)
    pairwise = {}
    for a in members:
        pairwise[a] = {b: round(float(np.mean(dec[a] != dec[b])), 4) for b in members}
    stack = np.stack([dec[m] for m in members], axis=0)
    unanimous = (stack.min(axis=0) == stack.max(axis=0))
    majority = (stack.mean(axis=0) >= 0.5).astype(int)
    # mean pairwise disagreement (excluding the diagonal zeros)
    off = [pairwise[a][b] for a in members for b in members if a != b]
    return {
        "pairwise_disagreement_rate": pairwise,
        "mean_pairwise_disagreement": round(float(np.mean(off)), 4),
        "max_pairwise_disagreement": round(float(np.max(off)), 4),
        "unanimous_fraction": round(float(unanimous.mean()), 4),
        "unanimous_n": int(unanimous.sum()),
        "n_samples": n,
        "majority_vote_confusion": mc.confusion(y, majority),
        "majority_vote_positive_rate": round(float(majority.mean()), 4),
        "per_member_positive_rate": {m: round(float(dec[m].mean()), 4) for m in members},
    }


def error_sets(y, dec, members):
    fp = {m: set(np.where((y == 0) & (dec[m] == 1))[0].tolist()) for m in members}
    fn = {m: set(np.where((y == 1) & (dec[m] == 0))[0].tolist()) for m in members}
    return fp, fn


def overlap_report(y, dec, members, seqs):
    fp, fn = error_sets(y, dec, members)
    out = {
        "false_positive_sets": {
            "per_member": {m: {"n": len(fp[m]), "seqs": sorted(int(seqs[i]) for i in fp[m])}
                           for m in members},
            "iou": pairwise_matrix(fp, members, lambda a, b: overlap_stats(a, b)["iou"]),
            "overlap_coefficient": pairwise_matrix(
                fp, members, lambda a, b: overlap_stats(a, b)["overlap_coefficient"]),
            "detail": {f"{a}|{b}": overlap_stats(fp[a], fp[b])
                       for a in members for b in members if a < b},
        },
        "false_negative_sets": {
            "per_member": {m: {"n": len(fn[m]), "seqs": sorted(int(seqs[i]) for i in fn[m])}
                           for m in members},
            "iou": pairwise_matrix(fn, members, lambda a, b: overlap_stats(a, b)["iou"]),
            "overlap_coefficient": pairwise_matrix(
                fn, members, lambda a, b: overlap_stats(a, b)["overlap_coefficient"]),
            "detail": {f"{a}|{b}": overlap_stats(fn[a], fn[b])
                       for a in members for b in members if a < b},
        },
    }
    # summary: mean off-diagonal IoU, skipping undefined pairs when both sets are empty
    for key in ("false_positive_sets", "false_negative_sets"):
        vals = [v["iou"] for k, v in out[key]["detail"].items() if v["iou"] is not None]
        out[key]["mean_pairwise_iou"] = round(float(np.mean(vals)), 4) if vals else None
        out[key]["pairs_with_undefined_iou"] = sum(
            1 for v in out[key]["detail"].values() if v["iou"] is None)
    return out


def operating_row(y, score, target_fpr=0.03, fp_budget=None):
    """Accuracy/TPR at an FPR-first operating point + the manual's guardrail metrics.

    The returned threshold admits at most `fp_budget` false positives (or
    floor(target_fpr * n_real) when the budget is None) on the real OOF scores, so the
    achieved FPR is integral: with 63 real samples, 1 FP = 1.59% and 2 FP = 3.17%.
    """
    real = np.asarray(score)[y == 0]
    thr = mc.fpr_capped_threshold(real, target_fpr, fp_budget)
    pred = (np.asarray(score) >= thr).astype(int)
    m = mc.metrics_at(y, pred)
    m["threshold"] = round(float(thr), 6)
    m["allowed_fp"] = (int(np.floor(target_fpr * len(real))) if fp_budget is None
                       else int(fp_budget))
    m["n_real"] = int(len(real))
    m["tpr@fpr=1%"] = mc.tpr_at_fpr(y, score, 0.01)
    m["auc"] = mc.auc(y, score)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--target-fpr", type=float, default=0.03,
                    help="real-data false-positive level used for the own-threshold tables")
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/member_analysis.json")
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    y, seqs = a["y"], a["seqs"]
    members = list(mc.MEMBER_ORDER)
    scores = a["oof_calibrated"]                     # calibrated OOF score per route
    fused_cfg = mc.load_deployed_config(ws / "models")

    # ---------------- 1. single-route table (manual 表 12-1)
    single = {}
    for m in members:
        strict = operating_row(y, scores[m], args.target_fpr, 1)          # 1 FP of 63
        deployed_like = operating_row(y, scores[m], args.target_fpr,
                                      mc.FP_BUDGET_DEFAULT)                 # 2 FP of 63
        single[m] = {"definition": bundle["members"][m],
                     "in_deployed_fusion": m in fused_cfg["weights"],
                     "frozen_weight": fused_cfg["weights"].get(m, 0.0),
                     "oof_auc": mc.auc(y, a["oof_logits"][m]),
                     "auc_of_fused_single_member": deployed_like["auc"],
                     "at_fp_budget_1_of_63": strict,
                     "at_fp_budget_2_of_63": deployed_like,
                     "accuracy": deployed_like["accuracy"],
                     "fpr": deployed_like["fpr"],
                     "tpr": deployed_like["tpr"],
                     "tpr@fpr=1%": deployed_like["tpr@fpr=1%"],
                     "inference_cost": "see reports/resource_profile.json",
                     "note": "the 1% guardrail is not resolvable with 63 real samples "
                             "(1 FP = 1.59% FPR); both integer budgets are reported"}
        print(f"[member] {m:14s} AUC={single[m]['oof_auc']:.4f} "
              f"acc@2FP={single[m]['accuracy']:.4f} TPR={single[m]['tpr']:.4f} "
              f"TPR@FPR=1%(interp)={single[m]['tpr@fpr=1%']['tpr_interp']:.4f}", flush=True)

    # ---------------- 2. correlation
    pearson, spearman, corr_detail = correlation_matrices(scores, members)

    # ---------------- 3./4. decision disagreement + error overlap
    conventions = {}
    for conv in CONVENTIONS:
        if conv == "own_fpr1pct":
            thr = thresholds_per_member(y, scores, members, 0.01, 0)
            note = ("each route thresholded at its own order statistic admitting 0 false "
                    "positives on the real OOF scores (target FPR 1% is unreachable at "
                    "n_real=63, where one FP already costs 1.59%)")
        else:
            thr = {m: round(float(fused_cfg["thr_high"]), 6) for m in members}
            note = ("every route compared against the frozen deployed thr_high from "
                    "models/fusion.json, i.e. the decision the shipped pipeline would "
                    "make if that route had been the only evidence")
        dec = decisions(scores, members, thr)
        conventions[conv] = {
            "note": note,
            "thresholds": thr,
            "per_member_confusion": {m: mc.confusion(y, dec[m]) for m in members},
            "disagreement": disagreement_analysis(y, dec, members),
            "error_overlap": overlap_report(y, dec, members, seqs),
        }
        print(f"[member] convention {conv}: unanimous={conventions[conv]['disagreement']['unanimous_fraction']} "
              f"mean_pairwise_disagreement={conventions[conv]['disagreement']['mean_pairwise_disagreement']}",
              flush=True)

    # ---------------- 5. marginal value of adding a route to the incumbent fusion
    base_w = {"combined": 0.60, "forensic": 0.25}
    base_score, base_members = mc.fuse_probabilities(scores, base_w)
    base_row = operating_row(y, base_score, args.target_fpr, mc.FP_BUDGET_DEFAULT)
    base_errors = set(np.where((( base_score >= base_row["threshold"]) != y) )[0].tolist())
    add_one_in = {}
    for m in members:
        if m in base_members:
            continue
        w = dict(base_w)
        w[m] = mc.EXTENDED_WEIGHTS[m]
        sc, ms = mc.fuse_probabilities(scores, w)
        row = operating_row(y, sc, args.target_fpr, mc.FP_BUDGET_DEFAULT)
        errs = set(np.where(((sc >= row["threshold"]) != y))[0].tolist())
        fixed = sorted(int(seqs[i]) for i in (base_errors - errs))
        broken = sorted(int(seqs[i]) for i in (errs - base_errors))
        add_one_in[m] = {
            "weights": w, "members": ms,
            "accuracy": row["accuracy"], "fpr": row["fpr"], "tpr": row["tpr"],
            "auc": row["auc"],
            "delta_accuracy_vs_incumbent": round(row["accuracy"] - base_row["accuracy"], 4),
            "delta_tpr_at_fpr1": round(row["tpr@fpr=1%"]["tpr_interp"]
                                       - base_row["tpr@fpr=1%"]["tpr_interp"], 4),
            "errors_fixed_n": len(fixed), "errors_fixed_seqs": fixed,
            "errors_introduced_n": len(broken), "errors_introduced_seqs": broken,
            "net_errors": len(broken) - len(fixed),
        }
        print(f"[member] add {m:14s} fixed={len(fixed)} broken={len(broken)} "
              f"dAcc={add_one_in[m]['delta_accuracy_vs_incumbent']:+.4f}", flush=True)

    # ---------------- 6. stepwise incremental gain (manual 表 12-2)
    order = sorted(members, key=lambda m: -mc.auc(y, a["oof_logits"][m]))
    stepwise, prev_tpr = [], None
    for k in range(1, len(order) + 1):
        subset = order[:k]
        w = {m: mc.EXTENDED_WEIGHTS[m] for m in subset}
        sc, _ = mc.fuse_probabilities(scores, w)
        row = operating_row(y, sc, args.target_fpr, mc.FP_BUDGET_DEFAULT)
        tpr5 = mc.tpr_at_fpr(y, sc, 0.05)["tpr_interp"]
        entry = {
            "combination": " + ".join(subset),
            "added": order[k - 1],
            "auc": row["auc"], "accuracy": row["accuracy"], "fpr": row["fpr"],
            "tpr": row["tpr"],
            "tpr@fpr=1%": row["tpr@fpr=1%"]["tpr_interp"],
            "tpr@fpr=5%": tpr5,
            "delta_tpr_at_fpr1_vs_previous_row": None if prev_tpr is None
            else round(row["tpr@fpr=1%"]["tpr_interp"] - prev_tpr, 4),
            "delta_tpr_at_fpr5_vs_previous_row": None if k == 1
            else round(tpr5 - stepwise[-1]["tpr@fpr=5%"], 4),
            "weights": w,
        }
        prev_tpr = row["tpr@fpr=1%"]["tpr_interp"]
        stepwise.append(entry)
        print(f"[member] stepwise {entry['combination']:58s} auc={entry['auc']} "
              f"tpr@5%={tpr5} dTPR@5%={entry['delta_tpr_at_fpr5_vs_previous_row']}", flush=True)

    # ---------------- 7. leave-one-out
    loo = {}
    tables = {"deployed_frozen": dict(fused_cfg["weights"]),
              "extended_5route": dict(mc.EXTENDED_WEIGHTS),
              "equal_5route": {m: 1.0 for m in members}}
    for name, w in tables.items():
        present = [m for m in members if m in w]
        full_score, _ = mc.fuse_probabilities(scores, w, present)
        full_row = operating_row(y, full_score, args.target_fpr, mc.FP_BUDGET_DEFAULT)
        entries = {}
        for m in present:
            rest = [x for x in present if x != m]
            sc, ms = mc.fuse_probabilities(scores, w, rest)
            row = operating_row(y, sc, args.target_fpr, mc.FP_BUDGET_DEFAULT)
            entries[m] = {
                "members_after_removal": ms,
                "auc": row["auc"], "accuracy": row["accuracy"], "fpr": row["fpr"],
                "tpr": row["tpr"],
                "tpr@fpr=1%": row["tpr@fpr=1%"]["tpr_interp"],
                "delta_tpr_at_fpr1_full_minus_removed": round(
                    full_row["tpr@fpr=1%"]["tpr_interp"]
                    - row["tpr@fpr=1%"]["tpr_interp"], 4),
                # the 1% point is granularity-degenerate here (63 real samples), so the
                # same Δ is also given at FPR=5%, where the ROC is resolvable
                "delta_tpr_at_fpr5pct_full_minus_removed": round(
                    mc.tpr_at_fpr(y, full_score, 0.05)["tpr_interp"]
                    - mc.tpr_at_fpr(y, sc, 0.05)["tpr_interp"], 4),
                "delta_accuracy": round(full_row["accuracy"] - row["accuracy"], 4),
                "delta_auc": round(full_row["auc"] - row["auc"], 4),
                "delta_curve_tpr_at_fpr": {
                    f"fpr={t:.3f}": round(
                        mc.tpr_at_fpr(y, full_score, t)["tpr_interp"]
                        - mc.tpr_at_fpr(y, sc, t)["tpr_interp"], 4)
                    for t in FPR_GRID},
                "contribution_verdict": (
                    "below the pre-registered 0.02 ΔTPR@FPR=1% gate - candidate for removal"
                    if abs(full_row["tpr@fpr=1%"]["tpr_interp"]
                           - row["tpr@fpr=1%"]["tpr_interp"]) < PRESET["delta_tpr_at_fpr1_min"]
                    else "contributes at least the pre-registered ΔTPR margin"),
            }
        # only routes that are actually present can be removed within this table
        missing = [m for m in members if m not in present]
        loo[name] = {
            "weights": w, "members": present,
            "routes_not_in_this_table": missing,
            "full_combination": {"auc": full_row["auc"], "accuracy": full_row["accuracy"],
                                 "fpr": full_row["fpr"], "tpr": full_row["tpr"],
                                 "tpr@fpr=1%": full_row["tpr@fpr=1%"],
                                 "tpr@fpr=5%": mc.tpr_at_fpr(y, full_score, 0.05)["tpr_interp"],
                                 "threshold": full_row["threshold"]},
            "granularity_note": "TPR@FPR=1% is an integer count over 63 real samples: the "
                                "0.01 target is below the smallest achievable non-zero FPR "
                                "(1/63 = 1.59%), so Δ values at 1% are dominated by one "
                                "sample. The accompanying FPR-grid curve and the 5% column "
                                "show the region where the ranking is resolvable.",
            "remove": entries,
        }
        print(f"[member] LOO {name}: full TPR@FPR=1%="
              f"{full_row['tpr@fpr=1%']['tpr_interp']:.4f} -> "
              + ", ".join(f"-{m}: Δ{entries[m]['delta_tpr_at_fpr1_full_minus_removed']:+.4f}"
                          for m in present), flush=True)

    # retention gate: mean FP IoU under the frozen-common convention
    fp_iou = conventions["frozen_common"]["error_overlap"]["false_positive_sets"]
    overlap_gate = {}
    for m in members:
        vals = [fp_iou["detail"][k]["iou"] for k in fp_iou["detail"]
                if m in k.split("|") and fp_iou["detail"][k]["iou"] is not None]
        overlap_gate[m] = {
            "mean_fp_iou_vs_others": round(float(np.mean(vals)), 4) if vals else None,
            "pairs_used": len(vals),
            "passes_overlap_gate": (None if not vals
                                    else bool(np.mean(vals) < PRESET["fp_overlap_iou_max"])),
        }

    payload = {
        "report": "member_analysis",
        "headline": {
            "n_samples": int(len(y)),
            "members": members,
            "single_route_auc": {m: single[m]["oof_auc"] for m in members},
            "mean_pairwise_spearman": round(float(np.mean(
                [spearman[a][b] for a in members for b in members if a < b])), 4),
            "unanimous_fraction_at_frozen_threshold": (
                conventions["frozen_common"]["disagreement"]["unanimous_fraction"]),
            "mean_pairwise_disagreement_at_frozen_threshold": (
                conventions["frozen_common"]["disagreement"]["mean_pairwise_disagreement"]),
            "mean_fp_overlap_iou_at_frozen_threshold": (
                conventions["frozen_common"]["error_overlap"]["false_positive_sets"]
                ["mean_pairwise_iou"]),
            "mean_fn_overlap_iou_at_frozen_threshold": (
                conventions["frozen_common"]["error_overlap"]["false_negative_sets"]
                ["mean_pairwise_iou"]),
            "incumbent_fusion_accuracy": base_row["accuracy"],
        },
        "manual_sections": ["12.1 多模型组合与错误相关性实验", "15.1 诊断指标",
                            "2.3 多路信号的必要性"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/member_analysis.py",
        "data": {
            "n_samples": len(y),
            "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
            "score_definition": "out-of-fold calibrated probability per route "
                                "(scripts/metrics_common.py; per-member temperature "
                                "scaling fitted on inner-CV training logits)",
            "correlation_scored_on": "calibrated OOF probability; Spearman is invariant to "
                                     "any monotone rescaling, Pearson is not",
            "protocol": bundle["protocol"],
        },
        "preset_decision_rule": PRESET,
        "deployed_fusion": {"weights": fused_cfg["weights"], "source": fused_cfg["source"],
                            "thr_high": fused_cfg["thr_high"]},
        "single_route_table": single,
        "score_correlation": {"pearson": pearson, "spearman": spearman, "pairs": corr_detail},
        "decision_disagreement_and_overlap": conventions,
        "add_one_in_vs_incumbent_fusion": {
            "incumbent": {"members": base_members, "weights": base_w,
                          "accuracy": base_row["accuracy"], "fpr": base_row["fpr"],
                          "tpr": base_row["tpr"], "auc": base_row["auc"],
                          "threshold": base_row["threshold"],
                          "error_seqs": sorted(int(seqs[i]) for i in base_errors)},
            "operating_point": f"threshold from the real OOF scores with a false-positive "
                               f"budget of {mc.FP_BUDGET_DEFAULT} (target FPR 3% at 7:3)",
            "add": add_one_in,
        },
        "stepwise_incremental": {"order": order, "rows": stepwise},
        "leave_one_out": loo,
        "retention_gate": overlap_gate,
        "conclusions": None,   # filled below
        "limitations": [
            "All numbers are out-of-fold on the 119 competition samples; the analysis "
            "tables that include clip_domain/npr are analysis configurations, not the "
            "deployed fusion (models/fusion.json uses combined+forensic only).",
            "TPR@FPR=1% cannot be measured exactly with 63 real samples (1 FP = 1.59%); "
            "the interpolated ROC value is reported and the integer granularity is "
            "noted wherever a Δ of one sample is discussed.",
            "Error-set overlap is computed at a fixed operating point; a route that is "
            "redundant under one threshold can be complementary under another, so both "
            "conventions are reported.",
        ],
    }

    # retention decision: the ΔTPR gate is evaluated at the pre-registered primary FPR
    # level (5%), where the ROC is resolvable over 63 real samples
    gate1 = {m: e["delta_tpr_at_fpr1_full_minus_removed"]
             for m, e in loo["extended_5route"]["remove"].items()}
    gate5 = {m: e["delta_tpr_at_fpr5pct_full_minus_removed"]
             for m, e in loo["extended_5route"]["remove"].items()}
    below = [m for m, d in gate5.items() if abs(d) < PRESET["delta_tpr_at_fpr1_min"]]
    for m, e in loo["extended_5route"]["remove"].items():
        d5 = e["delta_tpr_at_fpr5pct_full_minus_removed"]
        e["contribution_verdict"] = (
            f"below the pre-registered |ΔTPR| >= {PRESET['delta_tpr_at_fpr1_min']} gate at "
            f"FPR={PRESET['primary_fpr_level']:.0%} - candidate for removal"
            if abs(d5) < PRESET["delta_tpr_at_fpr1_min"]
            else f"clears the pre-registered |ΔTPR| gate at FPR={PRESET['primary_fpr_level']:.0%}")
        e["contribution_verdict_basis"] = f"delta_tpr_at_fpr{PRESET['primary_fpr_level']:.0%}"
    payload["conclusions"] = {
        "routes_below_gate_primary_fpr5": below,
        "routes_below_gate_fpr1_informational": [
            m for m, d in gate1.items() if abs(d) < PRESET["delta_tpr_at_fpr1_min"]],
        "loo_delta_at_fpr1pct": gate1,
        "loo_delta_at_fpr5pct": gate5,
        "single_member_auc_note": "a one-member fusion equals that route's clipped "
                                  "calibrated probability, so its AUC can differ from "
                                  "oof_auc in the 4th decimal (fusion.py clips at 1e-6)",
        "npr_note": (
            "npr is the weakest route (see single_route_table) and the stepwise table "
            "shows whether its incremental ΔTPR clears the pre-registered gate"),
        "closest_pairs_by_correlation": sorted(
            corr_detail.items(), key=lambda kv: -abs(kv[1]["spearman"]))[:3],
        "highest_fp_overlap_pairs": sorted(
            [(k, v["iou"]) for k, v in
             conventions["frozen_common"]["error_overlap"]["false_positive_sets"]["detail"].items()
             if v["iou"] is not None], key=lambda kv: -kv[1])[:3],
        "reading": "a route is retained only if it clears the ΔTPR gate AND its errors are "
                   "not dominated by overlap with the existing routes (manual ch.12.1)",
    }
    mc.write_json(out_path, payload)
    print(f"\n[member] routes below the gate at FPR="
          f"{PRESET['primary_fpr_level']:.0%}: {below}")


if __name__ == "__main__":
    main()
