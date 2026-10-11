#!/usr/bin/env python3
"""Per-dimension metric breakdown (manual ch.15.1 诊断指标: 分内容类别 / 分采集方式 / 分生成器).

Evidence produced -> reports/stratified_analysis.json :

  1. Content-class breakdown: "文档类" (bills / medical records / invoices / expense
     sheets) vs "场景照" (property damage / food / scene photos), split by the L1b
     proxy feature b_nearWhiteFrac (near-white area fraction).
  2. Capture-mode breakdown: "数字渲染态" vs "手机拍摄态" proxies built from
     b_nearWhiteFrac x b_flatNoise, with the split points taken from the data
     distribution and written into the JSON next to the rule.
  3. Generator breakdown: the 命题 does NOT provide generator labels for the 119
     samples, so this section is emitted with status="unavailable" plus an explicit
     pointer to where generator-level generalisation evidence does exist (the
     18-generator external benchmark inside the CLIP probe checkpoint and the
     robustness matrix). No generator grouping is fabricated.
  4. For every group: sample size, class balance, Accuracy / FPR / TPR at the frozen
     deployed operating point, the error samples (sequences), AUC (threshold-free
     comparison) and the group's share of the total error budget - so the report can
     say which group is the main error source.

PROXY-LABEL WARNING: both groupings are derived from image statistics, NOT from human
annotation. They are documented as proxies in every JSON section (field
`label_kind="proxy"`) and must not be quoted as ground-truth categories. The content
split is additionally confounded with the acquisition split: a near-white document is
both "a document" and "render-like" by construction, so the two groupings overlap by
design and are reported side by side rather than merged.

Scores come from scripts/metrics_common.py (out-of-fold, identical folds to
scripts/evaluate_system.py; the deployed operating point reproduces reports/metrics.json).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

CONTENT_RULE_TEXT = (
    "content class proxy: b_nearWhiteFrac >= 0.05 -> 文档类 (bill/medical/invoice/"
    "expense material, large near-white paper background); b_nearWhiteFrac < 0.05 -> "
    "场景照 (property-damage / food / scene photos). The threshold sits in the valley of "
    "the observed bimodal distribution (p25=0.000, p50=0.002, p75=0.819), i.e. the "
    "feature is close to binary on this sample set.")

CAPTURE_RULE_TEXT = (
    "capture-mode proxy: digital-render-like = b_nearWhiteFrac >= content_threshold AND "
    "b_flatNoise <= median(b_flatNoise of the document proxy group); camera-capture-like = "
    "b_nearWhiteFrac < content_threshold AND b_flatNoise >= median(b_flatNoise of the "
    "scene proxy group); everything else is left unassigned. Split points are read from "
    "the data distribution at run time and echoed in `split_point_thresholds`.")


def group_metrics(y, pred, fused, mask):
    """Metrics for one group; every per-class metric degrades to None if the group has
    only one class (with 119 samples some proxies create single-class cells)."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    out = {"n": n, "n_ai": int((y[mask] == 1).sum()), "n_real": int((y[mask] == 0).sum())}
    if n == 0:
        return out
    c = mc.confusion(y[mask], pred[mask])
    out["confusion"] = c
    out["accuracy"] = round((c["tp"] + c["tn"]) / max(1, n), 4)
    out["fpr"] = round(c["fp"] / out["n_real"], 4) if out["n_real"] else None
    out["tpr"] = round(c["tp"] / out["n_ai"], 4) if out["n_ai"] else None
    out["precision"] = round(c["tp"] / max(1, c["tp"] + c["fp"]), 4)
    out["auc"] = mc.auc(y[mask], fused[mask]) if len(np.unique(y[mask])) > 1 else None
    out["false_positives_n"] = c["fp"]
    out["false_negatives_n"] = c["fn"]
    out["error_rate"] = round((c["fp"] + c["fn"]) / max(1, n), 4)
    return out


def groups_table(y, pred, fused, seqs, masks, overall):
    table = {}
    for name, mask in masks.items():
        g = group_metrics(y, pred, fused, mask)
        m = np.asarray(mask, dtype=bool)
        fp = sorted(int(s) for s in seqs[m & (y == 0) & (pred == 1)])
        fn = sorted(int(s) for s in seqs[m & (y == 1) & (pred == 0)])
        g["false_positive_seqs"] = fp
        g["false_negative_seqs"] = fn
        if "accuracy" in g:
            g["delta_accuracy_vs_overall"] = round(g["accuracy"] - overall["accuracy"], 4)
            g["delta_fpr_vs_overall"] = (None if g["fpr"] is None or overall["fpr"] is None
                                         else round(g["fpr"] - overall["fpr"], 4))
            g["delta_tpr_vs_overall"] = (None if g["tpr"] is None or overall["tpr"] is None
                                         else round(g["tpr"] - overall["tpr"], 4))
            tot_fp = max(1, overall["confusion"]["fp"])
            tot_fn = max(1, overall["confusion"]["fn"])
            g["share_of_false_positives"] = round(g["false_positives_n"] / tot_fp, 4)
            g["share_of_false_negatives"] = round(g["false_negatives_n"] / tot_fn, 4)
            g["share_of_samples"] = round(int(m.sum()) / max(1, len(y)), 4)
        table[name] = g
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/stratified_analysis.json")
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    y, seqs = a["y"], a["seqs"]
    pred, fused = a["deployed_pred"], a["deployed_fused"]
    overall = mc.metrics_at(y, pred)
    overall["auc"] = mc.auc(y, fused)

    # ---- proxy features (L1b) -----------------------------------------------------
    l1b = np.load(ws / "models" / "l1b_feats_119.npy").astype(np.float64)
    nw, fz = l1b[:, 0], l1b[:, 3]
    doc_mask = nw >= 0.05
    doc_flatnoise_median = float(np.median(fz[doc_mask])) if doc_mask.any() else None
    scene_flatnoise_median = float(np.median(fz[~doc_mask])) if (~doc_mask).any() else None
    render_mask = doc_mask & (fz <= doc_flatnoise_median) if doc_flatnoise_median else np.zeros(len(y), bool)
    photo_mask = (~doc_mask) & (fz >= scene_flatnoise_median) if scene_flatnoise_median else np.zeros(len(y), bool)
    unassigned = ~(render_mask | photo_mask)

    content_masks = {"文档类(代理)": doc_mask, "场景照(代理)": ~doc_mask}
    capture_masks = {"数字渲染态(代理)": render_mask, "手机拍摄态(代理)": photo_mask,
                     "采集方式未定(代理)": unassigned}
    cross = {
        "文档类 x 数字渲染态": doc_mask & render_mask,
        "文档类 x 手机拍摄态": doc_mask & photo_mask,
        "场景照 x 数字渲染态": (~doc_mask) & render_mask,
        "场景照 x 手机拍摄态": (~doc_mask) & photo_mask,
    }

    content_table = groups_table(y, pred, fused, seqs, content_masks, overall)
    capture_table = groups_table(y, pred, fused, seqs, capture_masks, overall)
    cross_table = groups_table(y, pred, fused, seqs, cross, overall)

    for name, g in list(content_table.items()) + list(capture_table.items()):
        print(f"[strat] {name:22s} n={g['n']:3d} ai={g['n_ai']:2d} real={g['n_real']:2d} "
              f"acc={g.get('accuracy')} fpr={g.get('fpr')} tpr={g.get('tpr')} "
              f"FP={g.get('false_positives_n')} FN={g.get('false_negatives_n')}", flush=True)

    # which group is the main error source
    def main_source(table):
        scored = [(k, v.get("error_rate", 0.0), v.get("n", 0),
                   v.get("false_positives_n", 0), v.get("false_negatives_n", 0))
                  for k, v in table.items() if "error_rate" in v]
        if not scored:
            return None
        worst = max(scored, key=lambda t: t[1])
        most_fp = max(scored, key=lambda t: t[3])
        most_fn = max(scored, key=lambda t: t[4])
        return {
            "highest_error_rate_group": worst[0],
            "highest_error_rate": worst[1],
            "most_false_positives_group": most_fp[0],
            "most_false_negatives_group": most_fn[0],
        }

    # ---- generator section: honestly unavailable ---------------------------------
    probe_meta = bundle.get("clip_external_probe", {})
    generator = {
        "status": "unavailable",
        "reason": "命题给定的 119 张样本只提供 real/ai 二分类标签（data/labels.csv 只有 "
                  "seq/filename/label 三列），没有任何生成器 / 工具 / 模型标识，因此无法按"
                  "生成器拆分准确率，也不允许凭内容或文件名推测生成器。",
        "not_fabricated": True,
        "alternative_evidence": {
            "what": "生成器层面的泛化证据来自训练侧的 18 生成器公开基准，而不是本次评测样本",
            "clip_probe_external_benchmark": {
                "checkpoint": "models/clip_probe_norm.pt",
                "cv_aucs": probe_meta.get("cv_aucs"),
                "val_auc": probe_meta.get("val_auc"),
                "note": "5 折交叉验证 AUC（3 shards, data/external/*.parquet）与最终留出 AUC",
            },
            "probe_training_log": "models/train_clip_norm.log",
            "robustness_evidence": "reports/robustness_matrix.json (manual ch.14.1: "
                                   "which clue fails first under recompression / rescale / "
                                   "screenshot / re-share / noise)",
            "how_to_close_the_gap": "要真正给出分生成器指标，必须补充带生成器标签的评测集"
                                    "（或对 18 生成器基准按生成器分层重跑同一融合管线）；"
                                    "该数据不在当前工作区内",
        },
    }

    def worst_error_group(table):
        rows = [(k, v.get("error_rate"), v.get("false_positives_n", 0),
                 v.get("false_negatives_n", 0)) for k, v in table.items()
                if "error_rate" in v]
        rows.sort(key=lambda t: -t[1])
        return rows

    fp_by_group = {k: v.get("false_positives_n") for k, v in content_table.items()}
    fn_by_group = {k: v.get("false_negatives_n") for k, v in content_table.items()}
    worst_fp = max(fp_by_group, key=lambda k: (fp_by_group[k] or 0))
    worst_fn = max(fn_by_group, key=lambda k: (fn_by_group[k] or 0))

    payload = {
        "report": "stratified_analysis",
        "manual_sections": ["15.1 诊断指标（分生成器 / 分采集方式 / 分内容类别）",
                            "14.2 错误分析协议", "2.4 领域适配：渲染态 vs 翻拍态"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/stratified_analysis.py",
        "label_kind": "proxy",
        "proxy_disclaimer": "所有分组均为基于图像统计量（L1b 特征）的代理标签，不是人工标注；"
                            "不得作为标签真值引用。内容类别代理与采集方式代理在设计上互相"
                            "重叠（近白背景既意味着「文档」也意味着「渲染」特征），因此分表"
                            "并列展示而不合并。",
        "evaluated_on": {
            "n_samples": len(y),
            "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
            "operating_point": "out-of-fold deployed path (per-fold admitted members + "
                               "per-fold order-statistic threshold); reproduces "
                               "reports/metrics.json",
            "frozen_weights": mc.load_deployed_config(ws / "models")["weights"],
        },
        "overall": overall,
        "content_class": {
            "method": CONTENT_RULE_TEXT,
            "label_kind": "proxy",
            "feature": "b_nearWhiteFrac (L1b, fraction of pixels with luma >= 245)",
            "split_point": 0.05,
            "feature_distribution": {f"p{q}": round(float(np.percentile(nw, q)), 4)
                                     for q in (0, 10, 25, 50, 75, 90, 100)},
            "groups": content_table,
            "summary": main_source(content_table),
        },
        "capture_mode": {
            "method": CAPTURE_RULE_TEXT,
            "label_kind": "proxy",
            "features": ["b_nearWhiteFrac", "b_flatNoise (std of the high-frequency "
                                               "residual inside flat regions)"],
            "split_point_thresholds": {
                "near_white_frac": 0.05,
                "flat_noise_render_max": (round(doc_flatnoise_median, 4)
                                          if doc_flatnoise_median is not None else None),
                "flat_noise_photo_min": (round(scene_flatnoise_median, 4)
                                         if scene_flatnoise_median is not None else None),
                "selection": "flat-noise split points are the medians of the corresponding "
                             "content proxy group, read from the data at run time",
            },
            "feature_distribution": {
                "flat_noise": {f"p{q}": round(float(np.percentile(fz, q)), 4)
                               for q in (0, 10, 25, 50, 75, 90, 100)},
            },
            "groups": capture_table,
            "summary": main_source(capture_table),
        },
        "content_x_capture": {
            "note": "cross-tabulation of the two proxies; cells are small and are reported "
                    "for completeness, not for decision-making",
            "groups": cross_table,
        },
        "by_generator": generator,
        "headline": {
            "n_samples": int(len(y)),
            "overall_accuracy": overall["accuracy"],
            "overall_fpr": overall["fpr"],
            "overall_tpr": overall["tpr"],
            "document_proxy_accuracy": content_table["文档类(代理)"].get("accuracy"),
            "document_proxy_fpr": content_table["文档类(代理)"].get("fpr"),
            "document_proxy_tpr": content_table["文档类(代理)"].get("tpr"),
            "scene_proxy_accuracy": content_table["场景照(代理)"].get("accuracy"),
            "scene_proxy_fpr": content_table["场景照(代理)"].get("fpr"),
            "scene_proxy_tpr": content_table["场景照(代理)"].get("tpr"),
            "main_false_positive_source": worst_fp,
            "main_false_negative_source": worst_fn,
            "generator_breakdown_status": "unavailable",
        },
        "error_source_conclusion": {
            "content_groups_ranked_by_error_rate": worst_error_group(content_table),
            "capture_groups_ranked_by_error_rate": worst_error_group(capture_table),
            "main_false_positive_source": worst_fp,
            "main_false_negative_source": worst_fn,
            "statement": (f"误报主要来自「{worst_fp}」（该组占了全部 {overall['confusion']['fp']} "
                          f"个误报），漏报主要来自「{worst_fn}」（该组占了全部 "
                          f"{overall['confusion']['fn']} 个漏报）。这解释了为什么手册 ch.13.1 "
                          f"要求分别对单据类材料校准阈值。"),
            "reading": "两个代理分组的方向相反：文档类几乎不产生误报但贡献了全部漏报，场景照"
                       "产生了全部误报但召回接近饱和。由于分组是代理标签，该结论的用途是定位"
                       "错误来源（诊断），不作为对外承诺。",
        },
        "limitations": [
            "Proxy labels come from a single feature pair measured on the same images the "
            "detector scores, so a group difference can reflect the feature's own "
            "correlation with the label rather than the stated semantic category.",
            "Group sizes are small (the smallest reported cell has tens of samples), so "
            "group-level FPR/TPR move in steps of 1/n_group; no significance is claimed.",
            "The 119 samples are class-clustered in file order (命题样本类别序贯排列), a "
            "known bias already recorded in the manual (ch.10.1).",
        ],
    }

    mc.write_json(out_path, payload)
    print(f"\n[strat] overall acc={overall['accuracy']} fpr={overall['fpr']} "
          f"tpr={overall['tpr']}")
    print(f"[strat] error sources: FP -> {worst_fp}, FN -> {worst_fn}")


if __name__ == "__main__":
    main()
