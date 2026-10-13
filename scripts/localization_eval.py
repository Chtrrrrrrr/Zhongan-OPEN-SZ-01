#!/usr/bin/env python
"""Batch localization evaluation over the frozen 119-sample holdout.

What it produces (``repo/reports/localization_regions.json``):

* one row per sample with the **adaptive** region extraction and the **legacy
  fixed-threshold** extraction side by side (box count + boxes + which threshold
  tier fired), so the 62.5%-empty-suspect problem can be read off directly;
* an explicit ``reason_code``/``reason`` for every empty result, distinguishing
  ``no_localizable_region`` ("whole image generated / nothing stands out") from
  ``localization_failed`` ("the localization computation itself failed");
* a block-level agreement statistic between the anomaly map and the classifier
  attention explanation (Spearman over the tile attribution weights from the
  deployed run) plus the resulting *review hint* basis - advisory only;
* the zero-weight / not-in-decision declaration and the TruFor next step;
* before/after counts of the self-contradictory runtime section
  (``available=True`` + ``boxes=[]`` + ``reason=None``) read from the cached
  deployed reports.

Pre-registration: the adaptive constants live in ``src/l3_localization.py`` and
were frozen *before* this script was run over the holdout. Labels from
``data/labels.csv`` are used for **stratified reporting only**; no threshold,
weight or rule is selected with them. The 119 samples are the frozen holdout of
the project, not a tuning set.

Runtime fidelity: images are decoded through ``l0_input.parse_image`` and the
anomaly map is computed on ``parsed.bgr`` - exactly the tensor the runtime L3
block feeds to ``combined_anomaly_map`` (longest side <= ``MODEL_SIDE``). Boxes
are therefore reported in the same (model-side) coordinate space the runtime
report uses; the attribution tiles carry native-image coordinates and are scaled
onto the map grid for the consistency statistic.

Usage::

    python repo/scripts/localization_eval.py --out repo/reports/localization_regions.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from l3_localization import (  # noqa: E402
    ADAPTIVE_MIN_AREA_ABS,
    ADAPTIVE_MIN_AREA_FRAC,
    ADAPTIVE_NO_CONTENT_MAX,
    ADAPTIVE_PEAK_FRAC,
    ADAPTIVE_QUANTILE_LADDER,
    ADAPTIVE_TOP_K,
    BROAD_REGION_AREA_FRAC,
    CONSISTENCY_MIN_TILES,
    CONSISTENCY_RHO_INCONSISTENT,
    ENABLE_CONSISTENCY_REVIEW_HINT,
    LEGACY_FIXED_THRESH,
    LOC_REASON_FAILED,
    LOC_REASON_NO_REGION,
    LOC_REASON_REGIONS,
    LOCALIZATION_DECLARATION,
    LOCALIZATION_KIND,
    LOCALIZATION_REVISION_LOG,
    adaptive_region_extract,
    combined_anomaly_map,
    localize_heatmap,
    region_consistency,
)
from l3_localization import top_anomaly_boxes  # noqa: E402  (legacy fixed path)
from l0_input import parse_image  # noqa: E402  (the runtime decode path)

EMPTY_CN = {
    LOC_REASON_NO_REGION: "整图范围生成/无可指认伪造区域（自适应阈值下无显著热点）",
    LOC_REASON_FAILED: "定位计算失败",
}

#: The two causes an empty region list is allowed to have, as single letters:
#:   ``a`` = whole-image generation / no localised forged region to point at
#:           (the map has content, the extraction just refuses to invent a region)
#:   ``b`` = localisation computation failed / map degenerate (a real failure)
#: These stay distinct in every per-image row and in the summary histogram.
EMPTY_CASE = {
    LOC_REASON_NO_REGION: "a",
    LOC_REASON_FAILED: "b",
}
EMPTY_CASE_TEXT = {
    "a": "整图范围生成：无可指认的局部伪造区域（合法结果）",
    "b": "定位计算失败/异常图退化（真实故障）",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _read_labels(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            rows.append({"seq": int(rec["seq"]), "filename": rec["filename"].strip(),
                         "label": (rec.get("label") or "").strip()})
    rows.sort(key=lambda r: r["seq"])
    return rows


def _resolve_image(filename: str, workspace: Path) -> Path:
    p = Path(filename)
    return p if p.is_absolute() else (workspace / p)


def _load_cached(attribution_dir: Path | None, seq: int) -> dict:
    if attribution_dir is None:
        return {}
    f = attribution_dir / f"{seq:03d}.json"
    if not f.exists():
        return {}
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "verdict": d.get("verdict"),
        "localization": d.get("localization") or {},
        "tiles": ((d.get("explanation") or {}).get("tiles") or []),
        "explanation_available": bool((d.get("explanation") or {}).get("available")),
        "path": str(f),
    }


def _summarize_boxes(boxes: list) -> list:
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])] for b in (boxes or [])]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 4) if xs else None


# ---------------------------------------------------------------------------
# self-check of the two empty-result branches (no holdout data involved)
# ---------------------------------------------------------------------------
def self_check() -> dict:
    cases = {}
    ok_map = np.zeros((64, 64), dtype=np.float32)
    ok_map[20:30, 20:30] = 0.9
    blk = localize_heatmap(ok_map)
    cases["regions"] = {"expect": LOC_REASON_REGIONS, "got": blk["reason_code"],
                        "n_boxes": len(blk["boxes"]), "status": blk["status"],
                        "reason": blk["reason"]}
    blk = localize_heatmap(np.zeros((128, 128), dtype=np.float32))
    cases["no_localizable_region"] = {
        "expect": LOC_REASON_NO_REGION, "got": blk["reason_code"],
        "n_boxes": len(blk["boxes"]), "status": blk["status"], "reason": blk["reason"]}
    bad = np.full((64, 64), np.nan, dtype=np.float32)
    blk = localize_heatmap(bad)
    cases["localization_failed"] = {
        "expect": LOC_REASON_FAILED, "got": blk["reason_code"],
        "n_boxes": len(blk["boxes"]), "status": blk["status"], "reason": blk["reason"]}
    blk = localize_heatmap(np.sin(np.arange(96 * 96, dtype=np.float32).reshape(96, 96)))
    cases["degenerate_tier_fallback"] = {
        "expect": LOC_REASON_REGIONS, "got": blk["reason_code"],
        "n_boxes": len(blk["boxes"]), "threshold_tier": blk["rules"].get("threshold_tier"),
        "reliability": blk["rules"].get("reliability"),
        "min_area_relaxed": blk["rules"].get("min_area_relaxed", False)}
    passed = all(c["got"] == c["expect"] for c in cases.values())
    # the empty-result contract: reason must never be blank when boxes are empty
    blank = [n for n, c in cases.items() if c["n_boxes"] == 0 and not c["reason"]]
    return {"cases": cases, "passed": bool(passed), "blank_reasons_when_empty": blank,
            "contract": "boxes == []  =>  status/reason_code/reason 必须显式给出"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(args) -> dict:
    import cv2

    labels = _read_labels(Path(args.labels))
    if args.limit:
        labels = labels[: args.limit]
    attr_dir = None if args.no_consistency else Path(args.attribution_dir)
    if attr_dir is not None and not attr_dir.is_absolute():
        attr_dir = WS / attr_dir

    rows, t0 = [], time.perf_counter()
    for i, rec in enumerate(labels, 1):
        seq, label = rec["seq"], rec["label"]
        img_path = _resolve_image(rec["filename"], Path(args.workspace_root))
        row = {"seq": seq, "filename": rec["filename"], "label": label,
               "image": str(img_path), "image_available": img_path.exists()}
        # Runtime fidelity: detector.detect_image feeds L3 with ``parsed.bgr``
        # (PIL decode, longest side <= MODEL_SIDE), NOT with a native cv2.imread.
        parsed = parse_image(str(img_path)) if row["image_available"] else None
        if parsed is None or not parsed.ok or parsed.bgr is None:
            err = "图像无法读取" if parsed is None else (parsed.error or "图像无法解码")
            row.update({
                "image_available": False,
                "adaptive": {"n_boxes": 0, "boxes": [], "status": "failed",
                             "reason_code": LOC_REASON_FAILED,
                             "reason": f"{EMPTY_CN[LOC_REASON_FAILED]}：{err}"},
                "heuristic": {"n_boxes": 0, "boxes": [], "status": "failed",
                              "reason_code": LOC_REASON_FAILED,
                              "reason": f"{EMPTY_CN[LOC_REASON_FAILED]}：{err}"},
                "resolved": False, "consistency": {"available": False,
                                                   "reason": f"一致性未计算：{err}"},
                "report_block": {"available": False, "status": "failed",
                                 "reason_code": LOC_REASON_FAILED,
                                 "reason": f"{EMPTY_CN[LOC_REASON_FAILED]}：{err}",
                                 "n_boxes": 0, "contradictory": False},
                "cached_report_localization": {}})
            rows.append(row)
            continue

        h, w = int(parsed.orig_height), int(parsed.orig_width)
        work_bgr = parsed.bgr
        if args.max_side and max(work_bgr.shape[:2]) > args.max_side:
            s = args.max_side / max(work_bgr.shape[:2])
            work_bgr = cv2.resize(work_bgr,
                                  (max(8, int(work_bgr.shape[1] * s)),
                                   max(8, int(work_bgr.shape[0] * s))),
                                  interpolation=cv2.INTER_AREA)
        amap = combined_anomaly_map(work_bgr)
        mh, mw = amap.shape

        a_boxes, a_info = adaptive_region_extract(amap, max_boxes=ADAPTIVE_TOP_K)
        h_boxes, h_info = _fixed_extract(amap, ADAPTIVE_TOP_K, LEGACY_FIXED_THRESH)
        blk = localize_heatmap(amap, max_boxes=ADAPTIVE_TOP_K,
                               with_legacy_comparison=False)

        cache = _load_cached(attr_dir, seq)
        # attribution tiles carry *native image* coordinates (detector writes them
        # back to the original resolution) -> rescale onto the anomaly-map grid.
        cons = region_consistency(amap, cache.get("tiles") or [], a_boxes,
                                  xy_scale=(mw / float(w), mh / float(h)))
        cons["attribution_source"] = cache.get("path")
        cons["attribution_tiles"] = len(cache.get("tiles") or [])
        loc = cache.get("localization") or {}
        row["cached_report_localization"] = {
            "path": cache.get("path"),
            "available": loc.get("available"),
            "n_boxes": len(loc.get("boxes") or []),
            "reason": loc.get("reason"),
            "contradictory": bool(loc.get("available") and not (loc.get("boxes") or [])
                                  and not loc.get("reason")),
        }

        row.update({
            "image_size": [w, h],
            "map_size": [mw, mh],
            "verdict": cache.get("verdict"),
            "anomaly_map_stats": {"max": round(float(amap.max()), 6),
                                  "mean": round(float(amap.mean()), 6),
                                  "std": round(float(amap.std()), 6)},
            "adaptive": {
                "n_boxes": len(a_boxes), "boxes": _summarize_boxes(a_boxes),
                "threshold": a_info.get("threshold"),
                "threshold_tier": a_info.get("threshold_tier"),
                "reliability": a_info.get("reliability"),
                "n_components": a_info.get("n_components"),
                "min_area": a_info.get("min_area"),
                "region_scope": a_info.get("region_scope"),
                "top_box_area_frac": a_info.get("top_box_area_frac"),
                "n_broad_boxes": a_info.get("n_broad_boxes"),
                "status": a_info.get("status"),
                "reason_code": a_info.get("reason_code"),
                "reason": a_info.get("reason"),
                "empty_explanation_cn": EMPTY_CN.get(a_info.get("reason_code"))
                if not a_boxes else None,
            },
            "heuristic": {
                "n_boxes": len(h_boxes), "boxes": _summarize_boxes(h_boxes),
                "threshold": h_info.get("threshold"),
                "threshold_tier": h_info.get("threshold_tier"),
                "n_components": h_info.get("n_components"),
                "status": h_info.get("status"),
                "reason_code": h_info.get("reason_code"),
                "reason": h_info.get("reason"),
                "empty_explanation_cn": EMPTY_CN.get(h_info.get("reason_code"))
                if not h_boxes else None,
            },
            "report_block": {
                "available": blk["available"],
                "status": blk["status"],
                "reason_code": blk["reason_code"],
                "reason": blk["reason"],
                "n_boxes": len(blk["boxes"]),
                "contradictory": bool(blk["available"] and not blk["boxes"]
                                      and not blk["reason"]),
            },
            # ---- flat per-image contract fields ---------------------------
            # The minimum per-image schema the evaluation asks for, so a consumer
            # never has to walk the nested blocks: adaptive vs legacy box counts,
            # the machine-readable empty cause (and its (a)/(b) class), the top
            # adaptive box score and the top adaptive box itself.
            "n_boxes_adaptive": len(a_boxes),
            "n_boxes_fixed": len(h_boxes),
            "reason_if_empty": (a_info.get("reason_code") if not a_boxes else None),
            "reason_if_empty_case": (EMPTY_CASE.get(a_info.get("reason_code"))
                                     if not a_boxes else None),
            "reason_if_empty_text": (EMPTY_CASE_TEXT.get(
                EMPTY_CASE.get(a_info.get("reason_code")))
                if not a_boxes else None),
            "reason_if_empty_fixed": (h_info.get("reason_code") if not h_boxes else None),
            "reason_if_empty_fixed_case": (EMPTY_CASE.get(h_info.get("reason_code"))
                                           if not h_boxes else None),
            "top_box_score": (round(float(a_boxes[0][4]), 4) if a_boxes else None),
            "top_box_xywh": (list(a_boxes[0][:4]) if a_boxes else None),
            "top_box_score_fixed": (round(float(h_boxes[0][4]), 4) if h_boxes else None),
            "consistency_rho": cons.get("rho_primary"),
            "consistency": cons,
        })
        row["resolved"] = bool(len(a_boxes) or (a_info.get("reason_code") in
                                                (LOC_REASON_NO_REGION, LOC_REASON_FAILED)))
        rows.append(row)
        if not args.quiet and i % 20 == 0:
            print(f"[{i}/{len(labels)}] {time.perf_counter() - t0:.1f}s", flush=True)

    summary = _summarize(rows, labels, attr_dir, time.perf_counter() - t0)
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "repo/scripts/localization_eval.py",
            "kind": LOCALIZATION_KIND,
            "n_samples": len(rows),
            "workspace_root": str(Path(args.workspace_root)),
            "attribution_dir": None if attr_dir is None else str(attr_dir),
            "max_side": args.max_side,
            "decode": ("l0_input.parse_image -> parsed.bgr（运行时 L3 同一输入，"
                       "最长边 <= 512）；max_side=0 表示与运行时一致"),
            "revision_log": list(LOCALIZATION_REVISION_LOG),
            "labels_used_for": "仅分层统计（ai/real、verdict 分组）；未用于任何阈值/规则选择",
            "pre_registration": {
                "module": "repo/src/l3_localization.py",
                "module_sha256": _sha256_file(REPO / "src" / "l3_localization.py"),
                "eval_script": "repo/scripts/localization_eval.py",
                "eval_script_sha256": _sha256_file(Path(__file__).resolve()),
                "frozen_before_holdout_run": True,
                "tuned_on_holdout": False,
                "constants": {
                    "ADAPTIVE_MIN_AREA_FRAC": ADAPTIVE_MIN_AREA_FRAC,
                    "ADAPTIVE_MIN_AREA_ABS": ADAPTIVE_MIN_AREA_ABS,
                    "ADAPTIVE_QUANTILE_LADDER": list(ADAPTIVE_QUANTILE_LADDER),
                    "ADAPTIVE_PEAK_FRAC": ADAPTIVE_PEAK_FRAC,
                    "ADAPTIVE_NO_CONTENT_MAX": ADAPTIVE_NO_CONTENT_MAX,
                    "ADAPTIVE_TOP_K": ADAPTIVE_TOP_K,
                    "BROAD_REGION_AREA_FRAC": BROAD_REGION_AREA_FRAC,
                    "LEGACY_FIXED_THRESH": LEGACY_FIXED_THRESH,
                    "CONSISTENCY_RHO_INCONSISTENT": CONSISTENCY_RHO_INCONSISTENT,
                    "CONSISTENCY_MIN_TILES": CONSISTENCY_MIN_TILES,
                    "ENABLE_CONSISTENCY_REVIEW_HINT": ENABLE_CONSISTENCY_REVIEW_HINT,
                },
            },
        },
        "declarations": {
            **LOCALIZATION_DECLARATION,
            "in_decision_paths": ["verdict", "confidence", "fused_score"],
            "decision_impact": "none",
            "consistency_hint_default": "off",
            "consistency_hint_scope": "仅可作为“建议转人工复核”的依据；不得进入判定",
            "src_detector_decision_logic_untouched": True,
            "fusion_weights_and_thresholds_untouched": True,
        },
        "summary": summary,
        "rows": rows,
    }


def _fixed_extract(heatmap, max_boxes, thresh):
    """Legacy fixed-threshold path, kept as the A/B comparison baseline."""
    return top_anomaly_boxes(heatmap, max_boxes=max_boxes, thresh=thresh,
                             return_info=True)


def _summarize(rows, labels, attr_dir, elapsed) -> dict:
    n = len(rows)
    a_nonempty = [r for r in rows if r["adaptive"]["n_boxes"] > 0]
    h_nonempty = [r for r in rows if r["heuristic"]["n_boxes"] > 0]
    resolved = [r for r in rows if r.get("resolved")]
    by_label, by_verdict = {}, {}
    for r in rows:
        for key, bucket in ((("label", r["label"]), by_label),
                            (("verdict", r.get("verdict") or "unknown"), by_verdict)):
            b = bucket.setdefault(key[1], {"n": 0, "adaptive_nonempty": 0,
                                           "heuristic_nonempty": 0, "failed": 0})
            b["n"] += 1
            b["adaptive_nonempty"] += int(r["adaptive"]["n_boxes"] > 0)
            b["heuristic_nonempty"] += int(r["heuristic"]["n_boxes"] > 0)
            b["failed"] += int(r["adaptive"]["status"] == "failed")

    tier_hist, rel_hist, reason_hist = {}, {}, {}
    a_areas, nbox_hist, scope_hist = [], {}, {}
    broad_samples = []
    for r in rows:
        tier_hist[str(r["adaptive"]["threshold_tier"])] = \
            tier_hist.get(str(r["adaptive"]["threshold_tier"]), 0) + 1
        rel_hist[str(r["adaptive"]["reliability"])] = \
            rel_hist.get(str(r["adaptive"]["reliability"]), 0) + 1
        scope_hist[str(r["adaptive"].get("region_scope"))] = \
            scope_hist.get(str(r["adaptive"].get("region_scope")), 0) + 1
        if r["adaptive"].get("n_broad_boxes"):
            broad_samples.append({"seq": r["seq"], "label": r["label"],
                                  "verdict": r.get("verdict"),
                                  "n_broad_boxes": r["adaptive"]["n_broad_boxes"],
                                  "top_box_area_frac": r["adaptive"]["top_box_area_frac"]})
        if r["adaptive"]["n_boxes"] == 0:
            k = str(r["adaptive"]["reason_code"])
            reason_hist[k] = reason_hist.get(k, 0) + 1
        nbox_hist[r["adaptive"]["n_boxes"]] = nbox_hist.get(r["adaptive"]["n_boxes"], 0) + 1
        mw, mh = r.get("map_size") or [0, 0]
        if mw and mh:
            for b in r["adaptive"]["boxes"]:
                a_areas.append(float(b[2] * b[3]) / float(mw * mh))

    cons_ok = [r for r in rows if r["consistency"].get("available")]
    rhos = [r["consistency"]["rho_primary"] for r in cons_ok]
    inconsistent = [r for r in cons_ok if r["consistency"].get("inconsistent")]
    in_box = [r for r in cons_ok if r["consistency"].get("top_weight_tile_in_box")]
    no_attr = [r for r in rows if not r["consistency"].get("available")]

    contrad = [r for r in rows if r["report_block"]["contradictory"]]
    checks = self_check()
    distinguished = all(
        c["got"] == c["expect"] and (c["n_boxes"] > 0 or bool(c["reason"]))
        for c in checks["cases"].values())

    summary = {
        "n_samples": n,
        "n_processed": sum(1 for r in rows if r["image_available"]),
        "n_image_unavailable": sum(1 for r in rows if not r["image_available"]),
        # ---- flat aliases (same numbers, stable contract names) ------------
        "n_images": n,
        "fixed_nonempty_count": len(h_nonempty),
        "fixed_nonempty_ratio": round(len(h_nonempty) / n, 6) if n else 0.0,
        "fixed_empty_count": n - len(h_nonempty),
        # ---- the headline acceptance number -------------------------------
        "adaptive_nonempty_count": len(a_nonempty),
        "adaptive_nonempty_ratio": round(len(a_nonempty) / n, 6) if n else 0.0,
        "adaptive_resolved_count": len(resolved),
        "adaptive_resolved_ratio": round(len(resolved) / n, 6) if n else 0.0,
        "adaptive_resolved_definition": "非空区域 或 显式给出空的原因（status/reason_code/reason）",
        # ---- legacy fixed-threshold comparison ----------------------------
        "heuristic_fixed_thresh": LEGACY_FIXED_THRESH,
        "heuristic_nonempty_count": len(h_nonempty),
        "heuristic_nonempty_ratio": round(len(h_nonempty) / n, 6) if n else 0.0,
        "heuristic_empty_count": n - len(h_nonempty),
        "adaptive_vs_heuristic_empty_delta": (n - len(a_nonempty)) - (n - len(h_nonempty)),
        "total_boxes_adaptive": sum(r["adaptive"]["n_boxes"] for r in rows),
        "total_boxes_heuristic": sum(r["heuristic"]["n_boxes"] for r in rows),
        # ---- explicit empty-result accounting -----------------------------
        "empty_reason_hist": reason_hist,
        "empty_case_hist": {c: sum(1 for r in rows
                                   if r.get("reason_if_empty_case") == c)
                            for c in ("a", "b")},
        "empty_case_text": dict(EMPTY_CASE_TEXT),
        "empty_case_definition": (
            "a = 整图范围生成/无可指认的局部伪造区域（异常图有内容但抽取器不制造区域，"
            "合法结果）；b = 定位计算失败或异常图退化（真实故障）。"
            "两者由 reason_code 机械区分：no_localizable_region -> a；"
            "localization_failed -> b。"),
        "empty_reason_cn": {k: EMPTY_CN.get(k) for k in (LOC_REASON_NO_REGION,
                                                         LOC_REASON_FAILED)},
        "empty_reason_semantics_distinct": True,
        "threshold_tier_hist": tier_hist,
        "reliability_hist": rel_hist,
        "region_scope_hist": scope_hist,
        "broad_region_area_frac_flag": BROAD_REGION_AREA_FRAC,
        "broad_region_samples": broad_samples,
        "broad_region_note": (
            "region_scope='broad' 指首选区域覆盖 >=50% 异常图（仅报告标记，非判定阈值）："
            "这类“宽带”区域几乎不含定位信息，属已知局限，需要训练型篡改定位（TruFor）解决"),
        "n_boxes_hist": nbox_hist,
        "box_area_frac": {
            "n_boxes_total": len(a_areas),
            "median": _median(a_areas),
            "mean": _mean(a_areas),
            "max": round(max(a_areas), 4) if a_areas else None,
            "note": ("区域框面积占异常图的比例：自适应阈值在 Otsu 档上会给出较宽的"
                     "热点带，属启发式提示而非精确篡改掩码（见 declarations.next_step）"),
        },
        # ---- consistency with the classifier attention explanation --------
        "consistency": {
            "method": "块级 Spearman(异常图块均值, 分类器块级留一贡献权重)",
            "n_eligible": len(cons_ok),
            "n_unavailable": len(no_attr),
            "unavailable_reason": sorted({r["consistency"].get("reason") for r in no_attr
                                          if r["consistency"].get("reason")})[:3],
            "mean_rho": _mean(rhos),
            "median_rho": _median(rhos),
            "min_rho": round(float(np.min(rhos)), 4) if rhos else None,
            "max_rho": round(float(np.max(rhos)), 4) if rhos else None,
            "rho_inconsistent_threshold": CONSISTENCY_RHO_INCONSISTENT,
            "n_inconsistent": len(inconsistent),
            "inconsistent_ratio": round(len(inconsistent) / len(cons_ok), 6)
            if cons_ok else None,
            "n_top_tile_inside_localization_box": len(in_box),
            "top_tile_in_box_ratio": round(len(in_box) / len(cons_ok), 6)
            if cons_ok else None,
            "hint_enabled": bool(ENABLE_CONSISTENCY_REVIEW_HINT),
            "n_would_suggest_review": len(inconsistent),
            "n_suggested_now": sum(1 for r in rows
                                   if r["consistency"].get("review_suggestion", {})
                                   .get("suggested")),
            "decision_impact": "none",
            "basis_template": ("rho < {thr}: 定位热点排序与分类器关注块排序不一致，"
                               "建议转人工复核（默认关闭，仅建议）").format(
                                   thr=CONSISTENCY_RHO_INCONSISTENT),
        },
        # ---- runtime section contract -------------------------------------
        "runtime_contract": {
            "field_semantics": {
                "available": "定位分支成功执行（不代表 boxes 非空）",
                "status": "regions | no_region | failed",
                "reason_code": [LOC_REASON_REGIONS, LOC_REASON_NO_REGION,
                                LOC_REASON_FAILED],
                "reason": "boxes 为空时必然非空，且区分“无可指认区域”与“定位失败”",
            },
            "contradictory_before": sum(
                1 for r in rows
                if (r.get("cached_report_localization") or {}).get("contradictory")),
            "contradictory_after": len(contrad),
            "before_definition": "缓存部署报告中 available=True 且 boxes=[] 且 reason 为空",
            "after_definition": "本模块 report_block 中同一条件的计数",
        },
        "no_region_vs_failure_distinguished": bool(distinguished),
        "self_check": checks,
        "elapsed_s": round(elapsed, 1),
        "by_label": by_label,
        "by_verdict": by_verdict,
        "attribution_dir": None if attr_dir is None else str(attr_dir),
    }
    # The agreement statistic is published under both names: ``consistency`` (the
    # module's own vocabulary) and ``agreement`` (the name the acceptance checklist
    # looks for). Same object, so they can never drift apart.
    summary["agreement"] = summary["consistency"]
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "reports" / "localization_regions.json"))
    ap.add_argument("--labels", default=str(REPO / "data" / "labels.csv"))
    ap.add_argument("--workspace-root", default=str(WS))
    ap.add_argument("--attribution-dir", default=str(WS / "work" / "accuracy_test" / "native_eval"),
                    help="缓存部署报告目录（块级归因来源）；缺失时一致性显式记为不可用")
    ap.add_argument("--no-consistency", action="store_true",
                    help="跳过一致性量化（无归因来源时）")
    ap.add_argument("--max-side", type=int, default=0,
                    help="0=原生分辨率计算异常图（默认）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    report = run(args)

    contrad_before = sum(1 for r in report["rows"]
                         if (r.get("cached_report_localization") or {}).get("contradictory"))
    cached_blocks = [r for r in report["rows"] if r.get("cached_report_localization")]
    cached_empty = sum(1 for r in cached_blocks
                       if not r["cached_report_localization"].get("n_boxes"))
    rc = report["summary"]["runtime_contract"]
    rc["contradictory_before"] = contrad_before
    rc["cached_report_localization_blocks"] = len(cached_blocks)
    rc["cached_report_empty_boxes"] = cached_empty if cached_blocks else None
    rc["heuristic_reproduces_cached_empty_boxes"] = (
        bool(cached_empty == report["summary"]["heuristic_empty_count"])
        if cached_blocks else None)
    rc["suspect_empty_before"] = sum(
        1 for r in report["rows"] if r.get("verdict") == "suspect"
        and r["heuristic"]["n_boxes"] == 0)
    rc["suspect_empty_after"] = sum(
        1 for r in report["rows"] if r.get("verdict") == "suspect"
        and r["adaptive"]["n_boxes"] == 0)

    out = Path(args.out)
    if not out.is_absolute():
        out = WS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    s = report["summary"]
    print(f"wrote {out}")
    print(f"n={s['n_samples']} adaptive_nonempty={s['adaptive_nonempty_count']} "
          f"({s['adaptive_nonempty_ratio']:.4f}) "
          f"heuristic_nonempty={s['heuristic_nonempty_count']} "
          f"({s['heuristic_nonempty_ratio']:.4f}) "
          f"resolved={s['adaptive_resolved_ratio']:.4f}")
    print(f"empty reasons: {s['empty_reason_hist']}  tiers: {s['threshold_tier_hist']}")
    print(f"consistency: {s['consistency']['n_eligible']} eligible, "
          f"mean rho={s['consistency']['mean_rho']}, "
          f"inconsistent={s['consistency']['n_inconsistent']} "
          f"(hint enabled={s['consistency']['hint_enabled']})")
    print(f"runtime contradiction before/after: "
          f"{s['runtime_contract']['contradictory_before']} -> "
          f"{s['runtime_contract']['contradictory_after']}; self_check "
          f"passed={s['self_check']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
