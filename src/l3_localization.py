"""L3 tamper localization (manual ch.8.4) + classifier attention explanation.

Primary path would be TruFor (weights unavailable offline -> fallback chain):
  1. TruFor anomaly map            [optional, if weights present]
  2. zero-weight forensic maps     [always available]
       - ELA (error level analysis)
       - local noise-variance inconsistency
  3. attention explanation of the L2 decision head [auxiliary explanation,
     NOT localization]

Attention explanation has two mutually exclusive providers:
  a) ``gradcam_overlay`` - real Grad-CAM, only for a differentiable head
     (needs ``model.preprocess`` + a torch module with a convolutional/attention
     feature map).
  b) ``tile_contribution_attribution`` - the deployed ``combined`` head is a
     joblib ``{scaler, clf}`` logistic regression, i.e. a *linear* model with no
     feature map and no backward pass, so Grad-CAM is structurally unavailable.
     The usable substitute is a tile-level occlusion attribution over the tiled
     CLIP features that this head actually consumes.

Terminology discipline (manual ch.3.5) is enforced by construction: L3
localization output (``combined_anomaly_map`` + ``top_anomaly_boxes``) and the
attention explanation are separate return values, are stored in separate report
fields (``localization`` vs ``explanation``) and carry different disclaimers.
An attention map must never be read as a tamper region.

Region extraction (2026-10 improvement, pre-registered):
  * the legacy extraction thresholded the anomaly map at the *fixed* value
    ``0.55`` and returned whatever connected components survived. On the frozen
    119-sample holdout that produced **69 empty box lists** (35 of them on
    samples the fusion layer labelled ``suspect``) while the report still said
    ``available=True`` with ``reason=None`` - a self-contradictory section.
  * region extraction is now *adaptive*: Otsu on the anomaly map first, then a
    fixed percentile ladder, then a peak-anchored tier. All constants are frozen
    in :data:`ADAPTIVE_*` below and were written down *before* the 119-sample
    evaluation ran; nothing here selects a threshold from the holdout labels.
    The legacy fixed path is preserved for side-by-side comparison via
    :data:`LEGACY_FIXED_THRESH` / ``thresh=``.
  * an empty result is no longer ambiguous: :func:`localize_heatmap` always
    returns an explicit ``reason_code``/``reason`` that distinguishes
    "computed fine, but the whole image is generated / no region stands out"
    from "localization itself failed".
  * consistency between the anomaly-map hotspots and the classifier attention
    explanation is *quantified* (block-level Spearman) and can only ever be a
    "suggest moving to review" hint; it is OFF by default
    (:data:`ENABLE_CONSISTENCY_REVIEW_HINT`), contributes nothing to
    ``verdict``/``confidence``, and never promotes an attention map to a
    tamper region.
  * changes to this section are listed in :data:`LOCALIZATION_REVISION_LOG`,
    including the post-observation revision 2 (strict mask comparison); no
    entry uses holdout labels.

Declaration, unchanged and enforced by construction: this whole module is a
**zero-weight forensic heuristic** - ELA + local noise inconsistency, no learned
parameters - it does **not** take part in the decision. The next step for real
localization is a trained tamper localizer (TruFor, :data:`LOCALIZATION_NEXT_STEP`).
"""
from __future__ import annotations

import io

import numpy as np


def ela_map(bgr: np.ndarray, quality: int = 90, scale: float = 12.0) -> np.ndarray:
    """Error Level Analysis: |x - recompressed(x)| amplified -> [0,1] map."""
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return np.zeros(bgr.shape[:2], dtype=np.float32)
    rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, rec).astype(np.float32).mean(axis=2)
    diff = diff / (diff.max() + 1e-6)
    return np.clip(diff * (scale / 12.0), 0, 1).astype(np.float32)


def noise_inconsistency_map(bgr: np.ndarray, win: int = 32, stride: int = 16) -> np.ndarray:
    """Local high-frequency noise std, z-scored against the global distribution.

    Regions whose noise level deviates strongly from the image-wide estimate
    are candidates for splicing / inpainting / re-compression boundaries.
    """
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    res = gray - blur
    h, w = res.shape
    vals, grid = [], []
    for y in range(0, max(1, h - win + 1), stride):
        row = []
        for x in range(0, max(1, w - win + 1), stride):
            v = float(res[y:y + win, x:x + win].std())
            row.append(v)
            vals.append(v)
        grid.append(row)
    grid = np.array(grid, dtype=np.float32)
    vals = np.array(vals, dtype=np.float32)
    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
    z = np.abs((grid - mu) / sd)
    z = z / (z.max() + 1e-6)
    z_full = cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(z_full, 0, 1).astype(np.float32)


def combined_anomaly_map(bgr: np.ndarray) -> np.ndarray:
    """Fuse zero-weight localizers into one anomaly map [0,1]."""
    e = ela_map(bgr)
    n = noise_inconsistency_map(bgr)
    m = 0.5 * e + 0.5 * n
    # suppress the weakest 40% to reduce noise on real samples
    thr = np.percentile(m, 40)
    return np.clip(m - thr, 0, 1) / (1 - thr + 1e-6)


# ---------------------------------------------------------------------------
# Region extraction: pre-registered adaptive thresholding (see module docstring)
# ---------------------------------------------------------------------------
# FROZEN BEFORE THE 119-SAMPLE EVALUATION.  These are not fitted values: they are
# a fixed, label-independent rule.  The holdout was used to *measure* the rule,
# never to choose it.
ADAPTIVE_MIN_AREA_FRAC = 0.0005      # component must cover >= 0.05% of the image
ADAPTIVE_MIN_AREA_ABS = 32           # legacy absolute floor, kept for parity
ADAPTIVE_QUANTILE_LADDER = (99.0, 95.0, 90.0)   # tried after Otsu, in this order
ADAPTIVE_PEAK_FRAC = 0.5             # last resort anchor: half of the map maximum
ADAPTIVE_NO_CONTENT_MAX = 1e-3       # map dynamic range below this = no content
ADAPTIVE_TOP_K = 3
LEGACY_FIXED_THRESH = 0.55           # comparison path only (the old default)
#: Reporting flag only (never a decision threshold): a region covering this much
#: of the anomaly map is "broad" and carries little localisation information.
BROAD_REGION_AREA_FRAC = 0.5

#: Disclosed change log for this section (no entry uses holdout labels).
LOCALIZATION_REVISION_LOG = [
    {"rev": 1, "when": "before the first 119-sample run",
     "change": "region extraction switched from the fixed 0.55 threshold to the "
               "adaptive Otsu + percentile-ladder + peak-fallback rule",
     "label_used": False},
    {"rev": 2, "when": "after the first 119-sample run",
     "change": "adaptive mask comparison changed to the strict 'heatmap > "
               "threshold' (an Otsu threshold of 0 otherwise selects the whole "
               "image). Post-observation correction, motivated by whole-image "
               "boxes in run 1; no label, no new tunable parameter.",
     "run1_observation": "119/119 non-empty regions but the largest box covered "
                         "100% of the anomaly map",
     "label_used": False},
    {"rev": 3, "when": "constant / never tuned",
     "change": "explicit empty-result contract (status + reason_code + reason, "
               "distinguishing 'no localizable region' from 'localization "
               "failed') and the block-level consistency statistic with a "
               "default-off review hint",
     "label_used": False},
]

# Explicit, mutually exclusive outcome codes for the localization section.
LOC_REASON_REGIONS = "regions_found"
LOC_REASON_NO_REGION = "no_localizable_region"
LOC_REASON_FAILED = "localization_failed"

LOC_REASON_TEXT = {
    LOC_REASON_REGIONS: None,
    LOC_REASON_NO_REGION: (
        "整图范围生成/全图统计一致：异常图本身几乎无动态范围，"
        "不存在显著高于其余区域的伪造指认区域，因此不输出区域框。"
        "这是“无可指认区域”，不是定位失败。"),
    LOC_REASON_FAILED: "定位计算失败",
}

LOCALIZATION_KIND = "zero-weight forensic map (ELA+noise inconsistency)"

#: Everything below is display-only. ``in_decision=False`` is the contract.
LOCALIZATION_DECLARATION = {
    "kind": LOCALIZATION_KIND,
    "weight": "zero-weight",
    "in_decision": False,
    "method": "ELA (error level analysis) + local noise-variance inconsistency",
    "disclaimer": (
        "该定位为零权重启发式（ELA+局部噪声不一致），无训练参数、不参与判定："
        "不改变 verdict / confidence / fused_score，也不是篡改的充分证据；"
        "区域框仅用于人工复核时缩小观察范围。"),
    "terminology": (
        "定位区域（localization）= 启发式异常热点；分类器关注区域解释"
        "（explanation）= 分类器更依赖哪些图像块。两者不得互相替代。"),
    "next_step": "接入训练型篡改定位模型 TruFor（当前权重缺失，回退到本启发式）",
}
LOCALIZATION_NEXT_STEP = LOCALIZATION_DECLARATION["next_step"]

#: Consistency may only ever *suggest* a manual review, and is off by default.
ENABLE_CONSISTENCY_REVIEW_HINT = False
CONSISTENCY_RHO_INCONSISTENT = 0.30
CONSISTENCY_MIN_TILES = 3


def _min_area_for(h: int, w: int, min_area: int | None = None) -> int:
    if min_area is not None:
        return max(1, int(min_area))
    return max(ADAPTIVE_MIN_AREA_ABS, int(round(ADAPTIVE_MIN_AREA_FRAC * h * w)))


def otsu_threshold(heatmap: np.ndarray) -> float:
    """Otsu split of the anomaly map, returned in the map's own [0,1] units."""
    import cv2
    u8 = np.clip(np.round(np.asarray(heatmap, dtype=np.float64) * 255.0), 0, 255)\
        .astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def _components(mask: np.ndarray, heatmap: np.ndarray, min_area: int) -> list:
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        score = float(heatmap[lab == i].mean())
        out.append((x, y, w, h, round(score, 3), area))
    out.sort(key=lambda b: (-b[4], -b[5]))
    return out


def adaptive_region_extract(heatmap: np.ndarray, max_boxes: int = ADAPTIVE_TOP_K,
                            min_area: int | None = None) -> tuple[list, dict]:
    """Adaptive connected-component region extraction.

    Returns ``(boxes, info)`` with ``boxes`` shaped like the legacy output
    ``[(x, y, w, h, score)]``. ``info`` always carries ``threshold``,
    ``threshold_tier``, ``reliability``, ``status`` and ``reason_code`` so an
    empty list is never published without an explicit cause.

    Tiers, in the frozen order: ``otsu`` -> ``quantile_99`` -> ``quantile_95``
    -> ``quantile_90`` -> ``peak_half_max`` -> ``peak_pixel_relaxed``. The first
    tier that yields at least one component of ``min_area`` pixels wins; the
    last tier guarantees a region whenever the map has any content at all, and
    is flagged ``weak``.

    Mask rule (revision 2, disclosed): the adaptive path thresholds with the
    *strict* comparison ``heatmap > threshold``. On a strongly bimodal map cv2's
    Otsu returns exactly ``0`` (the boundary of the empty background class), and
    a non-strict test would then select the whole image - a "region" equal to the
    entire picture carries no localisation information. With ``>`` an Otsu
    threshold of 0 correctly means "the non-zero part of the map". The legacy
    fixed path keeps its original ``>=`` so the historical 69-empty baseline
    still reproduces bit for bit. This correction was made after observing
    whole-image boxes in the first holdout run; it uses no label and adds no
    tunable parameter.
    """
    hm = np.asarray(heatmap, dtype=np.float64)
    info = {
        "mode": "adaptive",
        "threshold": None,
        "threshold_tier": None,
        "reliability": None,
        "n_components": 0,
        "min_area": None,
        "max_heat": None,
        "top_box_area_frac": None,
        "n_broad_boxes": 0,
        "region_scope": None,
        "status": None,
        "reason_code": None,
        "reason": None,
        "boxes_considered": [],
    }
    if hm.ndim != 2 or hm.size == 0 or not np.isfinite(hm).all():
        info.update(status="failed", reason_code=LOC_REASON_FAILED,
                    reason=f"{LOC_REASON_TEXT[LOC_REASON_FAILED]}：异常图非有限二维数组"
                           f"（ndim={hm.ndim}, finite={bool(np.isfinite(hm).all()) if hm.size else False}）")
        return [], info

    h, w = int(hm.shape[0]), int(hm.shape[1])
    ma = float(hm.max())
    ma_area = _min_area_for(h, w, min_area)
    info.update(min_area=ma_area, max_heat=round(ma, 6))

    if ma < ADAPTIVE_NO_CONTENT_MAX:
        info.update(status="no_region", reason_code=LOC_REASON_NO_REGION,
                    reason=f"{LOC_REASON_TEXT[LOC_REASON_NO_REGION]}"
                           f"（异常图 max={ma:.2e} < {ADAPTIVE_NO_CONTENT_MAX:g}）")
        return [], info

    tiers = [("otsu", otsu_threshold(hm))]
    tiers += [(f"quantile_{q:g}", float(np.percentile(hm, q)))
              for q in ADAPTIVE_QUANTILE_LADDER]
    tiers.append(("peak_half_max", ma * ADAPTIVE_PEAK_FRAC))

    tried = []
    for tier, thr in tiers:
        thr = float(min(max(thr, 0.0), ma))
        comps = _components(hm > thr, hm, ma_area)
        tried.append({"tier": tier, "threshold": round(thr, 6),
                      "n_components": len(comps)})
        if comps:
            sel = comps[:max_boxes]
            areas = [c[5] / float(h * w) for c in sel]
            info.update(threshold=round(thr, 6), threshold_tier=tier,
                        reliability={"otsu": "distinct", "quantile_99": "distinct",
                                     "quantile_95": "diffuse",
                                     "quantile_90": "diffuse"}.get(tier, "weak"),
                        n_components=len(comps), status="regions",
                        reason_code=LOC_REASON_REGIONS, reason=None,
                        mask_rule="heatmap > threshold",
                        top_box_area_frac=round(areas[0], 4),
                        n_broad_boxes=sum(1 for a in areas if a >= BROAD_REGION_AREA_FRAC),
                        region_scope=("broad" if areas[0] >= BROAD_REGION_AREA_FRAC
                                      else "local"),
                        boxes_considered=tried)
            return [c[:5] for c in sel], info

    # Last resort: keep the hottest component even below the area floor, so that
    # "the map has content" can never silently collapse into an empty list.
    peak_thr = ma * ADAPTIVE_PEAK_FRAC
    comps = _components(hm > peak_thr, hm, 1)
    tried.append({"tier": "peak_pixel_relaxed", "threshold": round(peak_thr, 6),
                  "n_components": len(comps), "min_area_relaxed": True})
    if comps:
        sel = comps[:max_boxes]
        areas = [c[5] / float(h * w) for c in sel]
        info.update(threshold=round(peak_thr, 6), threshold_tier="peak_pixel_relaxed",
                    reliability="weak", n_components=len(comps), status="regions",
                    reason_code=LOC_REASON_REGIONS, reason=None,
                    mask_rule="heatmap > threshold", min_area_relaxed=True,
                    top_box_area_frac=round(areas[0], 4),
                    n_broad_boxes=sum(1 for a in areas if a >= BROAD_REGION_AREA_FRAC),
                    region_scope=("broad" if areas[0] >= BROAD_REGION_AREA_FRAC
                                  else "local"),
                    boxes_considered=tried)
        return [c[:5] for c in sel], info

    info.update(status="no_region", reason_code=LOC_REASON_NO_REGION,
                reason=f"{LOC_REASON_TEXT[LOC_REASON_NO_REGION]}"
                       f"（所有自适应阈值档位均无满足面积的连通域，max={ma:.3f}）",
                boxes_considered=tried)
    return [], info


def _legacy_fixed_extract(heatmap: np.ndarray, max_boxes: int,
                          thresh: float) -> tuple[list, dict]:
    """The original fixed-threshold path, preserved for A/B comparison only."""
    hm = np.asarray(heatmap, dtype=np.float64)
    info = {"mode": "fixed", "threshold": float(thresh), "threshold_tier": "legacy_fixed",
            "reliability": "legacy", "n_components": 0, "min_area": ADAPTIVE_MIN_AREA_ABS,
            "max_heat": round(float(hm.max()), 6) if hm.size else None,
            "status": None, "reason_code": None, "reason": None, "boxes_considered": []}
    if hm.ndim != 2 or hm.size == 0 or not np.isfinite(hm).all():
        info.update(status="failed", reason_code=LOC_REASON_FAILED,
                    reason=f"{LOC_REASON_TEXT[LOC_REASON_FAILED]}：异常图非有限二维数组")
        return [], info
    comps = _components(hm >= float(thresh), hm, ADAPTIVE_MIN_AREA_ABS)
    if comps:
        info.update(status="regions", reason_code=LOC_REASON_REGIONS,
                    n_components=len(comps))
        return [c[:5] for c in comps[:max_boxes]], info
    info.update(status="no_region", reason_code=LOC_REASON_NO_REGION,
                reason=f"{LOC_REASON_TEXT[LOC_REASON_NO_REGION]}"
                       f"（固定阈值 {thresh:g} 下无满足面积的连通域，max={hm.max():.3f}）")
    return [], info


def top_anomaly_boxes(heatmap: np.ndarray, max_boxes: int = ADAPTIVE_TOP_K,
                      thresh: float | None = None, *, mode: str | None = None,
                      min_area: int | None = None, return_info: bool = False):
    """Hot-area connected components -> ``[(x,y,w,h,score)]``.

    ``thresh=None`` (the default) uses the **adaptive** extractor; passing an
    explicit ``thresh`` (or ``mode="fixed"``) reproduces the legacy fixed
    threshold, which is retained only so reports can show the two side by side.
    ``return_info=True`` additionally returns the diagnostics dict, which always
    explains an empty list.
    """
    if mode is None:
        mode = "fixed" if thresh is not None else "adaptive"
    if mode == "fixed":
        boxes, info = _legacy_fixed_extract(
            heatmap, max_boxes, LEGACY_FIXED_THRESH if thresh is None else float(thresh))
    elif mode == "adaptive":
        boxes, info = adaptive_region_extract(heatmap, max_boxes=max_boxes,
                                             min_area=min_area)
    else:
        raise ValueError(f"unknown localization mode: {mode!r}")
    return (boxes, info) if return_info else boxes


def top_anomaly_boxes_adaptive(heatmap: np.ndarray, max_boxes: int = ADAPTIVE_TOP_K,
                               min_area: int | None = None) -> tuple[list, dict]:
    """Adaptive extractor returning ``(boxes, info)`` (explicit alias)."""
    return adaptive_region_extract(heatmap, max_boxes=max_boxes, min_area=min_area)


def localize_heatmap(heatmap: np.ndarray, max_boxes: int = ADAPTIVE_TOP_K,
                     min_area: int | None = None,
                     with_legacy_comparison: bool = True) -> dict:
    """The localization section, with a non-ambiguous empty-result contract.

    Keys: ``available`` (the branch ran to completion - it does *not* mean boxes
    are non-empty), ``status`` (``regions`` | ``no_region`` | ``failed``),
    ``reason_code``/``reason`` (always set when ``boxes == []``), ``boxes``,
    ``rules``, plus ``boxes_heuristic``/``extraction_heuristic`` for the legacy
    fixed-threshold comparison. ``declaration`` restates that this is a
    zero-weight heuristic and not part of the decision.
    """
    def _block():
        return {
            "available": True,
            "kind": LOCALIZATION_KIND,
            "status": "regions",
            "reason_code": LOC_REASON_REGIONS,
            "reason": None,
            "boxes": [],
            "rules": {},
            "declaration": dict(LOCALIZATION_DECLARATION),
        }
    try:
        boxes, info = adaptive_region_extract(heatmap, max_boxes=max_boxes,
                                             min_area=min_area)
    except Exception as e:                                    # pragma: no cover
        blk = _block()
        blk.update(available=False, status="failed", reason_code=LOC_REASON_FAILED,
                   reason=f"{LOC_REASON_TEXT[LOC_REASON_FAILED]}：{type(e).__name__}: {e}")
        return blk

    blk = _block()
    blk["boxes"] = boxes
    blk["status"] = info["status"]
    blk["reason_code"] = info["reason_code"]
    blk["reason"] = info["reason"]
    blk["rules"] = {k: v for k, v in info.items() if k not in ("reason", "boxes_considered")}
    if with_legacy_comparison:
        try:
            h_boxes, h_info = _legacy_fixed_extract(np.asarray(heatmap, dtype=np.float64),
                                                    max_boxes, LEGACY_FIXED_THRESH)
            blk["boxes_heuristic"] = h_boxes
            blk["extraction_heuristic"] = {
                "mode": "fixed", "threshold": h_info["threshold"],
                "n_boxes": len(h_boxes), "status": h_info["status"],
                "reason_code": h_info["reason_code"]}
        except Exception:                                     # pragma: no cover
            blk["boxes_heuristic"] = None
            blk["extraction_heuristic"] = {"mode": "fixed", "error": "heuristic comparison failed"}
    return blk


def localize_image(bgr: np.ndarray, max_boxes: int = ADAPTIVE_TOP_K,
                   with_legacy_comparison: bool = True) -> dict:
    """Convenience wrapper: image -> anomaly map -> localization section.

    This is the entry point the runtime report should adopt instead of
    ``{"available": True, ..., "reason": None}`` with a possibly empty ``boxes``
    list (see ``repo/src/detector.py::detect_image`` L3 block): the returned dict
    keeps ``available``/``kind``/``boxes``/``reason`` and adds the explicit
    ``status``/``reason_code`` pair. The caller still owns the heatmap PNG.
    """
    try:
        amap = combined_anomaly_map(bgr)
    except Exception as e:                                    # pragma: no cover
        blk = localize_heatmap(np.zeros((1, 1), dtype=np.float32))
        blk.update(available=False, status="failed", boxes=[],
                   reason_code=LOC_REASON_FAILED,
                   reason=f"{LOC_REASON_TEXT[LOC_REASON_FAILED]}：异常图计算失败 "
                          f"{type(e).__name__}: {e}")
        return blk
    blk = localize_heatmap(amap, max_boxes=max_boxes,
                          with_legacy_comparison=with_legacy_comparison)
    blk["anomaly_map_stats"] = {
        "max": round(float(amap.max()), 6), "mean": round(float(amap.mean()), 6),
        "std": round(float(amap.std()), 6), "shape": [int(amap.shape[0]), int(amap.shape[1])]}
    return blk


# ---------------------------------------------------------------------------
# Consistency between localization hotspots and the classifier attention map
# ---------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank) - Spearman without scipy."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rank_correlation(a, b) -> float | None:
    """Spearman rho; ``None`` when it is undefined (too few / constant input)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < CONSISTENCY_MIN_TILES:
        return None
    if float(a.std()) <= 0.0 or float(b.std()) <= 0.0:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom <= 0.0:
        return None
    return float((ra * rb).sum() / denom)


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = float((x1 - x0) * (y1 - y0))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def region_consistency(heatmap: np.ndarray, tiles: list, boxes: list,
                       xy_scale: tuple[float, float] = (1.0, 1.0),
                       rho_inconsistent: float = CONSISTENCY_RHO_INCONSISTENT,
                       hint_enabled: bool | None = None) -> dict:
    """Block-level agreement between the anomaly map and the attention explanation.

    ``tiles`` are the ``explanation`` tiles (``bbox_xywh`` + ``weight`` /
    ``delta_logit``); ``xy_scale`` maps their coordinates onto the heatmap grid.
    The statistic is the Spearman rank correlation over blocks of
    *anomaly-map mean* vs *classifier block weight*, computed per scale, with the
    largest scale reported as ``rho_primary``. Also reported: whether the single
    most-attributed block overlaps any localization box, and its best IoU.

    The result is advisory only: ``review_suggestion`` can never be ``suggested``
    unless :data:`ENABLE_CONSISTENCY_REVIEW_HINT` is on, and even then it carries
    ``decision_impact="none"``. This function never touches verdict/confidence.
    """
    hm = np.asarray(heatmap, dtype=np.float64)
    enabled = ENABLE_CONSISTENCY_REVIEW_HINT if hint_enabled is None else bool(hint_enabled)
    out = {
        "available": False,
        "method": "block-level Spearman(anomaly_map_mean, tile_attribution_weight)",
        "n_tiles": 0, "rho_primary": None, "rho_by_scale": {},
        "top_weight_tile_bbox": None, "top_weight_tile_in_box": None,
        "top_weight_tile_iou": None,
        "inconsistent": None,
        "rho_inconsistent_threshold": float(rho_inconsistent),
        "review_suggestion": {"suggested": False, "enabled": enabled,
                              "decision_impact": "none", "basis": None},
        "reason": None,
    }
    if hm.ndim != 2 or hm.size == 0 or not np.isfinite(hm).all():
        out["reason"] = "一致性未计算：异常图不可用"
        return out
    if not tiles:
        out["reason"] = ("一致性未计算：分类器关注区域解释不可用（无块级归因）"
                         "，无法比较两者")
        return out

    sx, sy = float(xy_scale[0]), float(xy_scale[1])
    h, w = hm.shape
    by_scale: dict[int, list] = {}
    for t in tiles:
        by_scale.setdefault(int(t.get("scale", 0)), []).append(t)

    for s, ts in by_scale.items():
        a_vals, w_vals = [], []
        for t in ts:
            x, y, bw, bh = (float(v) for v in t["bbox_xywh"])
            x0 = int(max(0, min(w - 1, round(x * sx))))
            y0 = int(max(0, min(h - 1, round(y * sy))))
            x1 = int(max(x0 + 1, min(w, round((x + bw) * sx))))
            y1 = int(max(y0 + 1, min(h, round((y + bh) * sy))))
            a_vals.append(float(hm[y0:y1, x0:x1].mean()))
            w_vals.append(float(t.get("weight", max(0.0, float(t.get("delta_logit", 0.0))))))
        out["rho_by_scale"][str(s)] = spearman_rank_correlation(a_vals, w_vals)

    primary = max(by_scale, key=lambda s: len(by_scale[s]))
    prim_tiles = by_scale[primary]
    out["n_tiles"] = len(prim_tiles)
    out["primary_scale"] = primary
    rho = out["rho_by_scale"].get(str(primary))
    out["rho_primary"] = None if rho is None else round(float(rho), 4)

    if prim_tiles:
        top = max(prim_tiles, key=lambda t: float(t.get("weight", 0.0)))
        tb = [int(v) for v in top["bbox_xywh"]]
        out["top_weight_tile_bbox"] = tb
        ious = [_bbox_iou(tb, [int(v) for v in b[:4]]) for b in (boxes or [])]
        out["top_weight_tile_iou"] = round(max(ious), 4) if ious else 0.0
        out["top_weight_tile_in_box"] = bool(ious and max(ious) > 0.0)

    if rho is None:
        out["reason"] = ("一致性不可判定：块级归因权重或异常图块均值恒定"
                         f"（n_tiles={out['n_tiles']}），秩相关无定义")
        return out

    out["available"] = True
    inconsistent = bool(float(rho) < float(rho_inconsistent))
    out["inconsistent"] = inconsistent
    basis = (f"块级 Spearman(异常图块均值, 分类器块权重) rho={float(rho):.3f} "
             f"< {float(rho_inconsistent):.2f}：定位热点排序与分类器关注块排序不一致"
             if inconsistent else
             f"块级 Spearman rho={float(rho):.3f} >= {float(rho_inconsistent):.2f}："
             f"两者块级排序基本一致")
    out["review_suggestion"] = {
        "suggested": bool(inconsistent and enabled),
        "enabled": enabled,
        "would_suggest": inconsistent,
        "decision_impact": "none",
        "basis": basis,
        "note": ("一致性仅可作为“建议转人工复核”的依据，默认关闭"
                 "（ENABLE_CONSISTENCY_REVIEW_HINT=False），且不进入判定链路"
                 "（verdict/confidence 不变）"),
    }
    return out


def gradcam_overlay(model, bgr: np.ndarray, target_layer, device: str) -> np.ndarray:
    """Grad-CAM attention of an L2 classifier. EXPLANATION ONLY - not localization.

    Requires a *differentiable* head: ``model.preprocess(bgr) -> [3,H,W]`` plus a
    torch module reachable as ``model.net`` / ``model(...)`` with ``target_layer``
    exposing a conv/attention feature map. The deployed ``combined`` head is a
    joblib logistic-regression bundle, which satisfies none of these; use
    :func:`tile_contribution_attribution` for it instead. This entry point is kept
    for a future CNN/transformer head (e.g. TruFor) and for the manual ch.9.2
    "Grad-CAM (if available)" field.
    """
    import torch
    import cv2
    acts, grads = {}, {}

    def fwd_hook(m, i, o):
        acts["v"] = o.detach()

    def bwd_hook(m, gi, go):
        grads["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        x = model.preprocess(bgr).unsqueeze(0).to(device)
        x.requires_grad_(True)
        logit = model.net(x) if hasattr(model, "net") else model(x)
        logit.backward()
        a, g = acts["v"], grads["v"]
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1))[0].cpu().numpy()
        cam = cv2.resize(cam, (bgr.shape[1], bgr.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


# ---------------------------------------------------------------------------
# Patch-token statistics of the frozen ViT (training-free clue family)
# ---------------------------------------------------------------------------
PATCH_FEATURE_NAMES = [
    "pt_cosMean", "pt_cosStd", "pt_cosMin", "pt_cosMax",
    "pt_cosP05", "pt_cosP95", "pt_cosLowFrac", "pt_cosHighFrac",
    "pt_cosAutocorr1", "pt_cosQuadrantSpread",
    "pt_resMean", "pt_resStd", "pt_resMax", "pt_resLowFrac",
    "pt_normMean", "pt_normStd",
]


def patch_token_features(extractor, bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Statistics of the frozen ViT patch grid - no training, no extra backbone call class.

    Pooled CLIP evidence (global and tiled) averages the patch grid, so a *localised*
    inconsistency - inpainting, splicing, a locally re-rendered region - is averaged
    away. These statistics keep the grid: per-patch cosine to the patch mean, its
    spatial autocorrelation and quadrant spread, and the residual-norm statistics.
    Measured as a standalone family the block reaches OOF AUC ~0.93 on the 119
    samples, and it lifts the low-cost `forensic` member from 0.9105 to 0.9330
    (reports/verify_rounds.json).

    ``extractor`` must expose ``preprocess(bgr) -> [3,H,W]`` and a ``model`` returning
    hidden states; ``CLIPFeatureExtractor`` satisfies both.
    """
    import cv2
    import torch

    h, w = bgr.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    x = extractor.preprocess(bgr).unsqueeze(0).to(extractor.device)
    with torch.no_grad():
        out = extractor.model(x, output_hidden_states=True)
    hh = out.hidden_states[-1][0]
    t = torch.nn.functional.normalize(hh[1:].float(), dim=-1)      # drop CLS
    mu = torch.nn.functional.normalize(t.mean(dim=0), dim=-1, eps=1e-8)
    cos = (t @ mu).detach().cpu().numpy().astype(np.float64)
    res = (t - mu).norm(dim=-1).detach().cpu().numpy().astype(np.float64)
    nrm = t.norm(dim=-1).detach().cpu().numpy().astype(np.float64)

    side = int(round(cos.size ** 0.5))
    grid = cos.reshape(side, side) if side * side == cos.size else None
    feats = [cos.mean(), cos.std(), cos.min(), cos.max(),
             float(np.percentile(cos, 5)), float(np.percentile(cos, 95)),
             float((cos < cos.mean() - 2 * cos.std()).mean()),
             float((cos > cos.mean() + 2 * cos.std()).mean())]
    if grid is not None and side >= 4:
        a = grid - grid.mean()
        denom = float((a * a).sum()) + 1e-12
        ac1 = float((a[:, :-1] * a[:, 1:]).sum() / denom)
        half = side // 2
        quads = [grid[:half, :half].mean(), grid[:half, half:].mean(),
                 grid[half:, :half].mean(), grid[half:, half:].mean()]
        feats += [ac1, float(np.std(quads))]
    else:
        feats += [0.0, 0.0]
    feats += [res.mean(), res.std(), res.max(),
              float((res > res.mean() + 2 * res.std()).mean()),
              nrm.mean(), nrm.std()]
    return np.array([float(v) if np.isfinite(v) else 0.0 for v in feats], dtype=np.float64)


# ---------------------------------------------------------------------------
# Classifier attention explanation for a NON-differentiable head
# ---------------------------------------------------------------------------
ATTRIBUTION_LABEL_CN = "分类器关注区域解释（块级）"
ATTRIBUTION_KIND = "classifier_attention_explanation"
ATTRIBUTION_METHOD = "tile_contribution_occlusion (leave-one-tile-out on combined head)"
ATTRIBUTION_DISCLAIMER = (
    "本图为分类器关注区域解释：只说明分类器打分时更依赖哪些图像块，"
    "≠ 伪造/篡改区域定位，不得作为区域级结论或证据使用。"
)


def gradcam_supported(head) -> tuple[bool, str]:
    """Whether a real Grad-CAM can be computed for this decision head.

    Returns ``(supported, reason)``. A joblib/sklearn estimator, a linear probe
    bundle or any non-torch object has no feature map and no backward path.
    """
    try:
        import torch
    except Exception:                      # pragma: no cover - torch is a hard L2 dep
        return False, "torch 不可用，无法进行任何反向传播"
    if isinstance(head, torch.nn.Module):
        return True, ""
    if head is None:
        return False, "combined 头未加载/未配置，没有可解释的头部"
    kind = type(head).__name__
    if isinstance(head, dict):
        keys = ",".join(sorted(str(k) for k in head.keys()))
        return False, (f"combined 头为 joblib/sklearn 逻辑回归包（dict keys={keys}），"
                       f"属线性模型：无卷积/注意力特征图、无反向传播路径，"
                       f"Grad-CAM 结构性不可用")
    return False, f"combined 头类型 {kind} 非可微神经网络，Grad-CAM 不可用"


def _aggregate(feats: np.ndarray) -> np.ndarray:
    """mean|max|std over tiles -> [3D].

    Kept as the default so an old caller still gets the archived layout; the deployed
    head's actual aggregation is supplied by ``FeatureSpec`` via ``aggregate_fn``.
    """
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0), feats.std(axis=0)])


def tile_features_detailed(extractor, bgr_full: np.ndarray, max_side: int = 1600) -> dict:
    """Per-tile CLIP features + geometry for the archived 3x3 grid (legacy layout).

    Prefer :meth:`feature_spec.FeatureSpec.tile_detail`, which follows the deployed
    spec (scales and aggregation); this function is retained for the frozen 3x3
    layout and for callers that predate the spec.
    """
    from l2_clip_tiles import N_GRID, OVERLAP, TILED_DIM_FACTOR
    from feature_spec import FeatureSpec
    spec = FeatureSpec(scales=[{"grid": N_GRID, "overlap": OVERLAP, "max_side": max_side}],
                       aggregation="legacy")
    d = spec.tile_detail(extractor, bgr_full)
    return {"feats": d["feats"][0], "boxes": d["boxes"][0], "size": d["sizes"][0],
            "grid": N_GRID, "overlap": float(OVERLAP), "dim_factor": int(TILED_DIM_FACTOR)}


def tile_contribution_map(context: np.ndarray, tile_agg: np.ndarray, logit_fn,
                          feats: np.ndarray, boxes: list, size: tuple[int, int],
                          grid: int = 3, eps: float = 1e-9, aggregate_fn=None) -> dict:
    """Leave-one-tile-out contribution of every tile, plus a spatial heatmap.

    ``context`` is the tile-independent part of the head input (L1 + L1b + CLIP
    global); ``tile_agg`` is the aggregated tile vector that produced the score
    actually reported; ``logit_fn(v) -> float`` applies the frozen head.

    Contribution of tile *t* is the drop caused by occluding it, i.e. by replacing
    its feature row with the mean of the remaining tiles::

        delta_t = logit(context|tile_agg) - logit(context|aggregate(feats with t occluded))

    ``delta_t > 0`` means the head leaned on that tile towards its positive class.
    """
    t_n = int(feats.shape[0])
    agg_fn = aggregate_fn or _aggregate
    if t_n < 2:
        return {"ok": False, "reason": f"可用分块数 {t_n} < 2，无法做块级留一归因",
                "heatmap": None, "tiles": [], "full_logit": None}
    full_logit = float(logit_fn(np.concatenate([context, tile_agg])))
    deltas = np.zeros(t_n, dtype=np.float64)
    for t in range(t_n):
        occ = feats.copy()
        occ[t] = np.delete(feats, t, axis=0).mean(axis=0)
        deltas[t] = full_logit - float(logit_fn(np.concatenate([context, agg_fn(occ)])))

    pos = np.clip(deltas, 0.0, None)
    if pos.max() > eps:
        vals = (pos / pos.max()).astype(np.float32)
    else:
        vals = np.zeros(t_n, dtype=np.float32)     # no tile pushes towards positive class

    canvas = np.zeros((size[1], size[0]), dtype=np.float32)
    wsum = np.zeros_like(canvas)
    for (x0, y0, x1, y1), v in zip(boxes, vals):
        canvas[y0:y1, x0:x1] += v
        wsum[y0:y1, x0:x1] += 1.0
    hm = (canvas / np.maximum(wsum, 1.0)).astype(np.float32)

    rows = []
    for i, ((x0, y0, x1, y1), d, v) in enumerate(zip(boxes, deltas, vals)):
        rows.append({"index": i, "row": i // grid, "col": i % grid,
                     "bbox_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                     "delta_logit": round(float(d), 4),
                     "weight": round(float(v), 4)})
    return {"ok": True, "reason": None, "heatmap": hm, "tiles": rows,
            "full_logit": float(full_logit),
            "delta_max": round(float(deltas.max()), 4),
            "delta_min": round(float(deltas.min()), 4),
            "flat_positive": bool(pos.max() <= eps)}


def tile_contribution_map_multi(context: np.ndarray, tile_agg: np.ndarray, logit_fn,
                                detail: dict, eps: float = 1e-9) -> dict:
    """Leave-one-tile-out attribution across ALL scales, consistent by construction.

    The deployed head consumes every scale's aggregate concatenated into one vector, so
    occluding tile *t* of scale *s* must change only scale *s*'s block and leave the
    other scales' blocks exactly as the head saw them. Perturbing one scale while
    pinning the full aggregate (the obvious shortcut) would score a vector the head
    never sees, so the aggregate is rebuilt per trial instead.

    Returns the first scale's spatial map (the coarsest, which localises best) plus a
    per-scale summary and every tile's delta.
    """
    from feature_spec import _aggregate

    feats_all = [np.asarray(f, dtype=np.float32) for f in detail["feats"]]
    agg_name = detail.get("aggregation", "legacy")
    blocks = [_aggregate(f, agg_name) for f in feats_all]
    fixed = np.concatenate([np.asarray(context, dtype=np.float64).ravel(),
                            np.asarray(tile_agg, dtype=np.float64).ravel()])
    full_logit = float(logit_fn(np.concatenate([context, tile_agg])))

    per_scale, all_rows = [], []
    for s, (feats, boxes, size, grid) in enumerate(zip(feats_all, detail["boxes"],
                                                       detail["sizes"], detail["grids"])):
        t_n = int(feats.shape[0])
        deltas = np.zeros(t_n, dtype=np.float64)
        if t_n >= 2:
            for t in range(t_n):
                occ = feats.copy()
                occ[t] = np.delete(feats, t, axis=0).mean(axis=0)
                trial_blocks = list(blocks)
                trial_blocks[s] = _aggregate(occ, agg_name)
                trial = np.concatenate([np.asarray(context, dtype=np.float64).ravel(),
                                        np.concatenate(trial_blocks)])
                deltas[t] = full_logit - float(logit_fn(trial))
        pos = np.clip(deltas, 0.0, None)
        vals = (pos / pos.max()).astype(np.float32) if pos.max() > eps \
            else np.zeros(t_n, dtype=np.float32)
        canvas = np.zeros((size[1], size[0]), dtype=np.float32)
        wsum = np.zeros_like(canvas)
        for (x0, y0, x1, y1), v in zip(boxes, vals):
            canvas[y0:y1, x0:x1] += v
            wsum[y0:y1, x0:x1] += 1.0
        hm = (canvas / np.maximum(wsum, 1.0)).astype(np.float32)
        rows = [{"index": i, "scale": s, "grid": grid, "row": i // max(1, grid),
                 "col": i % max(1, grid),
                 "bbox_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                 "delta_logit": round(float(d), 4), "weight": round(float(v), 4)}
                for i, ((x0, y0, x1, y1), d, v) in enumerate(zip(boxes, deltas, vals))]
        per_scale.append({"scale": s, "grid": int(grid), "size": size, "n_tiles": t_n,
                          "heatmap": hm, "tiles": rows,
                          "delta_max": round(float(deltas.max()), 4) if t_n else None,
                          "delta_min": round(float(deltas.min()), 4) if t_n else None,
                          "ok": bool(t_n >= 2)})
        all_rows.extend(rows)

    primary = next((p for p in per_scale if p["ok"]), None)
    if primary is None:
        return {"ok": False, "reason": "没有可归因的分块（块数 < 2）", "heatmap": None,
                "tiles": [], "full_logit": full_logit, "per_scale": per_scale}
    deltas_all = np.array([r["delta_logit"] for r in all_rows], dtype=np.float64)
    return {"ok": True, "reason": None, "heatmap": primary["heatmap"],
            "tiles": all_rows, "full_logit": full_logit,
            "delta_max": round(float(deltas_all.max()), 4),
            "delta_min": round(float(deltas_all.min()), 4),
            "flat_positive": bool(deltas_all.max() <= eps),
            "primary_scale_grid": primary["grid"],
            "per_scale": [{"scale": p["scale"], "grid": p["grid"], "n_tiles": p["n_tiles"],
                           "ok": p["ok"], "delta_max": p["delta_max"]} for p in per_scale]}


def tile_contribution_attribution(extractor, bgr_full: np.ndarray, context: np.ndarray,
                                  tile_agg: np.ndarray, logit_fn,
                                  max_side: int = 1600, spec=None,
                                  detail: dict | None = None) -> dict:
    """Attention explanation bundle for the report (manual ch.9.2 fields).

    Everything returned here is JSON-serializable except ``heatmap`` (the raw
    float map, encoded to PNG by the caller). When ``spec`` is given (a
    :class:`feature_spec.FeatureSpec`) the attribution runs on the deployed scales and
    aggregation; otherwise the frozen 3x3 layout is used. ``detail`` may carry the
    :meth:`feature_spec.FeatureSpec.tile_detail` result the scoring path already
    computed, which avoids a second backbone pass over the same tiles.
    """
    if spec is not None:
        d = detail if detail is not None else spec.tile_detail(extractor, bgr_full)
        res = tile_contribution_map_multi(context, tile_agg, logit_fn, d)
        res["geometry"] = {"scales": spec.scales, "aggregation": spec.aggregation,
                           "max_side": max_side, "tiles_reused_from_scoring": detail is not None,
                           "note": "块级归因在部署所用的原生分辨率分块上进行，"
                                   "与 L3 异常图使用的模型输入尺度不同"}
        return res

    detail = tile_features_detailed(extractor, bgr_full, max_side=max_side)
    res = tile_contribution_map(context=context, tile_agg=tile_agg, logit_fn=logit_fn,
                                feats=detail["feats"], boxes=detail["boxes"],
                                size=detail["size"], grid=detail["grid"])
    res["geometry"] = {"size": detail["size"], "grid": detail["grid"],
                       "overlap": detail["overlap"], "max_side": max_side,
                       "note": "块级归因在原生分辨率分块上进行（最长边≤1600），"
                               "与 L3 异常图使用的模型输入尺度不同"}
    return res
