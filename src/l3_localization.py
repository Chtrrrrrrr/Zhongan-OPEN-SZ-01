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


def top_anomaly_boxes(heatmap: np.ndarray, max_boxes: int = 3, thresh: float = 0.55):
    """Connected components over the hot area -> [(x,y,w,h,score)]."""
    import cv2
    mask = (heatmap >= thresh).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 32:
            continue
        score = float(heatmap[lab == i].mean())
        boxes.append((int(x), int(y), int(w), int(h), round(score, 3)))
    boxes.sort(key=lambda b: -b[4])
    return boxes[:max_boxes]


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
