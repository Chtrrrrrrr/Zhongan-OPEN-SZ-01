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

    Mirrors ``l2_clip_tiles.extract_tiled_features`` so a perturbed aggregate is
    expressed in exactly the layout the ``combined`` head was trained on.
    """
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0), feats.std(axis=0)])


def tile_features_detailed(extractor, bgr_full: np.ndarray, max_side: int = 1600) -> dict:
    """Per-tile CLIP features + the tile geometry behind ``extract_tiled_features``.

    Same grid / resize / aggregation as ``l2_clip_tiles.extract_tiled_features``,
    but the individual tile features are kept so each tile can be occluded later.
    """
    import cv2
    from l2_clip_tiles import N_GRID, OVERLAP, tile_boxes, TILED_DIM_FACTOR
    h, w = bgr_full.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(bgr_full, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    else:
        img = bgr_full
    hh, ww = img.shape[:2]
    boxes, tiles = [], []
    for (x0, y0, x1, y1) in tile_boxes(hh, ww, N_GRID, OVERLAP):
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        boxes.append((int(x0), int(y0), int(x1), int(y1)))
        tiles.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
    if not tiles:
        boxes, tiles = [(0, 0, int(ww), int(hh))], [img]
    feats = np.asarray(extractor.batch(tiles, batch_size=len(tiles)), dtype=np.float32)
    return {"feats": feats, "boxes": boxes, "size": (int(ww), int(hh)),
            "grid": int(N_GRID), "overlap": float(OVERLAP),
            "dim_factor": int(TILED_DIM_FACTOR)}


def tile_contribution_map(context: np.ndarray, tile_agg: np.ndarray, logit_fn,
                          feats: np.ndarray, boxes: list, size: tuple[int, int],
                          grid: int = 3, eps: float = 1e-9) -> dict:
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
    if t_n < 2:
        return {"ok": False, "reason": f"可用分块数 {t_n} < 2，无法做块级留一归因",
                "heatmap": None, "tiles": [], "full_logit": None}
    full_logit = float(logit_fn(np.concatenate([context, tile_agg])))
    deltas = np.zeros(t_n, dtype=np.float64)
    for t in range(t_n):
        occ = feats.copy()
        occ[t] = np.delete(feats, t, axis=0).mean(axis=0)
        deltas[t] = full_logit - float(logit_fn(np.concatenate([context, _aggregate(occ)])))

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


def tile_contribution_attribution(extractor, bgr_full: np.ndarray, context: np.ndarray,
                                  tile_agg: np.ndarray, logit_fn,
                                  max_side: int = 1600) -> dict:
    """Attention explanation bundle for the report (manual ch.9.2 fields).

    Everything returned here is JSON-serializable except ``heatmap`` (the raw
    float map, encoded to PNG by the caller).
    """
    detail = tile_features_detailed(extractor, bgr_full, max_side=max_side)
    res = tile_contribution_map(context=context, tile_agg=tile_agg, logit_fn=logit_fn,
                                feats=detail["feats"], boxes=detail["boxes"],
                                size=detail["size"], grid=detail["grid"])
    res["geometry"] = {"size": detail["size"], "grid": detail["grid"],
                       "overlap": detail["overlap"], "max_side": max_side,
                       "note": "块级归因在原生分辨率分块上进行（最长边≤1600），"
                               "与 L3 异常图使用的模型输入尺度不同"}
    return res
