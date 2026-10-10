"""L3 tamper localization (manual ch.8.4).

Primary path would be TruFor (weights unavailable offline -> fallback chain):
  1. TruFor anomaly map            [optional, if weights present]
  2. zero-weight forensic maps     [always available]
       - ELA (error level analysis)
       - local noise-variance inconsistency
  3. Grad-CAM attention of L2 heads [auxiliary explanation, NOT localization]

The three kinds of output are kept strictly separate in the report
(manual ch.3.5 terminology discipline).
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
    """Grad-CAM attention of an L2 classifier. EXPLANATION ONLY - not localization."""
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
