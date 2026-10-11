"""L2 extension: tiled CLIP features at native resolution.

Why: a single 224x224 resize of a 3000px claim document throws away the very
detail that separates a digitally rendered bill (crisp, uniform, no sensor
noise, perfectly aligned table rules) from a phone photo of paper (paper
texture, perspective, uneven lighting, JPEG-of-noise). We therefore score
overlapping native-resolution tiles with the same frozen CLIP backbone and
aggregate mean / max / std, which keeps the backbone frozen and adds no
trainable parameters to the shared trunk.

Output dim = 3 x D (mean, max, std over tiles).
"""
from __future__ import annotations

import numpy as np

N_GRID = 3           # 3x3 overlapping tiles
OVERLAP = 0.25


def tile_boxes(h: int, w: int, n: int = N_GRID, overlap: float = OVERLAP):
    """Overlapping nxn tile boxes (x0, y0, x1, y1) in pixel coords.

    ``n - (n-1)*overlap`` is the number of tile-widths the grid spans, which is
    fractional for some (n, overlap) pairs; tile edges are rounded outward so the grid
    always covers the full image instead of clipping the right/bottom edge.
    """
    import math
    span = max(1.0, n - (n - 1) * overlap)
    th, tw = h / span, w / span
    step_y, step_x = th * (1 - overlap), tw * (1 - overlap)
    boxes = []
    for i in range(n):
        for j in range(n):
            y0, x0 = i * step_y, j * step_x
            y1 = min(h, int(math.ceil(y0 + th)))
            x1 = min(w, int(math.ceil(x0 + tw)))
            y0i = max(0, int(math.floor(y1 - th)))
            x0i = max(0, int(math.floor(x1 - tw)))
            boxes.append((x0i, y0i, x1, y1))
    return boxes


def extract_tiled_features(extractor, bgr_full: np.ndarray, n: int = N_GRID,
                           max_side: int = 1600) -> np.ndarray:
    """Tiled CLIP features for one image -> [3*D] vector (mean, max, std)."""
    import cv2
    h, w = bgr_full.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(bgr_full, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    else:
        img = bgr_full
    hh, ww = img.shape[:2]
    tiles = []
    for (x0, y0, x1, y1) in tile_boxes(hh, ww, n):
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        tiles.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
    if not tiles:
        tiles = [img]
    feats = extractor.batch(tiles, batch_size=len(tiles))      # [T, D]
    return np.concatenate([feats.mean(axis=0), feats.max(axis=0), feats.std(axis=0)])


TILED_DIM_FACTOR = 3
