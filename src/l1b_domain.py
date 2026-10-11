"""L1b domain forensics: camera-capture vs digital-render evidence.

Rationale (manual ch.2.4 / 3.3). In claim-material fraud the dominant surface is
NOT "which generator produced this" but "was this material captured by a camera
(paper document, phone photo) or produced digitally (rendered bill, screenshot,
inpainted region)". The two pipelines leave complementary traces:

  camera  : sensor noise floor, uneven illumination, paper texture, perspective
            blur gradient, JPEG re-compression of a noisy source
  render  : near-zero noise in flat areas, perfectly uniform backgrounds,
            crisp uniform edges, no illumination gradient, high white-point mass

These are cheap zero-weight measurements: they need no model and are directly
explainable in the report, which is what the "判定依据可说明" requirement asks for.

All features are computed at FULL resolution when available (noise estimates are
destroyed by resizing), falling back to the L0-normalized image.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES_B: list[str] = []


def _reg(names):
    FEATURE_NAMES_B.extend(names)
    return names


_reg(["b_nearWhiteFrac", "b_nearBlackFrac", "b_flatFrac", "b_flatNoise",
      "b_texNoise", "b_noiseRatio", "b_noiseUniform", "b_illumGradient",
      "b_sharpMean", "b_sharpCV", "b_colorfulness", "b_satMean", "b_satStd",
      "b_hfRatio", "b_spectralSlope", "b_spectralPeak", "b_edgeWidth",
      "b_chanNoiseCorr", "b_laplaceKurt", "b_grayLevelCount"])

assert len(FEATURE_NAMES_B) == 20, len(FEATURE_NAMES_B)


def _grayscale(bgr: np.ndarray) -> np.ndarray:
    # BT.601 luma without cv2 dependency ordering issues
    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def extract_l1b_features(parsed) -> dict:
    """Return the 20-dim camera-vs-render feature dict."""
    import cv2

    out = {k: 0.0 for k in FEATURE_NAMES_B}
    if not getattr(parsed, "ok", False):
        return out
    bgr = getattr(parsed, "bgr_full", None)
    if bgr is None:
        bgr = getattr(parsed, "bgr", None)
    if bgr is None:
        return out

    # keep the noise analysis honest but bounded (full-res documents can be 4000px)
    h, w = bgr.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    bgr = bgr.astype(np.uint8) if bgr.dtype != np.uint8 else bgr
    gray = _grayscale(bgr)

    out["b_nearWhiteFrac"] = float((gray >= 245).mean())
    out["b_nearBlackFrac"] = float((gray <= 25).mean())
    # distinct 8-bit gray levels: bincount is O(n) where np.unique sorts (this was
    # the single most expensive L1b operation on 1.6 MP inputs)
    _hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256)
    out["b_grayLevelCount"] = float(int((_hist > 0).sum()))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    flat_thr = float(np.percentile(mag, 25))
    flat = mag <= flat_thr
    out["b_flatFrac"] = float(flat.mean())

    # high-frequency residual = proxy for sensor/compression noise
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    res = gray - blur
    flat_noise = float(res[flat].std()) if flat.any() else float(res.std())
    tex_noise = float(res[~flat].std()) if (~flat).any() else float(res.std())
    out["b_flatNoise"] = flat_noise
    out["b_texNoise"] = tex_noise
    out["b_noiseRatio"] = flat_noise / (tex_noise + 1e-6)

    # noise spatial uniformity: rendered/flat images have a very even noise map
    # Vectorised with box filters (same windows/stride as the original Python loop,
    # so feature values are unchanged) - the loop was the dominant L1b cost
    # (P95 535 ms vs the manual's 350 ms layer target).
    win, step = 32, 16
    n_y = len(range(0, max(1, res.shape[0] - win + 1), step))
    n_x = len(range(0, max(1, res.shape[1] - win + 1), step))
    if n_y > 0 and n_x > 0:
        mean_win = cv2.boxFilter(res, ddepth=-1, ksize=(win, win), normalize=True,
                                 borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(res * res, ddepth=-1, ksize=(win, win), normalize=True,
                                borderType=cv2.BORDER_REFLECT)
        var_win = np.maximum(mean_sq - mean_win * mean_win, 0.0)
        std_win = np.sqrt(var_win, dtype=np.float32)
        # the original loop sampled the window anchored at its top-left corner
        sampled = std_win[win // 2::step, win // 2::step][:n_y, :n_x].ravel()
        if sampled.size:
            v = np.asarray(sampled, dtype=np.float32)
            out["b_noiseUniform"] = float(v.std() / (v.mean() + 1e-6))

    # illumination gradient: std of a heavily blurred version (low-frequency energy)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(4.0, min(h, w) / 24.0))
    out["b_illumGradient"] = float(low.std())

    # sharpness uniformity across tiles (camera blur varies with depth of field)
    # Box-filter variance over 64x64 tiles, identically to the original tile loop.
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tile = 64
    ny_t = (max(1, gray.shape[0] - tile + 1) + tile - 1) // tile
    nx_t = (max(1, gray.shape[1] - tile + 1) + tile - 1) // tile
    if ny_t > 0 and nx_t > 0:
        m_lap = cv2.boxFilter(lap, ddepth=-1, ksize=(tile, tile), normalize=True,
                              borderType=cv2.BORDER_REFLECT)
        m_lap2 = cv2.boxFilter(lap * lap, ddepth=-1, ksize=(tile, tile), normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        var_lap = np.maximum(m_lap2 - m_lap * m_lap, 0.0)
        tv = var_lap[tile // 2::tile, tile // 2::tile][:ny_t, :nx_t].ravel()
        if tv.size:
            tv = np.asarray(tv, dtype=np.float32)
            out["b_sharpMean"] = float(tv.mean())
            out["b_sharpCV"] = float(tv.std() / (tv.mean() + 1e-6))

    # Hasler-Susstrunk colorfulness
    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    out["b_colorfulness"] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                                  + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    out["b_satMean"], out["b_satStd"] = float(sat.mean()), float(sat.std())

    # frequency-domain: high-frequency energy ratio, radial spectral slope, peaks
    g2 = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    g2 = g2 - g2.mean()
    win2 = np.outer(np.hanning(512), np.hanning(512))
    mag_spec = np.abs(np.fft.fftshift(np.fft.fft2(g2 * win2)))
    cy, cx = 256, 256
    yy, xx = np.mgrid[0:512, 0:512]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    prof = np.bincount(rr.ravel(), weights=mag_spec.ravel()) / (np.bincount(rr.ravel()) + 1e-6)
    n = len(prof)
    lo, hi = prof[1:n // 4].mean(), prof[n // 4:n // 2].mean()
    out["b_hfRatio"] = float(hi / (lo + 1e-6))
    k = np.arange(1, n)
    p = prof[1:]
    if p.std() > 1e-9:
        slope = float(np.polyfit(np.log(k + 1), np.log(p + 1e-6), 1)[0])
        out["b_spectralSlope"] = slope
    mid = prof[n // 8: n // 2]
    out["b_spectralPeak"] = float((mid.max() / (mid.mean() + 1e-6))) if mid.size else 0.0

    # edge width (blur estimate): mean gradient magnitude at strong edges
    strong = mag >= np.percentile(mag, 99)
    out["b_edgeWidth"] = float(1.0 / (mag[strong].mean() + 1e-6)) if strong.any() else 0.0

    # per-channel noise correlation: sensor noise is channel-correlated after demosaic.
    # Computed with explicit moments instead of np.corrcoef on 3 stacked full-res
    # planes - same Pearson r, far less memory traffic.
    rb = cv2.GaussianBlur(bgr[:, :, 0].astype(np.float32), (5, 5), 1.2)
    gb = cv2.GaussianBlur(bgr[:, :, 1].astype(np.float32), (5, 5), 1.2)
    bb = cv2.GaussianBlur(bgr[:, :, 2].astype(np.float32), (5, 5), 1.2)
    nB = (bgr[:, :, 0] - rb).ravel()
    nG = (bgr[:, :, 1] - gb).ravel()
    nR = (bgr[:, :, 2] - bb).ravel()

    def _pearson(a, b):
        am, bm = a.mean(), b.mean()
        da, db = a - am, b - bm
        denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
        return float((da * db).sum() / denom) if denom > 1e-12 else 0.0

    if nB.std() > 1e-9 and nG.std() > 1e-9 and nR.std() > 1e-9:
        out["b_chanNoiseCorr"] = (_pearson(nB, nG) + _pearson(nB, nR) + _pearson(nG, nR)) / 3.0
        del nB, nG, nR

    rv = res.ravel()
    rv = rv - rv.mean()
    s2 = float((rv ** 2).mean()) + 1e-12
    out["b_laplaceKurt"] = float((rv ** 4).mean() / (s2 ** 2))

    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}


def feature_vector_b(feat: dict) -> np.ndarray:
    return np.array([feat.get(k, 0.0) for k in FEATURE_NAMES_B], dtype=np.float64)
