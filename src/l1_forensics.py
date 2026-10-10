"""L1 low-cost forensic feature extraction (OPEN-SZ-01).

47-dim handcrafted feature vector, reproducing the pilot baseline described in
project manual ch.11.1:
  color statistics (9) | texture & gradient (8) | noise residual (6)
  autocorrelation (4)  | 8x8 blockiness (3)      | channel correlation (3)
  multi-scale residual (4) | PNG container (9) | aspect (1)

Container features are read from the raw file bytes (unscaled); pixel features
are computed on the L0-normalized image (longest side 512).
These features are AUXILIARY EVIDENCE (L1) - never the sole decision criterion.
"""
from __future__ import annotations

import numpy as np
import cv2

from l0_input import ParsedInput

FEATURE_NAMES: list[str] = []


def _register(names):
    FEATURE_NAMES.extend(names)
    return names


# ---- feature name registry (order = vector order) ----
_register(["meanR", "meanG", "meanB", "stdR", "stdG", "stdB", "cvR", "cvG", "cvB"])
_register(["gradMean", "gradStd", "lapVar", "entropy", "grayStd", "edgeDensity",
           "texContrast", "texHomogeneity"])
_register(["resMean", "resStd", "resAbsMax", "resKurt", "resSkew", "waveHH"])
_register(["ac1", "ac2", "ac4", "ac8"])
_register(["blockiness", "blockRatio", "blockDctHF"])
_register(["corrRG", "corrGB", "corrRB"])
_register(["msRes2", "msRes4", "msRes8", "msResRatio"])
_register(["nidat", "hasSrgb", "hasExif", "hasItxt", "hasGama", "hasChrm",
           "bytes", "ow", "oh"])
_register(["aspect"])

assert len(FEATURE_NAMES) == 47, len(FEATURE_NAMES)


def _autocorr(x: np.ndarray, lag: int) -> float:
    a = x[:-lag, :].ravel() if lag < x.shape[0] else x.ravel()
    b = x[lag:, :].ravel() if lag < x.shape[0] else x.ravel()
    if a.size < 16 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _blockiness(gray: np.ndarray, bs: int = 8) -> tuple[float, float]:
    """Mean |diff| across block boundaries vs inside blocks (8x8 grid)."""
    g = gray.astype(np.float32)
    # vertical boundaries
    vb = np.abs(g[:, bs - 1::bs] - g[:, bs::bs]).mean() if g.shape[1] > bs else 0.0
    hb = np.abs(g[bs - 1::bs, :] - g[bs::bs, :]).mean() if g.shape[0] > bs else 0.0
    # interior diffs (off-boundary)
    vi = np.abs(np.diff(g, axis=1)).mean()
    hi = np.abs(np.diff(g, axis=0)).mean()
    boundary = float((vb + hb) / 2)
    interior = float((vi + hi) / 2 + 1e-6)
    return boundary, boundary / interior


def _block_dct_hf(gray: np.ndarray, bs: int = 8) -> float:
    """Mean high-frequency DCT energy of 8x8 blocks."""
    h, w = gray.shape[0] // bs * bs, gray.shape[1] // bs * bs
    if h == 0 or w == 0:
        return 0.0
    g = gray[:h, :w].astype(np.float32)
    blocks = g.reshape(h // bs, bs, w // bs, bs).swapaxes(1, 2).reshape(-1, bs, bs)
    hf = 0.0
    step = max(1, len(blocks) // 256)  # cap work
    for blk in blocks[::step]:
        d = cv2.dct(blk)
        hf += float(np.abs(d[4:, 4:]).mean())
    return hf / (len(blocks[::step]) + 1e-6)


def _glcm_props(gray: np.ndarray) -> tuple[float, float]:
    """Small-footprint contrast & homogeneity from a quantized co-occurrence matrix."""
    q = (gray // 32).astype(np.int64)  # 8 gray levels
    P = np.zeros((8, 8), dtype=np.float64)
    np.add.at(P, (q[:, :-1].ravel(), q[:, 1:].ravel()), 1)
    if P.sum() == 0:
        return 0.0, 0.0
    P /= P.sum()
    i, j = np.indices(P.shape)
    contrast = float((P * (i - j) ** 2).sum())
    homogeneity = float((P / (1.0 + (i - j) ** 2)).sum())
    return contrast, homogeneity


def extract_l1_features(parsed: ParsedInput) -> dict:
    """Return {feature_name: float} for a parsed image. NaN-safe."""
    if not parsed.ok or parsed.bgr is None:
        return {k: float("nan") for k in FEATURE_NAMES}

    bgr = parsed.bgr
    f: dict[str, float] = {}
    B, G, R = [bgr[:, :, i].astype(np.float64) for i in range(3)]
    for name, ch in (("R", R), ("G", G), ("B", B)):
        m, s = float(ch.mean()), float(ch.std())
        f[f"mean{name}"] = m
        f[f"std{name}"] = s
        f[f"cv{name}"] = s / (m + 1e-6)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gf = gray.astype(np.float64)

    # texture & gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    f["gradMean"] = float(gm.mean())
    f["gradStd"] = float(gm.std())
    f["lapVar"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    f["entropy"] = float(-(p * np.log2(p)).sum())
    f["grayStd"] = float(gf.std())
    edges = cv2.Canny(gray, 60, 160)
    f["edgeDensity"] = float((edges > 0).mean())
    f["texContrast"], f["texHomogeneity"] = _glcm_props(gray)

    # noise residual (image - gaussian)
    blur = cv2.GaussianBlur(gf, (5, 5), 1.2)
    res = gf - blur
    f["resMean"] = float(res.mean())
    f["resStd"] = float(res.std())
    f["resAbsMax"] = float(np.abs(res).max())
    rs = res.ravel()
    rs_c = rs - rs.mean()
    s4 = (rs_c ** 4).mean() / ((rs_c ** 2).mean() ** 2 + 1e-12)
    s3 = (rs_c ** 3).mean() / ((rs_c ** 2).mean() ** 1.5 + 1e-12)
    f["resKurt"] = float(s4)
    f["resSkew"] = float(s3)
    try:
        import pywt
        coeffs = pywt.dwt2(gray.astype(np.float32), "haar")
        f["waveHH"] = float(np.abs(coeffs[1][2]).mean())
    except Exception:
        f["waveHH"] = 0.0

    # autocorrelation of residual
    for lag in (1, 2, 4, 8):
        f[f"ac{lag}"] = _autocorr(res, lag)

    # 8x8 blockiness
    bnd, ratio = _blockiness(gray)
    f["blockiness"] = bnd
    f["blockRatio"] = ratio
    f["blockDctHF"] = _block_dct_hf(gray)

    # channel correlation
    for n, a, b in (("RG", R, G), ("GB", G, B), ("RB", R, B)):
        af, bf = a.ravel(), b.ravel()
        f[f"corr{n}"] = float(np.corrcoef(af, bf)[0, 1]) if af.std() > 0 and bf.std() > 0 else 0.0

    # multi-scale residual energies
    prev_energy = None
    for k, factor in ((2, 2), (4, 4), (8, 8)):
        small = cv2.resize(gf, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (gf.shape[1], gf.shape[0]), interpolation=cv2.INTER_CUBIC)
        e = float(np.abs(gf - up).mean())
        f[f"msRes{k}"] = e
        if factor == 8 and prev_energy:
            f["msResRatio"] = e / (prev_energy + 1e-6)
        prev_energy = e
    f.setdefault("msResRatio", 0.0)

    # PNG container features (raw bytes; robust to missing chunks)
    chunks = parsed.png_chunks or []
    types = [c[0] for c in chunks]
    f["nidat"] = float(types.count("IDAT"))
    f["hasSrgb"] = float("sRGB" in types)
    f["hasExif"] = float("eXIf" in types)
    f["hasItxt"] = float("iTXt" in types)
    f["hasGama"] = float("gAMA" in types)
    f["hasChrm"] = float("cHRM" in types)
    f["bytes"] = float(parsed.raw_bytes)
    f["ow"] = float(parsed.orig_width)
    f["oh"] = float(parsed.orig_height)
    f["aspect"] = float(parsed.orig_width) / (float(parsed.orig_height) + 1e-6)

    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in f.items()}


def feature_vector(feat: dict) -> np.ndarray:
    return np.array([feat.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)
