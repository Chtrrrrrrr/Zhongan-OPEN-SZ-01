"""L1c: Chinese-document OCR + text-layer evidence (improvement task t6).

Why this layer exists
---------------------
The frozen stack has no text evidence at all: L1/L1b are pixel-forensic, L2 is CLIP.
The AI-generated claims it misses (003/029/030) are flat-render white "documents"
(invoice / receipt style), where the text itself is the most direct evidence. This
module extracts OCR text and a fixed, label-free block of text-consistency features.

Backend
-------
``rapidocr_onnxruntime`` 1.2.3 (PP-OCRv3 det + mobile cls + PP-OCRv3 rec, ONNX Runtime
CPU). Models ship inside the wheel, so after installation the extractor runs fully
offline. See work/improve/ocr_requirements.txt.

Label isolation (hard rule for this task)
-----------------------------------------
Nothing in this module reads a label. ``iter_document_rows`` reads
``data/meta.csv`` and keeps only ``seq`` and ``content`` (the label column is dropped
at load time and never enters a feature); ``data/labels.csv`` is never opened here.
The feature block below is a pure function of the decoded image pixels and the OCR
output.

Honest scope
------------
These features are *evidence*, not a validated detector. Whether they help is decided
out-of-fold in repo/reports/ocr_text_consistency.json, under the same folds/seeds as
the frozen deployment protocol; a negative result there must be reported as such.

Self-test
---------
    python repo/src/l1c_ocr.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
WS_DEFAULT = REPO.parent

# ---------------------------------------------------------------------------
# frozen parameter set (pre-registered constants; none was tuned on the samples)
# ---------------------------------------------------------------------------
PARAMS: dict = {
    "backend": "rapidocr_onnxruntime",
    "text_score_threshold": 0.5,        # backend default, kept explicit
    "use_angle_cls": True,              # labels import of the cls model
    "low_score": 0.5,                   # "unreliable recogniser output" cut
    "very_low_score": 0.7,              # secondary cut (reported separately)
    "max_side": 2000,                   # only downscale if larger (latency bound)
    "red_h_low": 10,                    # OpenCV H range for stamp red [0,10]
    "red_h_high": 170,                  # and [170,179]
    "red_min_s": 80,
    "red_min_v": 60,
    "red_min_area_px": 120,
    "red_dilate_px": 3,
    "ink_otsu_blur": 3,
    "cc_min_area_px": 1,
}

# Fixed feature order (never reorder: the report and the evaluator rely on it).
FEATURE_NAMES: tuple = (
    # --- recogniser volume / confidence -----------------------------------
    "ocr_has_text",              # 1 if the recogniser returned >= 1 box
    "n_boxes",
    "n_chars",
    "mean_score",
    "p10_score",
    "score_std",
    "frac_score_lt_50",
    "frac_score_lt_70",
    "char_density_per_mpx",      # recognised chars per megapixel
    "box_area_frac",             # summed box area / image area
    # --- layout / geometry consistency ------------------------------------
    "box_h_mean",
    "box_h_cv",
    "box_w_cv",
    "line_gap_cv",               # CV of vertical gaps between consecutive lines
    "left_edge_std_frac",        # std of box left edges, normalised by width
    "mean_aspect",               # mean box w/h
    # --- text-content legality (character-level, OCR-line-joining safe) ----
    "n_num_groups",              # number-like character groups in the recognised text
    "frac_amount_numbers",       # share of those that look like a monetary amount
    "has_date_like",             # a date pattern is present
    "frac_mixed_num_groups",     # share of digit groups carrying latin confusables
    "digit_ratio",
    "cjk_ratio",
    "frac_len1_boxes",           # single-character boxes (layout noise proxy)
    "char_entropy",              # Shannon entropy of the character histogram
    "has_currency",              # ￥ / ¥ / $ present in any recognised text
    # --- stamp / red-ink layer --------------------------------------------
    "red_frac",                  # share of red-ink pixels
    "red_region_count",
    "red_max_region_frac",
    "red_text_overlap_frac",     # share of text boxes whose centre sits on red ink
    # --- per-character stroke complexity (no reference text) --------------
    "ink_ratio_mean",            # binarised ink share inside text boxes
    "ink_ratio_cv",
    "stroke_cc_per_char",        # connected components per recognised character
    "stroke_laplacian_per_char",  # mean |Laplacian| per recognised character
)

# Character-level patterns. OCR returns whole text lines per box, so tokenising on
# whitespace would make "金额22.00" one token and a token-level amount regex would never
# fire (a design flaw found while validating on real invoices) - the numeric-group
# statistics below are insensitive to how boxes are split.
NUM_GROUP_RE = re.compile(r"\d[\d,.\-]*\d|\d")
AMOUNT_RE = re.compile(
    r"[¥￥$]\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"      # currency-signed
    r"|\d{1,3}(?:,\d{3})+\.\d{2}"                     # thousands-separated with cents
    r"|\d+\.\d{2}")                                   # plain cents
DATE_LIKE_RE = re.compile(
    r"\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
MIXED_NUM_RE = re.compile(r"\d+[OoIlIBbSsGgZz]+|\d*[OoIlIBbSsGgZz]+\d+")


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
_OCR = None


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def backend_info() -> dict:
    info = {"available": ocr_available(), "params": {k: PARAMS[k] for k in
            ("backend", "text_score_threshold", "use_angle_cls", "max_side")}}
    try:
        import rapidocr_onnxruntime as r
        import onnxruntime as ort
        info["rapidocr_onnxruntime"] = getattr(r, "__version__", "unknown")
        info["onnxruntime"] = ort.__version__
        model_dir = Path(r.__file__).parent / "models"
        info["models"] = sorted(p.name for p in model_dir.glob("*.onnx")) \
            if model_dir.is_dir() else []
    except Exception as e:                                     # pragma: no cover
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def get_ocr():
    """Resident RapidOCR instance (CPU, deterministic). Loaded once per process."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _cap_size(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / float(max(h, w))
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def run_ocr(image_bgr: np.ndarray) -> dict:
    """Run the OCR backend on an in-memory BGR image.

    The image is capped to ``PARAMS['max_side']`` first and the *capped* image is
    returned inside the result, so every box coordinate shares one coordinate space
    with the pixel statistics computed later.
    """
    img = _cap_size(image_bgr, int(PARAMS["max_side"]))
    t0 = time.perf_counter()
    res, _ = get_ocr()(img)
    ms = (time.perf_counter() - t0) * 1000
    items = []
    for it in (res or []):
        box, text, score = it[0], it[1], it[2]
        try:
            sc = float(score)
        except (TypeError, ValueError):
            sc = float("nan")
        items.append({"box": [[float(x), float(y)] for x, y in box],
                      "text": str(text), "score": sc})
    return {"items": items, "elapsed_ms": round(ms, 1), "image": img,
            "shape": (int(img.shape[0]), int(img.shape[1]))}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _box_geom(box) -> tuple:
    """(x0, y0, x1, y1, w, h) of an axis-aligned box from a 4-point quad."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return x0, y0, x1, y1, max(x1 - x0, 0.0), max(y1 - y0, 0.0)


def _cv(vals) -> float:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0 or abs(a.mean()) < 1e-9:
        return 0.0
    return float(a.std() / abs(a.mean()))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# family 1+2+3: confidence, layout, text legality
# ---------------------------------------------------------------------------
def _text_stats(items: list, shape: tuple) -> dict:
    if not items:
        return {k: 0.0 for k in FEATURE_NAMES}
    scores = np.asarray([it["score"] for it in items], dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    geoms = [_box_geom(it["box"]) for it in items]
    hs = [g[5] for g in geoms]
    ws = [g[4] for g in geoms]
    lefts = [g[0] for g in geoms]
    texts = [it["text"] for it in items]
    joined = "".join(texts)
    chars = [c for c in joined if not c.isspace()]
    n_chars = len(chars)
    tokens = [t for t in "".join(texts).replace(",", " ").replace("，", " ").split() if t]
    num_groups = NUM_GROUP_RE.findall(joined)
    n_num = len(num_groups)
    n_amount = len(AMOUNT_RE.findall(joined))
    n_mixed = len(MIXED_NUM_RE.findall(joined))
    n_date = len(DATE_LIKE_RE.findall(joined))
    h_img, w_img = shape
    img_area = float(h_img * w_img)
    box_area = sum(g[4] * g[5] for g in geoms)
    # line grouping by vertical centre for the gap statistic
    ys = sorted((g[1] + g[3]) / 2.0 for g in geoms)
    gaps = np.diff(ys) if len(ys) > 1 else np.asarray([])
    hist = Counter(chars)
    tot = sum(hist.values()) or 1
    entropy = -sum((c / tot) * math.log(c / tot) for c in hist.values())
    return {
        "ocr_has_text": 1.0,
        "n_boxes": float(len(items)),
        "n_chars": float(n_chars),
        "mean_score": float(scores.mean()) if scores.size else 0.0,
        "p10_score": float(np.percentile(scores, 10)) if scores.size else 0.0,
        "score_std": float(scores.std()) if scores.size else 0.0,
        "frac_score_lt_50": float((scores < PARAMS["low_score"]).mean()) if scores.size else 0.0,
        "frac_score_lt_70": float((scores < PARAMS["very_low_score"]).mean()) if scores.size else 0.0,
        "char_density_per_mpx": _safe_div(n_chars, img_area / 1e6),
        "box_area_frac": _safe_div(box_area, img_area),
        "box_h_mean": float(np.mean(hs)) if hs else 0.0,
        "box_h_cv": _cv(hs),
        "box_w_cv": _cv(ws),
        "line_gap_cv": _cv(gaps),
        "left_edge_std_frac": _safe_div(float(np.std(lefts)) if lefts else 0.0, float(w_img)),
        "mean_aspect": _safe_div(float(np.mean([_safe_div(w, h) for w, h in zip(ws, hs)])), 1.0)
        if hs else 0.0,
        "n_num_groups": float(n_num),
        "frac_amount_numbers": _safe_div(n_amount, n_num),
        "has_date_like": 1.0 if n_date else 0.0,
        "frac_mixed_num_groups": _safe_div(n_mixed, n_num),
        "digit_ratio": _safe_div(sum(c.isdigit() for c in chars), n_chars),
        "cjk_ratio": _safe_div(sum("\u4e00" <= c <= "\u9fff" for c in chars), n_chars),
        "frac_len1_boxes": _safe_div(sum(1 for t in texts if len(t.strip()) == 1), len(texts)),
        "char_entropy": float(entropy),
        "has_currency": 1.0 if any(s in joined for s in ("￥", "¥", "$")) else 0.0,
    }


# ---------------------------------------------------------------------------
# family 4: stamp / red-ink layer
# ---------------------------------------------------------------------------
def red_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo = (h <= PARAMS["red_h_low"]) | (h >= PARAMS["red_h_high"])
    m = (lo & (s >= PARAMS["red_min_s"]) & (v >= PARAMS["red_min_v"])).astype(np.uint8)
    if PARAMS["red_dilate_px"]:
        k = np.ones((int(PARAMS["red_dilate_px"]), int(PARAMS["red_dilate_px"])), np.uint8)
        m = cv2.dilate(m, k, iterations=1)
    return m


def _red_stats(image_bgr: np.ndarray, items: list) -> dict:
    m = red_mask(image_bgr)
    area = float(m.shape[0] * m.shape[1])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= PARAMS["red_min_area_px"]]
    overlap = 0.0
    if items and big:
        big_labels = {i for _, i in big}
        hit = 0
        for it in items:
            g = _box_geom(it["box"])
            cx = int(np.clip((g[0] + g[2]) / 2.0, 0, m.shape[1] - 1))
            cy = int(np.clip((g[1] + g[3]) / 2.0, 0, m.shape[0] - 1))
            if labels[cy, cx] in big_labels:
                hit += 1
        overlap = hit / len(items)
    return {
        "red_frac": _safe_div(float(m.sum()), area),
        "red_region_count": float(len(big)),
        "red_max_region_frac": _safe_div(float(max((a for a, _ in big), default=0)), area),
        "red_text_overlap_frac": overlap,
    }


# ---------------------------------------------------------------------------
# family 5: per-character stroke complexity (no reference text needed)
# ---------------------------------------------------------------------------
def _ink_mask(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (int(PARAMS["ink_otsu_blur"]),) * 2, 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float(bw.mean()) > 127.5:          # majority "ink" -> polarity is inverted
        bw = 255 - bw
    return bw


def _stroke_stats(image_bgr: np.ndarray, items: list) -> dict:
    if not items:
        return {"ink_ratio_mean": 0.0, "ink_ratio_cv": 0.0,
                "stroke_cc_per_char": 0.0, "stroke_laplacian_per_char": 0.0}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ratios, cc_per_char, lap_per_char = [], [], []
    for it in items:
        x0, y0, x1, y1, bw_, bh_ = _box_geom(it["box"])
        xi0, yi0 = int(max(0, x0)), int(max(0, y0))
        xi1, yi1 = int(min(w, x1)), int(min(h, y1))
        if xi1 - xi0 < 4 or yi1 - yi0 < 4:
            continue
        crop = gray[yi0:yi1, xi0:xi1]
        ink = _ink_mask(crop)
        ratios.append(float((ink > 0).mean()))
        n_chars_box = max(1, len(it["text"].strip()))
        ncc, _ = cv2.connectedComponents(ink, connectivity=8)
        cc_per_char.append((ncc - 1) / n_chars_box)
        lap = np.abs(cv2.Laplacian(crop, cv2.CV_32F, ksize=3))
        lap_per_char.append(float(lap.mean()) / n_chars_box)
    return {
        "ink_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "ink_ratio_cv": _cv(ratios),
        "stroke_cc_per_char": float(np.mean(cc_per_char)) if cc_per_char else 0.0,
        "stroke_laplacian_per_char": float(np.mean(lap_per_char)) if lap_per_char else 0.0,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def features_from_ocr(image_bgr: np.ndarray, ocr_result: dict) -> dict:
    """Pure function: image + OCR output -> the fixed feature block (no labels)."""
    items = ocr_result["items"]
    img = ocr_result.get("image", image_bgr)      # the exact image the boxes refer to
    shape = tuple(ocr_result.get("shape") or img.shape[:2])
    out = {}
    out.update(_text_stats(items, shape))
    out.update(_red_stats(img, items))
    out.update(_stroke_stats(img, items))
    missing = [k for k in FEATURE_NAMES if k not in out]
    if missing:                                                # pragma: no cover
        raise RuntimeError(f"feature block incomplete: {missing}")
    return {k: float(out[k]) for k in FEATURE_NAMES}


def extract_features(image_path: str | Path, image_bgr: np.ndarray | None = None) -> dict:
    """End-to-end: decode (if needed) -> OCR -> feature block."""
    img = image_bgr if image_bgr is not None else cv2.imread(str(image_path),
                                                              cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot decode image: {image_path}")
    res = run_ocr(img)
    feats = features_from_ocr(img, res)
    return {"features": feats,
            "ocr": {k: v for k, v in res.items() if k != "image"},
            "backend": backend_info(),
            "elapsed_ms": res["elapsed_ms"]}


def feature_vector(feats: dict) -> np.ndarray:
    return np.asarray([float(feats[n]) for n in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# label-isolated sample listing
# ---------------------------------------------------------------------------
USED_META_COLUMNS: tuple = ("seq", "content")


def iter_document_rows(meta_path: str | Path, content: str = "document",
                       require_images_dir: str | Path | None = None) -> list:
    """Rows of ``data/meta.csv`` with content == ``content``.

    Only ``USED_META_COLUMNS`` are copied out of each row, so the ``label`` column
    never enters this process' data structures. ``data/labels.csv`` is not opened by
    this module at all.
    """
    meta_path = Path(meta_path)
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # The label column exists on disk; only USED_META_COLUMNS are ever copied out,
        # so no label value can reach a feature. (data/labels.csv is not opened at all.)
        assert rdr.fieldnames and "label" in rdr.fieldnames, "unexpected meta.csv schema"
        for r in rdr:
            if r.get("content") != content:
                continue
            row = {k: r.get(k, "") for k in USED_META_COLUMNS}
            row["seq"] = int(row["seq"])
            if require_images_dir is not None:
                p = Path(require_images_dir) / f"{row['seq']:03d}.png"
                if not p.is_file():
                    continue
                row["image_path"] = str(p)
            rows.append(row)
    rows.sort(key=lambda r: r["seq"])
    return rows


def image_path_for(seq: int, images_dir: str | Path) -> Path:
    return Path(images_dir) / f"{seq:03d}.png"


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def run_self_test(seqs=(3, 80), images_dir: str | Path | None = None) -> dict:
    ws = WS_DEFAULT
    idir = Path(images_dir) if images_dir else (ws / "data" / "images")
    checks, rows = [], {}
    t0 = time.perf_counter()
    for s in seqs:
        p = idir / f"{s:03d}.png"
        r = extract_features(p)
        rows[s] = {"path": str(p), "n_boxes": r["features"]["n_boxes"],
                   "n_chars": r["features"]["n_chars"],
                   "mean_score": round(r["features"]["mean_score"], 4),
                   "features": {k: round(float(v), 6) for k, v in r["features"].items()},
                   "elapsed_ms": r["elapsed_ms"]}
    a, b = rows[seqs[0]]["features"], rows[seqs[1]]["features"]
    diff = [k for k in FEATURE_NAMES if abs(a[k] - b[k]) > 1e-9]
    checks.append({"id": "ocr_available", "passed": ocr_available(),
                   "detail": json.dumps(backend_info(), ensure_ascii=False)[:300]})
    checks.append({"id": "text_found", "passed": rows[seqs[0]]["n_boxes"] > 0,
                   "detail": f"seq{seqs[0]}: n_boxes={rows[seqs[0]]['n_boxes']} "
                             f"n_chars={rows[seqs[0]]['n_chars']}"})
    checks.append({"id": "features_finite",
                   "passed": bool(np.all(np.isfinite(feature_vector(a))))
                             and bool(np.all(np.isfinite(feature_vector(b)))),
                   "detail": f"feature block has {len(FEATURE_NAMES)} values, all finite"})
    checks.append({"id": "two_images_differ", "passed": len(diff) >= 5,
                   "detail": f"{len(diff)}/{len(FEATURE_NAMES)} features differ; "
                             f"examples: " + ", ".join(
                                 f"{k}={a[k]:.4g} vs {b[k]:.4g}" for k in diff[:5])})
    checks.append({"id": "label_columns_untouched",
                   "passed": USED_META_COLUMNS == ("seq", "content"),
                   "detail": f"meta columns read: {USED_META_COLUMNS}"})
    return {"module": "repo/src/l1c_ocr.py", "backend": backend_info(),
            "passed": all(c["passed"] for c in checks),
            "n_checks": len(checks), "n_passed": int(sum(c["passed"] for c in checks)),
            "checks": checks, "samples": rows,
            "params": dict(PARAMS), "feature_names": list(FEATURE_NAMES),
            "runtime_s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L1c Chinese-document OCR features (t6)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seqs", default="3,80")
    ap.add_argument("--images-dir", default="")
    args = ap.parse_args(argv)
    if not args.self_test:
        ap.print_help()
        return 2
    seqs = tuple(int(x) for x in args.seqs.split(",") if x.strip())
    res = run_self_test(seqs, args.images_dir or None)
    for c in res["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['detail']}")
    print(f"self-test: {res['n_passed']}/{res['n_checks']} passed ({res['runtime_s']} s)")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
