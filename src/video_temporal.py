"""Inter-frame temporal evidence for the video channel (improvement task t5).

Motivation
----------
The image path (src/detector.py) scores single frames. A video additionally exposes
*inter-frame* statistics that a still cannot show, and AI-generated video is expected
to break them: neighbouring frames are produced independently (or with weak temporal
coherence), so geometric consistency, grain/noise continuity and high-frequency energy
stability typically behave differently from optically coherent real footage.

This module computes three pre-registered families of statistics on a list of
consecutive BGR frames:

  1. optical-flow consistency  - Farneback forward/backward endpoint error
  2. noise-residual continuity - median-filter high-pass residual correlation + energy
  3. HF energy stability       - Laplacian-variance coefficient of variation / steps

Every parameter is a module-level constant with a fixed default (see PARAMS). Nothing
here is fitted, trained, or tuned on evaluation data, and the module has no dependency
on the detector, on CLIP, or on any model artifact.

Honest scope (read before quoting any number)
---------------------------------------------
These features are *not* a validated real-vs-AI video detector. On the t5 controlled
validation set they are reported as univariate AUCs only, under a channel-level
(not business-level) protocol: the brief contains no video, and both classes of the
t5 set are still images turned into genuinely encoded clips. See
reports/video_temporal_validation.json for the measured numbers, the construction of
the set, and the explicit gap statements.

Self-test
---------
    python repo/src/video_temporal.py --self-test

The self-test builds real encoded-free frame sequences from two different still images
and asserts (among other things) that the feature values actually differ between them.
It exits non-zero when any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Frozen parameter set (pre-registered; no value below was fitted on any data)
# ---------------------------------------------------------------------------
PARAMS: dict = {
    # working resolution for every statistic: long side is scaled down to this
    "work_width": 512,
    # Farneback dense optical flow (OpenCV defaults are used verbatim so the
    # numbers are reproducible on any OpenCV build with the same version)
    "fb_pyr_scale": 0.5,
    "fb_levels": 3,
    "fb_winsize": 15,
    "fb_iterations": 3,
    "fb_poly_n": 5,
    "fb_poly_sigma": 1.2,
    # forward/backward endpoint error normaliser, in pixels
    "fb_eps_px": 1.0,
    # noise-residual high-pass: median blur kernel (odd)
    "noise_kernel": 3,
    # HF energy: Laplacian kernel size
    "hf_ksize": 3,
    # deterministic clip synthesis used by the self-test and by the validation
    # builder (scripts/video_temporal_validation.py). Same motion field and same
    # noise realisation for EVERY clip, so two clips differ only in still content.
    "synth_frames": 24,
    "synth_canvas": 640,
    "synth_amplitude_px": 2.0,
    "synth_zoom": 0.02,
    "synth_noise_sigma": 3.0,
    "synth_seed": 12345,
    "synth_motion_cycles": 2.0,
}

# Fixed feature order. Never reorder: the report and any downstream consumer rely on it.
TEMPORAL_FEATURE_NAMES: tuple = (
    "flow_fb_error_ratio_mean",     # 1 px-normalised FB inconsistency, higher = more AI-like
    "flow_fb_error_ratio_max",      # worst consecutive pair
    "flow_magnitude_mean",          # mean flow magnitude (px); context, direction n/a
    "flow_magnitude_cv",            # std/mean of per-pair flow magnitude, higher = more AI-like
    "noise_residual_corr_mean",     # mean inter-frame residual correlation, higher = more AI-like
    "noise_residual_corr_min",      # weakest consecutive pair
    "noise_energy_cv",              # std/mean of per-frame residual energy, higher = more AI-like
    "noise_energy_step_ratio",      # mean |frame-to-frame energy step| / mean energy
    "hf_energy_cv",                 # std/mean of per-frame Laplacian variance, higher = more AI-like
    "hf_energy_step_ratio",         # mean |frame-to-frame HF step| / mean HF energy
)

# Pre-registered sign: +1 means "larger value is more AI-video-like", 0 means the
# feature is reported for context only and carries no signed claim.
AI_LIKE_DIRECTION: dict = {
    "flow_fb_error_ratio_mean": +1,
    "flow_fb_error_ratio_max": +1,
    "flow_magnitude_mean": 0,
    "flow_magnitude_cv": +1,
    "noise_residual_corr_mean": +1,
    "noise_residual_corr_min": +1,
    "noise_energy_cv": +1,
    "noise_energy_step_ratio": +1,
    "hf_energy_cv": +1,
    "hf_energy_step_ratio": +1,
}

SIGNED_FEATURE_NAMES: tuple = tuple(n for n in TEMPORAL_FEATURE_NAMES
                                    if AI_LIKE_DIRECTION[n] != 0)

MIN_FRAMES = 3


@dataclass
class TemporalFeatures:
    """Result of :func:`compute` for one frame sequence."""
    n_frames: int = 0
    n_pairs: int = 0
    work_size: tuple = (0, 0)
    flow_fb_error_ratio_mean: float = float("nan")
    flow_fb_error_ratio_max: float = float("nan")
    flow_magnitude_mean: float = float("nan")
    flow_magnitude_cv: float = float("nan")
    noise_residual_corr_mean: float = float("nan")
    noise_residual_corr_min: float = float("nan")
    noise_energy_cv: float = float("nan")
    noise_energy_step_ratio: float = float("nan")
    hf_energy_cv: float = float("nan")
    hf_energy_step_ratio: float = float("nan")
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def vector(self) -> np.ndarray:
        return np.asarray([float(getattr(self, n)) for n in TEMPORAL_FEATURE_NAMES],
                          dtype=np.float64)

    def signed_ai_like(self) -> dict:
        """Features oriented so that a larger value is more AI-video-like."""
        return {n: float(getattr(self, n)) for n in SIGNED_FEATURE_NAMES}


# ---------------------------------------------------------------------------
# frame preparation
# ---------------------------------------------------------------------------
def prepare_gray(frame_bgr: np.ndarray, work_width: int = PARAMS["work_width"]) -> np.ndarray:
    """BGR uint8 -> grayscale float32, long side scaled to ``work_width``."""
    if frame_bgr.ndim == 3:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = frame_bgr
    h, w = g.shape[:2]
    if max(h, w) > work_width > 0:
        s = work_width / float(max(h, w))
        g = cv2.resize(g, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def _farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        a, b, None,
        PARAMS["fb_pyr_scale"], PARAMS["fb_levels"], PARAMS["fb_winsize"],
        PARAMS["fb_iterations"], PARAMS["fb_poly_n"], PARAMS["fb_poly_sigma"], 0)


def _scale_clip(frames_bgr: list) -> list:
    return [prepare_gray(f) for f in frames_bgr]


# ---------------------------------------------------------------------------
# family 1: optical-flow consistency
# ---------------------------------------------------------------------------
def flow_stats(grays: list) -> dict:
    """Forward/backward flow consistency (r = A->B, r' = B->A, warped by r).

    For a physically coherent pair the round trip r + warp(r', r) is ~0 up to the
    normaliser; independently generated neighbouring frames break it.
    """
    errs, mags = [], []
    for a, b in zip(grays[:-1], grays[1:]):
        fwd = _farneback(a, b)
        bwd = _farneback(b, a)
        h, w = a.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        # sample the backward flow at the position the forward flow moved each pixel to
        mx = np.clip(gx + fwd[..., 0], 0, w - 1)
        my = np.clip(gy + fwd[..., 1], 0, h - 1)
        bwd_w = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.linalg.norm(fwd + bwd_w, axis=-1)
        denom = (np.linalg.norm(fwd, axis=-1) + np.linalg.norm(bwd_w, axis=-1)
                 + PARAMS["fb_eps_px"])
        errs.append(float(np.mean(err / denom)))
        mags.append(float(np.mean(np.linalg.norm(fwd, axis=-1))))
    m = np.asarray(mags, dtype=np.float64)
    return {
        "flow_fb_error_ratio_mean": float(np.mean(errs)) if errs else float("nan"),
        "flow_fb_error_ratio_max": float(np.max(errs)) if errs else float("nan"),
        "flow_magnitude_mean": float(m.mean()) if m.size else float("nan"),
        "flow_magnitude_cv": float(m.std() / m.mean()) if m.size and m.mean() > 1e-9
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# family 2: noise-residual continuity
# ---------------------------------------------------------------------------
def _residuals(grays: list) -> list:
    k = int(PARAMS["noise_kernel"])
    return [g - cv2.medianBlur(g, k) for g in grays]


def noise_stats(grays: list) -> dict:
    """Temporal continuity of the sensor-grain / fine-texture residual field.

    ``corr`` is the Pearson correlation of consecutive residual maps: independent
    per-frame grain gives ~0, a residual field that is redrawn coherently (or almost
    frozen) gives a large positive value. ``energy_cv``/``step_ratio`` describe how
    stable the residual energy level is across the sequence.
    """
    res = _residuals(grays)
    corrs = []
    for r0, r1 in zip(res[:-1], res[1:]):
        a = r0.ravel().astype(np.float64)
        b = r1.ravel().astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        da, db = np.linalg.norm(a), np.linalg.norm(b)
        corrs.append(float(a.dot(b) / (da * db)) if da > 1e-9 and db > 1e-9
                     else float("nan"))
    energies = np.asarray([float(r.std()) for r in res], dtype=np.float64)
    mean_e = float(energies.mean()) if energies.size else float("nan")
    steps = np.abs(np.diff(energies)) if energies.size > 1 else np.asarray([])
    return {
        "noise_residual_corr_mean": float(np.nanmean(corrs)) if corrs else float("nan"),
        "noise_residual_corr_min": float(np.nanmin(corrs)) if corrs else float("nan"),
        "noise_energy_cv": float(energies.std() / mean_e) if mean_e > 1e-9 else float("nan"),
        "noise_energy_step_ratio": float(steps.mean() / mean_e)
        if steps.size and mean_e > 1e-9 else float("nan"),
    }


# ---------------------------------------------------------------------------
# family 3: inter-frame high-frequency energy stability
# ---------------------------------------------------------------------------
def hf_stats(grays: list) -> dict:
    ks = int(PARAMS["hf_ksize"])
    energies = np.asarray([float(cv2.Laplacian(g, cv2.CV_32F, ksize=ks).var())
                           for g in grays], dtype=np.float64)
    mean_e = float(energies.mean()) if energies.size else float("nan")
    steps = np.abs(np.diff(energies)) if energies.size > 1 else np.asarray([])
    return {
        "hf_energy_cv": float(energies.std() / mean_e) if mean_e > 1e-9 else float("nan"),
        "hf_energy_step_ratio": float(steps.mean() / mean_e)
        if steps.size and mean_e > 1e-9 else float("nan"),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute(frames_bgr: list, work_width: int | None = None) -> TemporalFeatures:
    """Compute the frozen temporal feature block for one consecutive frame list.

    ``frames_bgr`` is a list of uint8 BGR images (as ``cv2.VideoCapture`` yields).
    Deterministic: identical frames in, identical values out.
    """
    if len(frames_bgr) < MIN_FRAMES:
        raise ValueError(f"need >= {MIN_FRAMES} frames, got {len(frames_bgr)}")
    ww = int(work_width or PARAMS["work_width"])
    grays = [prepare_gray(f, ww) for f in frames_bgr]
    out = TemporalFeatures(n_frames=len(grays), n_pairs=len(grays) - 1,
                           work_size=(int(grays[0].shape[1]), int(grays[0].shape[0])))
    for part in (flow_stats(grays), noise_stats(grays), hf_stats(grays)):
        for k, v in part.items():
            setattr(out, k, float(v))
    bad = [n for n in TEMPORAL_FEATURE_NAMES if not np.isfinite(getattr(out, n))]
    if bad:
        out.notes.append(f"non-finite features: {bad}")
    return out


def feature_vector(frames_bgr: list, work_width: int | None = None) -> np.ndarray:
    return compute(frames_bgr, work_width).vector()


# ---------------------------------------------------------------------------
# deterministic clip synthesis (shared by the self-test and the validation set)
# ---------------------------------------------------------------------------
def resize_fill(still_bgr: np.ndarray, canvas: int = PARAMS["synth_canvas"]) -> np.ndarray:
    """Cover-resize then centre-crop, exactly as the archived video validation did.

    No letterboxing: padding with a flat colour would inject a flat-region artefact
    whose area depends on the still's aspect ratio and would poison the forensic
    features (same rationale as scripts/video_channel_validation.py).
    """
    h, w = still_bgr.shape[:2]
    s = max(canvas / float(w), canvas / float(h))
    big = cv2.resize(still_bgr,
                     (max(canvas, int(round(w * s))), max(canvas, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0 = (big.shape[0] - canvas) // 2
    x0 = (big.shape[1] - canvas) // 2
    return np.ascontiguousarray(big[y0:y0 + canvas, x0:x0 + canvas])


def synthesize_sequence(still_bgr: np.ndarray, n_frames: int = PARAMS["synth_frames"],
                        canvas: int = PARAMS["synth_canvas"],
                        amplitude_px: float = PARAMS["synth_amplitude_px"],
                        zoom: float = PARAMS["synth_zoom"],
                        noise_sigma: float = PARAMS["synth_noise_sigma"],
                        seed: int = PARAMS["synth_seed"]) -> list:
    """Turn one still into a genuinely multi-frame sequence.

    Every frame is geometrically distinct (deterministic sub-pixel sinusoidal pan +
    slow zoom) and carries an independent deterministic noise realisation. The motion
    field and the noise realisation depend ONLY on the frame index, so two sequences
    built from two different stills are identical in everything except content.

    This is what makes the encoded clips real *encoded* video (the codec performs
    inter-frame prediction on different frames) rather than a repeated single frame,
    and it is also an explicit limitation: the temporal signal is not camera physics.
    """
    base = resize_fill(still_bgr, canvas).astype(np.float32)
    h, w = base.shape[:2]
    out = []
    for i in range(int(n_frames)):
        t = i / float(max(1, n_frames - 1))
        ph = 2.0 * np.pi * PARAMS["synth_motion_cycles"] * t
        dx = amplitude_px * np.sin(ph)
        dy = amplitude_px * 0.6 * np.cos(ph)
        sc = 1.0 + zoom * t
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, sc)
        m[0, 2] += dx
        m[1, 2] += dy
        fr = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        rng = np.random.default_rng(int(seed) + i)
        fr = fr + rng.normal(0.0, noise_sigma, size=fr.shape).astype(np.float32)
        out.append(np.clip(fr, 0, 255).astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# unit self-test
# ---------------------------------------------------------------------------
def _synth_still(kind: str, h: int = 200, w: int = 260, seed: int = 7) -> np.ndarray:
    """Portable fallback stills: a smooth AI-like gradient vs a textured photo-like one."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "smooth":
        img = 120 + 60 * np.sin(xx / 40.0) + 30 * np.cos(yy / 27.0)
        img = cv2.GaussianBlur(img, (0, 0), 3.0)
        return np.dstack([img] * 3).astype(np.uint8)
    img = 110 + 40 * np.sin(xx / 9.0) + 35 * np.cos(yy / 11.0)
    img = img + rng.normal(0, 9.0, size=(h, w)).astype(np.float32)
    img = np.clip(img, 0, 255)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _load_stills(data_dir: Path, seqs=(1, 57)) -> tuple:
    """Prefer two real still images from the workspace; fall back to synthetic ones."""
    got = []
    for s in seqs:
        p = data_dir / "images" / f"{s:03d}.png"
        if p.is_file():
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                got.append((str(p), im))
    if len(got) != len(seqs):
        return ([("synth:smooth", _synth_still("smooth")),
                 ("synth:textured", _synth_still("textured"))], True)
    return got, False


def run_self_test(data_dir: str | Path | None = None) -> dict:
    """Assert the module behaves as documented; return a structured result."""
    t0 = time.perf_counter()
    ws = Path(__file__).resolve().parents[2]
    dd = Path(data_dir) if data_dir else (ws / "data")
    (a_path, a_img), (b_path, b_img) = _load_stills(dd)[0]
    synthetic_fallback = bool(_load_stills(dd)[1])

    n = int(PARAMS["synth_frames"])
    seq_a = synthesize_sequence(a_img, n_frames=n)
    seq_b = synthesize_sequence(b_img, n_frames=n)
    fa = compute(seq_a)
    fb = compute(seq_b)
    fa2 = compute(seq_a)                      # determinism re-run
    frozen = compute([np.ascontiguousarray(seq_a[0])] * n)

    deltas = {k: float(getattr(fa, k) - getattr(fb, k)) for k in TEMPORAL_FEATURE_NAMES}
    rel = {k: (abs(deltas[k]) / max(abs(getattr(fa, k)), abs(getattr(fb, k)), 1e-12))
           for k in TEMPORAL_FEATURE_NAMES}
    differing_abs = [k for k in TEMPORAL_FEATURE_NAMES if abs(deltas[k]) > 1e-6]
    differing_rel = [k for k in TEMPORAL_FEATURE_NAMES if rel[k] > 0.01]

    checks = []

    def add(cid, desc, passed, detail):
        checks.append({"id": cid, "description": desc, "passed": bool(passed),
                       "detail": detail})

    add("determinism", "re-computing the same frames yields bit-identical values",
        all(float(getattr(fa, k)) == float(getattr(fa2, k)) for k in TEMPORAL_FEATURE_NAMES),
        "all 10 features identical on re-run")
    add("two_stills_differ", "two different still images yield different (not just "
                             "re-scaled) feature values",
        len(differing_abs) >= 3 and len(differing_rel) >= 3,
        {"n_abs_gt_1e-6": len(differing_abs), "n_rel_gt_1pct": len(differing_rel),
         "abs_deltas": {k: round(deltas[k], 6) for k in TEMPORAL_FEATURE_NAMES},
         "rel_deltas": {k: round(rel[k], 6) for k in TEMPORAL_FEATURE_NAMES}})
    add("feature_count", "feature vector length/order matches TEMPORAL_FEATURE_NAMES",
        fa.vector().shape == (len(TEMPORAL_FEATURE_NAMES),) and len(SIGNED_FEATURE_NAMES) >= 1,
        {"n_features": len(TEMPORAL_FEATURE_NAMES), "n_signed": len(SIGNED_FEATURE_NAMES)})
    add("all_finite", "every feature is finite (no NaN/inf) on a real sequence",
        all(np.isfinite(getattr(fa, k)) for k in TEMPORAL_FEATURE_NAMES),
        "10/10 finite")
    add("frozen_frames_are_static", "an all-identical frame list gives ~zero flow",
        float(frozen.flow_magnitude_mean) < 0.05,
        f"flow_magnitude_mean={frozen.flow_magnitude_mean:.6f} px on repeated frames")
    add("frozen_residual_corr_high", "repeated frames give residual correlation ~1 "
                                     "(documents what the continuity feature measures)",
        float(frozen.noise_residual_corr_mean) > 0.99,
        f"noise_residual_corr_mean={frozen.noise_residual_corr_mean:.6f}")
    add("motion_is_detected", "the synthesised pan/zoom produces non-zero flow",
        float(fa.flow_magnitude_mean) > 0.2,
        f"flow_magnitude_mean={fa.flow_magnitude_mean:.6f} px")
    add("guard_min_frames", "fewer than MIN_FRAMES frames raise instead of returning junk",
        _raises(lambda: compute([seq_a[0], seq_a[1]])),
        f"MIN_FRAMES={MIN_FRAMES}")

    result = {
        "module": "repo/src/video_temporal.py",
        "passed": all(c["passed"] for c in checks),
        "n_checks": len(checks),
        "n_passed": int(sum(c["passed"] for c in checks)),
        "checks": checks,
        "params": dict(PARAMS),
        "feature_names": list(TEMPORAL_FEATURE_NAMES),
        "ai_like_direction": dict(AI_LIKE_DIRECTION),
        "stills": {"a": a_path, "b": b_path, "synthetic_fallback": synthetic_fallback},
        "values_on_still_a": {k: round(float(getattr(fa, k)), 8)
                              for k in TEMPORAL_FEATURE_NAMES},
        "values_on_still_b": {k: round(float(getattr(fb, k)), 8)
                              for k in TEMPORAL_FEATURE_NAMES},
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="inter-frame temporal features (t5)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", default="", help="workspace data dir (default: auto)")
    ap.add_argument("--json", default="", help="also write the self-test result here")
    args = ap.parse_args(argv)
    if not args.self_test:
        ap.print_help()
        return 2
    res = run_self_test(args.data_dir or None)
    for c in res["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['description']}")
    print(f"\nself-test: {res['n_passed']}/{res['n_checks']} passed "
          f"({res['runtime_ms']} ms)")
    print(f"still A ({res['stills']['a']}): flow_fb_err="
          f"{res['values_on_still_a']['flow_fb_error_ratio_mean']:.4f} "
          f"noise_corr={res['values_on_still_a']['noise_residual_corr_mean']:.4f} "
          f"hf_cv={res['values_on_still_a']['hf_energy_cv']:.4f}")
    print(f"still B ({res['stills']['b']}): flow_fb_err="
          f"{res['values_on_still_b']['flow_fb_error_ratio_mean']:.4f} "
          f"noise_corr={res['values_on_still_b']['noise_residual_corr_mean']:.4f} "
          f"hf_cv={res['values_on_still_b']['hf_energy_cv']:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

