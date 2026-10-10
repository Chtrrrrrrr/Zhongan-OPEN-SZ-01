#!/usr/bin/env python3
"""Self-test for the L0/L1 pipeline (manual ch.7-8).

Checks:
  1. L0 parses a real sample and normalizes the long side to <= 512
  2. L1 returns exactly the 47 registered features, all finite
  3. Feature names are stable and unique (contract with features.csv)
  4. Container features are read from the raw file (unaffected by resize)
No pytest dependency: plain asserts, runnable as `python tests/test_l1.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l0_input import parse_image, MODEL_SIDE          # noqa: E402
from l1_forensics import (                             # noqa: E402
    FEATURE_NAMES, extract_l1_features, feature_vector,
)


def _sample_path() -> Path:
    """First available sample from the competition set."""
    for cand in (ROOT.parent / "data" / "images", ROOT / "data" / "images"):
        if cand.is_dir():
            pngs = sorted(cand.glob("*.png"))
            if pngs:
                return pngs[0]
    raise SystemExit("no sample image found (expected data/images/*.png)")


def main():
    path = _sample_path()
    print(f"[test] sample = {path}")

    # ---- L0 ----
    parsed = parse_image(str(path))
    assert parsed.ok, f"L0 parse failed: {parsed.error}"
    assert parsed.sha256 and len(parsed.sha256) == 64, "sha256 missing/short"
    assert parsed.phash, "perceptual hash missing"
    assert parsed.orig_width > 0 and parsed.orig_height > 0, "bad original size"
    long_side = max(parsed.bgr.shape[0], parsed.bgr.shape[1])
    assert long_side <= MODEL_SIDE, f"normalized long side {long_side} > {MODEL_SIDE}"
    assert len(parsed.png_chunks) > 0, "PNG chunk list empty"
    print(f"[test] L0 ok: {parsed.orig_width}x{parsed.orig_height} -> "
          f"{parsed.bgr.shape[1]}x{parsed.bgr.shape[0]}, "
          f"{len(parsed.png_chunks)} chunks, scale={parsed.resize_scale:.3f}")

    # ---- L1 ----
    feat = extract_l1_features(parsed)
    assert len(feat) == 47, f"expected 47 features, got {len(feat)}"
    assert set(feat) == set(FEATURE_NAMES), "feature name mismatch vs registry"
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature names"
    vals = np.array([feat[k] for k in FEATURE_NAMES], dtype=np.float64)
    assert np.all(np.isfinite(vals)), "non-finite feature value"
    vec = feature_vector(feat)
    assert vec.shape == (47,), f"feature_vector shape {vec.shape} != (47,)"
    print(f"[test] L1 ok: 47 finite features; "
          f"nidat={feat['nidat']:.0f}, hasSrgb={feat['hasSrgb']:.0f}, "
          f"blockRatio={feat['blockRatio']:.3f}")

    # ---- container features come from raw bytes, not the resized array ----
    assert feat["ow"] == parsed.orig_width, "ow must be the ORIGINAL width"
    assert feat["oh"] == parsed.orig_height, "oh must be the ORIGINAL height"
    assert feat["bytes"] == parsed.raw_bytes, "bytes must be the raw file size"
    print(f"[test] container features bound to raw file ok "
          f"(bytes={feat['bytes']:.0f}, ow={feat['ow']:.0f}, oh={feat['oh']:.0f})")

    # ---- degenerate input must not raise ----
    from l0_input import ParsedInput
    empty = extract_l1_features(ParsedInput(ok=False))
    assert set(empty) == set(FEATURE_NAMES), "failed parse must still return all names"
    print("[test] degraded path ok (ok=False -> 47 placeholder features)")

    print("\nALL L1 TESTS PASSED")


if __name__ == "__main__":
    main()
