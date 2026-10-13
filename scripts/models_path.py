#!/usr/bin/env python3
"""Single source of truth for locating the FROZEN deployment artifacts.

Why this module exists
----------------------
The deployment artifacts exist in two places and they are NOT the same files:

  repo/models/            <- the frozen deployment artifacts. This is what
                             reports/metrics.json, models/deployment_metrics.json
                             and reports/smoke_test.json were produced with.
  <workspace>/models/     <- a stale working copy (different thresholds and
                             different joblib bytes).

`repo/models/fusion.json` and `<workspace>/models/fusion.json` disagree, so a
script that happens to resolve `models/` relative to the CURRENT WORKING
DIRECTORY will silently load the stale copy and report numbers that no longer
match the documentation. That is a reproducibility defect, not a cosmetic one.

Resolution order (first hit wins)
---------------------------------
  1. an explicit --models-dir argument
  2. the MODEL_DIR environment variable
  3. <repo>/models            (the frozen artifacts shipped with the repository)
  4. <workspace>/models       (legacy fallback, reported loudly)

Everything that needs a deployed weight or threshold should go through
`resolve_models_dir()`, and should surface `describe()` in its output so a log
always says WHICH copy was loaded.

Two-level resolution
--------------------
The frozen repository copy is a SUBSET (it ships `fusion.json`, the deployed
heads, `feature_spec.json` and the 119-sample feature blocks, but not the large
working artefacts such as `clip_probe_norm.pt` / `clip_domain_lr.joblib` /
`hf_cache/`). `resolve_models_dir()` therefore picks the directory, and
`resolve_artifact()` picks each file inside it, falling back to
`<workspace>/models` only when the frozen copy does not ship that file. Both
labels end up in the log, so "which fusion.json / which probe" is never implicit.

The frozen threshold pair is published as FROZEN_THR_LOW / FROZEN_THR_HIGH and
`assert_expected_fusion()` checks the loaded `fusion.json` against it.
"""
from __future__ import annotations

import os
from pathlib import Path

# repo/ = parent of this file's directory (scripts/ -> repo/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_MODELS = REPO_ROOT / "models"

# The FROZEN deployment threshold pair published in docs/results.md / documented
# in docs/reproduce.md. Any other pair means a stale copy was loaded.
FROZEN_THR_LOW = 0.0266
FROZEN_THR_HIGH = 0.826623


def resolve_models_dir(explicit: str | os.PathLike | None = None,
                       workspace: Path | str | None = None) -> tuple[Path, str]:
    """Return (models_dir, source_label). Never raises; may return a missing path.

    `source_label` is one of: explicit, env, repo, workspace-fallback.
    """
    if explicit:
        return Path(explicit), "explicit"
    env = os.environ.get("MODEL_DIR")
    if env:
        return Path(env), "env"
    if DEFAULT_REPO_MODELS.is_dir():
        return DEFAULT_REPO_MODELS, "repo"
    if workspace is not None:
        ws_models = Path(workspace) / "models"
        if ws_models.is_dir():
            return ws_models, "workspace-fallback"
    return DEFAULT_REPO_MODELS, "repo-missing"


def resolve_artifact(models_dir: Path | str, name: str | Path,
                     workspace: Path | str | None = None) -> tuple[Path, str]:
    """Return (path, source_label) for ONE artifact inside the models directory.

    Why this exists on top of `resolve_models_dir()`
    -----------------------------------------------
    The frozen repository copy ships a *subset* of the artifact set: it carries
    `fusion.json` (the authoritative thresholds), both deployed heads,
    `feature_spec.json`, `deployment_metrics.json` and the 119-sample feature
    blocks, but NOT the large/optional working artefacts (`clip_probe_norm.pt`,
    `clip_domain_lr.joblib`, `l1_lr.joblib`, `hf_cache/`, `npr.pt`), which only
    exist in the workspace working copy.

    So "which models dir" is not sufficient to decide "which probe": choosing the
    frozen dir for `fusion.json` and then blindly joining the probe name onto it
    would produce a path that does not exist. This helper resolves per artifact:
    the frozen copy wins when it ships the file, otherwise the workspace copy is
    used and the caller is expected to log the fallback.

    `source_label` is one of: frozen, workspace-fallback, missing.
    """
    p = Path(models_dir) / name
    if p.exists():
        return p, "frozen"
    if workspace is not None:
        alt = Path(workspace) / "models" / name
        if alt.exists():
            return alt, "workspace-fallback"
    return p, "missing"


def describe(models_dir: Path, source: str) -> str:
    """One-line, loggable statement of which artifact copy is in use."""
    return (f"[models] {models_dir}  (source={source})"
            + ("" if models_dir.is_dir() else "  <-- DIRECTORY MISSING"))


def assert_expected_fusion(models_dir: Path) -> str:
    """Read thr_low/thr_high and warn if they are not the frozen deployment values.

    The frozen deployment thresholds are the pair published in docs/results.md.
    A mismatch means a stale copy was picked up; we report it instead of
    silently producing documentation-inconsistent metrics.
    """
    import json
    p = models_dir / "fusion.json"
    if not p.is_file():
        return f"[models] WARNING: {p} not found"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        lo, hi = float(d["thr_low"]), float(d["thr_high"])
    except Exception as e:  # noqa: BLE001
        return f"[models] WARNING: could not read {p}: {e}"
    if abs(lo - FROZEN_THR_LOW) < 5e-4 and abs(hi - FROZEN_THR_HIGH) < 5e-4:
        return f"[models] fusion thresholds OK (frozen): low={lo:.4f} high={hi:.6f}"
    return (f"[models] WARNING: fusion thresholds low={lo:.4f} high={hi:.6f} do NOT match "
            f"the frozen deployment pair ({FROZEN_THR_LOW} / {FROZEN_THR_HIGH}) documented in "
            f"docs/results.md - a stale models/ copy is probably being used")
