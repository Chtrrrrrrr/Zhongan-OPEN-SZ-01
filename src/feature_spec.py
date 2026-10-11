"""Feature specification for the decision heads - one source of truth, offline and online.

The `combined` head consumes a concatenation of feature blocks:

    [ L1(47) | L1b(20) | <optional patch-token stats> | CLIP global(1024) | tile aggregate ]

Every block is produced by a different code path: L1/L1b come from `l1_forensics` /
`l1b_domain`, the patch statistics and the tile aggregate both need the frozen CLIP
backbone. Historically the offline extractor and `src/detector.py` assembled that
concatenation independently, which is exactly the kind of drift that silently feeds a
head the wrong vector layout. This module makes the layout a single frozen artifact
(`models/feature_spec.json`) that both paths read:

  * :func:`FeatureSpec.forensic_vector` assembles the forensic member's input,
  * :func:`FeatureSpec.combined_vector` assembles the combined member's input,
  * :meth:`FeatureSpec.tile_features` extracts the tile block with the spec's scales
    and aggregation.

`aggregation` names are stable strings; `legacy` means the frozen
`models/clip_tiles*.npy` layout (3x3 tiles, mean/max/std).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AGGREGATIONS = ("legacy", "base", "meanmax", "rich", "spread", "lse")
DEFAULT_SPEC_NAME = "feature_spec.json"


def _aggregate(feats: np.ndarray, name: str) -> np.ndarray:
    f = np.asarray(feats, dtype=np.float32)
    if name in ("legacy", "base"):
        return np.concatenate([f.mean(0), f.max(0), f.std(0)]).astype(np.float32)
    if name == "meanmax":
        # cost-accuracy sweep winner: order-statistic-free and shorter than mean/max/std,
        # which keeps the head's input dimension smaller and measured better on every
        # seed tested (scripts/tile_clip_sweep.py)
        return np.concatenate([f.mean(0), f.max(0)]).astype(np.float32)
    if name == "rich":
        return np.concatenate([f.mean(0), f.max(0), f.std(0), f.min(0),
                               np.median(f, axis=0)]).astype(np.float32)
    if name == "spread":
        return np.concatenate([f.mean(0), f.max(0), f.std(0),
                               f.max(0) - f.min(0), f.mean(0) - f.min(0)]).astype(np.float32)
    if name == "lse":
        return np.concatenate([f.mean(0), f.max(0), f.std(0),
                               np.log(np.exp(f * 8.0).mean(0)) / 8.0]).astype(np.float32)
    raise ValueError(f"unknown aggregation {name!r}; expected one of {AGGREGATIONS}")


@dataclass
class FeatureSpec:
    """Which evidence blocks the heads expect, in which order."""

    scales: list[dict] = field(default_factory=lambda: [
        {"grid": 3, "overlap": 0.25, "max_side": 1600}])
    aggregation: str = "legacy"
    patch_tokens: bool = False
    patch_max_side: int = 1600
    name: str = "default"

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"name": self.name, "scales": self.scales,
                "aggregation": self.aggregation, "patch_tokens": self.patch_tokens,
                "patch_max_side": self.patch_max_side}

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSpec":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(scales=d.get("scales") or cls().scales,
                   aggregation=d.get("aggregation", "legacy"),
                   patch_tokens=bool(d.get("patch_tokens", False)),
                   patch_max_side=int(d.get("patch_max_side", 1600)),
                   name=d.get("name", p.stem))

    # ---- block assembly ----
    def forensic_vector(self, x1: np.ndarray, xb: np.ndarray,
                        patch: np.ndarray | None = None) -> np.ndarray:
        parts = [np.asarray(x1, dtype=np.float64).ravel(),
                 np.asarray(xb, dtype=np.float64).ravel()]
        if self.patch_tokens:
            if patch is None:
                raise ValueError("feature spec expects patch-token statistics but none were "
                                 "provided; extract them with "
                                 "scripts/extract_clip_patch_features.py")
            parts.append(np.asarray(patch, dtype=np.float64).ravel())
        return np.concatenate(parts)

    def combined_vector(self, x1, xb, clip_global, tile_agg, patch=None) -> np.ndarray:
        return np.concatenate([
            self.forensic_vector(x1, xb, patch),
            np.asarray(clip_global, dtype=np.float64).ravel(),
            np.asarray(tile_agg, dtype=np.float64).ravel(),
        ])

    def forensic_dim(self, n_l1: int = 47, n_l1b: int = 20, n_patch: int = 16) -> int:
        return n_l1 + n_l1b + (n_patch if self.patch_tokens else 0)

    def spec_block_dim(self, n_clip: int = 1024) -> int:
        """Dimension contributed by this spec's tile block (all scales concatenated)."""
        per_scale = {"legacy": 3, "base": 3, "meanmax": 2, "rich": 5, "spread": 5,
                     "lse": 4}[self.aggregation]
        return per_scale * n_clip * len(self.scales)

    # ---- tile block extraction (online path) ----
    def tile_features(self, extractor, bgr_full: np.ndarray, detail: dict | None = None):
        """Aggregated tile block for one image using this spec's scales.

        ``detail`` lets a caller reuse a :meth:`tile_detail` result it already computed
        (the attention explanation needs the same per-tile features, and running the
        backbone twice per item was the single largest waste in the pipeline).
        """
        if detail is None:
            detail = self.tile_detail(extractor, bgr_full)
        return self.aggregate_detail(detail)

    def aggregate_detail(self, detail: dict) -> np.ndarray:
        """Aggregate a :meth:`tile_detail` result with this spec's aggregation."""
        return np.concatenate([_aggregate(f, self.aggregation) for f in detail["feats"]])

    def tile_detail(self, extractor, bgr_full: np.ndarray) -> dict:
        """Per-scale tile features plus geometry, for the attribution explanation.

        Returns ``{"feats": [ [T_i, D], ... ], "boxes": [[(x0,y0,x1,y1), ...], ...],
        "sizes": [(w, h), ...], "scales": [...]}`` - the same grid/resize the score
        path uses, so an occlusion attribution is expressed in the layout the head
        actually consumes.
        """
        import cv2
        from l2_clip_tiles import tile_boxes

        feats, boxes_all, sizes = [], [], []
        for sc in self.scales:
            grid = int(sc.get("grid", 3))
            overlap = float(sc.get("overlap", 0.25))
            max_side = sc.get("max_side")
            img = bgr_full
            h, w = img.shape[:2]
            if max_side is not None and max(h, w) > int(max_side):
                s = int(max_side) / max(h, w)
                img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))),
                                 interpolation=cv2.INTER_AREA)
            hh, ww = img.shape[:2]
            boxes, tiles = [], []
            for (x0, y0, x1, y1) in tile_boxes(hh, ww, grid, overlap):
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                boxes.append((int(x0), int(y0), int(x1), int(y1)))
                tiles.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
            if not tiles:
                boxes, tiles = [(0, 0, int(ww), int(hh))], [img]
            feats.append(np.asarray(extractor.batch(tiles, batch_size=len(tiles)),
                                    dtype=np.float32))
            boxes_all.append(boxes)
            sizes.append((int(ww), int(hh)))
        return {"feats": feats, "boxes": boxes_all, "sizes": sizes,
                "grids": [int(sc.get("grid", 3)) for sc in self.scales],
                "aggregation": self.aggregation}
