#!/usr/bin/env python3
"""Tile-CLIP evidence search: accuracy at the 2-false-positive budget vs latency.

Why this script exists
----------------------
The frozen version scores every image with a 3x3 overlapping native-resolution tile
grid, pre-scaled to a 1600 px long side, aggregated with mean/max/std. That choice
was never swept. Two knobs plausibly move accuracy on this data:

  * grid / overlap / pre-scale   - how much native detail each tile keeps,
  * aggregation                  - mean/max/std vs order statistics / spreads.

This script (1) extracts per-tile CLIP features once per (grid, overlap, max_side)
cell into a cache, (2) scores every aggregation of that cell under the SAME
stratified-5-fold protocol the headline evaluation uses, reporting AUC and the
accuracy at a pre-registered false-positive budget, and (3) times the warm
end-to-end pipeline for the cells that look worth deploying.

Protocol discipline (unchanged from scripts/evaluate_system.py):
  * the forensic / combined heads are fit on the training fold only,
  * the decision threshold is the order statistic of TRAINING-fold real scores
    targeting `--fp-budget` false positives (2 = floor(0.03*70), the stated 7:3
    evaluation batch), so nothing is selected on the held-out fold.

Usage
  python scripts/tile_clip_sweep.py --configs C0,C6,C8          # subset
  python scripts/tile_clip_sweep.py --list                      # print the grid
  python scripts/tile_clip_sweep.py --aggregations base,rich
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from l0_input import parse_image                                  # noqa: E402
from l2_clip import CLIPFeatureExtractor                          # noqa: E402

CLIP_REPO = "openai/clip-vit-large-patch14"

# ---------------------------------------------------------------- config table
# A cell is one or more tile scales: name -> ((grid, overlap, max_side), ...).
# One scale = the deployed shape; several scales are concatenated after aggregation,
# which adds document-level AND character-level native detail to the same frozen
# backbone (no training, no extra parameters in the trunk).
CONFIGS: dict[str, tuple[tuple[int, float, int | None], ...]] = {
    "C0": ((3, 0.25, 1600),),                        # deployed baseline
    "C1": ((3, 0.25, None),),                        # deployed grid, no pre-scale
    "C3": ((3, 0.25, 2048),),
    "C5": ((4, 0.25, None),),
    "C6": ((5, 0.30, None),),
    "C7": ((5, 0.30, 1600),),
    "C8": ((6, 0.30, None),),
    "C9": ((4, 0.50, None),),
    # ---- multi-scale cells (added by the iteration) -------------------------
    "M1": ((3, 0.25, 1600), (6, 0.30, 1600)),        # coarse + fine, bounded pre-scale
    "M2": ((3, 0.25, 1600), (5, 0.30, 2560)),
    "M3": ((3, 0.25, 1600), (4, 0.25, 1600), (6, 0.30, 1600)),
    "M4": ((5, 0.30, 2560), (8, 0.30, 2560)),        # detail-dense pair
    "M5": ((2, 0.25, 1600), (4, 0.25, 1600), (6, 0.30, 1600)),
}

# aggregation name -> function over [T, D] tile features
AGGS = {
    "base": lambda f: np.concatenate([f.mean(0), f.max(0), f.std(0)]).astype(np.float32),
    "meanmax": lambda f: np.concatenate([f.mean(0), f.max(0)]).astype(np.float32),
    "rich": lambda f: np.concatenate([f.mean(0), f.max(0), f.std(0), f.min(0),
                                      np.median(f, axis=0)]).astype(np.float32),
    "spread": lambda f: np.concatenate([f.mean(0), f.max(0), f.std(0),
                                        f.max(0) - f.min(0),
                                        f.mean(0) - f.min(0)]).astype(np.float32),
    "lse": lambda f: np.concatenate([f.mean(0), f.max(0), f.std(0),
                                     np.log(np.exp(f * 8.0).mean(0)) / 8.0]).astype(np.float32),
}


def resolve_clip(name: str) -> str:
    p = Path(name)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p)
    for root in (ROOT / "models" / "hf_cache", ROOT.parent / "models" / "hf_cache"):
        snaps = root / ("models--" + CLIP_REPO.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            for d in sorted(snaps.iterdir(), reverse=True):
                if (d / "config.json").is_file():
                    return str(d)
    raise SystemExit(f"CLIP snapshot not found for {name!r}")


def tile_boxes(h: int, w: int, n: int, overlap: float):
    """Overlapping nxn tile boxes.

    ``n - (n-1)*overlap`` is the number of tile-widths spanned, which is fractional for
    many (n, overlap) pairs (8/0.30 -> 5.9). It is rounded UP so the grid always covers
    the image; rounding down would clip the right/bottom edge.
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


def tiles_for(img: np.ndarray, grid: int, overlap: float, max_side: int | None):
    import cv2
    h, w = img.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    hh, ww = img.shape[:2]
    out = []
    for (x0, y0, x1, y1) in tile_boxes(hh, ww, grid, overlap):
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        out.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
    return out or [img]


def extract(configs: list[str], data_dir: Path, cache: Path, device: str,
            batch_size: int, force: bool = False):
    seqs, labels = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            labels.append(1 if r["label"] == "ai" else 0)
    y = np.array(labels)

    cache.mkdir(parents=True, exist_ok=True)
    todo = [c for c in configs if force or not (cache / f"tiles_{c}.npy").exists()]
    if not todo:
        print("[cache] all tile feature cells present")
        return seqs, y
    ext = CLIPFeatureExtractor(resolve_clip(""), device=device, input_size=224)
    for c in todo:
        scales = CONFIGS[c]
        out_path = cache / f"tiles_{c}.npy"
        per_image, counts = [], []
        t0 = time.perf_counter()
        for i, s in enumerate(seqs):
            parsed = parse_image(str(data_dir / "images" / f"{s:03d}.png"), keep_full=True)
            if not parsed.ok:
                raise SystemExit(f"cannot parse {s}: {parsed.error}")
            parts, n_tiles = [], 0
            for (grid, overlap, max_side) in scales:
                tiles = tiles_for(parsed.bgr_full, grid, overlap, max_side)
                n_tiles += len(tiles)
                # one forward batch per scale keeps the GPU busy with the whole grid
                parts.append(ext.batch(tiles, batch_size=len(tiles)).astype(np.float32))
            per_image.append(parts)
            counts.append(n_tiles)
            if (i + 1) % 20 == 0:
                print(f"  [{c}] {i+1}/{len(seqs)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        np.save(out_path, np.array(per_image, dtype=object), allow_pickle=True)
        agg = aggregate_multi(per_image, "base")
        np.save(cache / f"agg_{c}.npy", agg)
        meta = {"config": c, "scales": [{"grid": g, "overlap": o, "max_side": m}
                                        for (g, o, m) in scales],
                "n": len(per_image), "tiles_per_image": counts,
                "tiles_p50": int(np.median(counts)),
                "extract_s": round(time.perf_counter() - t0, 1)}
        (cache / f"meta_{c}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[extract] {c} scales={meta['scales']} {len(per_image)} images, "
              f"tiles median={meta['tiles_p50']}, {meta['extract_s']}s", flush=True)
    return seqs, y


def load_tiles(cache: Path, c: str):
    """Load per-image tile features -> list of per-scale [T_i, D] arrays.

    A single-scale cache may hold a bare [T, D] array per image; a multi-scale cache
    holds one array per scale. Both layouts are accepted so an old cache stays usable.
    """
    a = np.load(cache / f"tiles_{c}.npy", allow_pickle=True)
    out = []
    for img in a:
        if isinstance(img, np.ndarray) and img.ndim == 2:
            out.append([np.asarray(img, dtype=np.float32)])            # legacy single scale
        else:
            out.append([np.asarray(part, dtype=np.float32) for part in img])
    return out


def aggregate_multi(per_image_scales, agg_name: str) -> np.ndarray:
    """Concatenate each scale's aggregation -> [sum(scale_dims)] per image."""
    fn = AGGS[agg_name]
    return np.stack([np.concatenate([fn(part) for part in parts])
                     for parts in per_image_scales]).astype(np.float32)


def fuse_weighted(logits: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    w = np.array([weights[k] for k in logits], dtype=np.float64)
    w = w / w.sum()
    L = np.stack([logits[k] for k in logits], axis=1)
    return (L * w).sum(axis=1)


def _fit_head(Xtr, ytr, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def _logits(sc, clf, Xq):
    return clf.decision_function(sc.transform(Xq))


def _fit_temperature(z, y):
    """Single-parameter temperature scaling on logits (Guo et al., ICML 2017)."""
    from scipy.optimize import minimize
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def nll(t):
        t = float(np.clip(t[0], 1e-2, 100.0))
        q = z / t
        return float(np.mean(np.maximum(q, 0) - q * y + np.log1p(np.exp(-np.abs(q)))))

    res = minimize(nll, x0=[1.0], method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-4})
    return float(np.clip(res.x[0], 1e-2, 100.0))


def _fused(z: dict[str, np.ndarray], temps: dict[str, float], weights: dict[str, float]):
    """Weighted-logit fusion exactly as src/fusion.py computes it."""
    names = [n for n in z if weights.get(n, 0.0) > 0]
    w = np.array([weights[n] for n in names], dtype=np.float64)
    w = w / w.sum()
    ps = np.stack([1.0 / (1.0 + np.exp(-z[n] / temps[n])) for n in names], axis=1)
    eps = 1e-6
    ps = np.clip(ps, eps, 1 - eps)
    L = np.log(ps / (1 - ps))
    return (L * w).sum(axis=1)


def _fpr_threshold(real_scores, fp_budget: int) -> float:
    s = np.sort(np.asarray(real_scores, dtype=np.float64))[::-1]
    k = max(0, min(int(fp_budget), len(s) - 1))
    return float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)


def _acc_at_fp(score: np.ndarray, y: np.ndarray, fp_budget: int) -> dict:
    """Best accuracy on the pooled OOF scores while admitting <= fp_budget real items.

    This is the operating point the competition's 3% rule maps to on this set
    (floor(0.03*70)=2 false positives per evaluation batch). It is an ORACLE over the
    pooled scores - reported because the per-fold threshold in the honest protocol is
    deliberately not tuned on held-out data, so this number is the reachable ceiling
    for the same score ranking, not a deployable threshold.
    """
    best = {"acc": 0.0, "thr": None, "fp": None, "fn": None}
    for t in np.unique(np.concatenate([score, [score.max() + 1e-9]])):
        p = (score >= t).astype(int)
        fp = int(((p == 1) & (y == 0)).sum())
        if fp > fp_budget:
            continue
        a = float((p == y).mean())
        if a > best["acc"]:
            best = {"acc": round(a, 4), "thr": round(float(t), 5), "fp": fp,
                    "fn": int(((p == 0) & (y == 1)).sum())}
    return best


def evaluate_fused(XF: np.ndarray, XC: np.ndarray, y: np.ndarray, n_folds: int,
                   fp_budget: int, weights: dict[str, float], seeds=(42,),
                   C: float = 1.0, select_fold_threshold: bool = False,
                   threshold_seeds=(11, 22, 33)):
    """Headline protocol: forensic + combined fusion, both fit on the training fold.

    ``select_fold_threshold=False`` reproduces the archived protocol (threshold from
    the nested-CV out-of-fold scores of the training fold). With
    ``select_fold_threshold=True`` the per-fold threshold is instead chosen to reach
    the false-positive budget on the *held-out* fold, which is optimistically biased
    and is reported only as an upper reference - the honest no-selection number comes
    from the nested CV split, which never selects the threshold on a held-out fold
    (see docs/../reports/threshold_selection.md).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    inner_splits = max(3, n_folds - 1)
    members = [m for m in ("combined", "forensic") if weights.get(m, 0.0) > 0]
    out = {k: [] for k in ("acc", "fpr", "tpr", "auc")}
    per_seed = []
    fp_counts = np.zeros(len(y))
    fn_counts = np.zeros(len(y))
    oof_acc = np.zeros(len(y))          # mean OOF fused score, for error attribution
    for seed in seeds:
        oof = np.zeros(len(y))
        pred = np.zeros(len(y), dtype=int)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(XF, y)):
            heads_tr = {}
            if "forensic" in members:
                heads_tr["forensic"] = _fit_head(XF[tr], y[tr], C)
            if "combined" in members:
                heads_tr["combined"] = _fit_head(XC[tr], y[tr], C)
            z_va = {m: _logits(*heads_tr[m], XC[va] if m == "combined" else XF[va])
                    for m in members}
            # inner nested CV over the training fold -> honest score scale
            inner = {m: np.zeros(len(tr)) for m in members}
            iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                   random_state=100 + fold)
            for itr, iva in iskf.split(XF[tr], y[tr]):
                for m in members:
                    a, b = _fit_head(XF[tr][itr] if m == "forensic" else XC[tr][itr],
                                     y[tr][itr], C)
                    inner[m][iva] = _logits(a, b,
                                            XF[tr][iva] if m == "forensic" else XC[tr][iva])
            temps = {m: _fit_temperature(inner[m], y[tr]) for m in members}
            train_fused = _fused({m: inner[m] for m in members}, temps, weights)
            if select_fold_threshold:
                # optimistic reference: pick the threshold on the fold being scored
                va_fused = _fused(z_va, temps, weights)
                cand = np.unique(np.concatenate([[va_fused.min() - 1e-6], va_fused,
                                                 [va_fused.max() + 1e-6]]))
                real_va = va_fused[y[va] == 0]
                ok = [t for t in cand if (real_va >= t).sum() <= fp_budget]
                thr = float(max(ok)) if ok else float(cand[-1])
            else:
                thr = _fpr_threshold(train_fused[y[tr] == 0], fp_budget)
            fva = _fused(z_va, temps, weights)
            oof[va] = fva
            pred[va] = (fva >= thr).astype(int)
        a = float((pred == y).mean())
        f = float(pred[y == 0].mean())
        t = float(pred[y == 1].mean())
        oof_acc += oof / len(seeds)
        fp_counts += ((pred == 1) & (y == 0)).astype(float)
        fn_counts += ((pred == 0) & (y == 1)).astype(float)
        out["acc"].append(a)
        out["fpr"].append(f)
        out["tpr"].append(t)
        out["auc"].append(float(roc_auc_score(y, oof)))
        oracle = _acc_at_fp(oof, y, fp_budget)
        per_seed.append({"seed": seed, "acc": round(a, 4), "fpr": round(f, 4),
                         "tpr": round(t, 4), "acc_at_fp_budget": oracle["acc"]})
    return {"members": members, "weights": weights,
            "acc_mean": round(float(np.mean(out["acc"])), 4),
            "acc_min": round(float(np.min(out["acc"])), 4),
            "fpr_mean": round(float(np.mean(out["fpr"])), 4),
            "tpr_mean": round(float(np.mean(out["tpr"])), 4),
            "auc_mean": round(float(np.mean(out["auc"])), 4),
            "acc_at_fp_budget_mean": round(float(np.mean(
                [s["acc_at_fp_budget"] for s in per_seed])), 4),
            "per_seed": per_seed,
            "fp_counts": fp_counts.tolist(), "fn_counts": fn_counts.tolist(),
            "oof_score_mean": [round(float(v), 5) for v in oof_acc]}


def evaluate_cell(X: np.ndarray, y: np.ndarray, n_folds: int, fp_budget: int,
                  seeds=(42,), C=1.0):
    """Stratified k-fold with the repo's nested-CV threshold rule.

    The threshold must come from OUT-OF-FOLD scores on the training fold: in-sample
    training-fold scores are saturated (the head has already fit those very samples),
    so an order statistic taken from them does not transfer and the realised FPR
    explodes. This mirrors scripts/evaluate_system.py exactly, restricted to the one
    head under test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    def fit_head(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
        return sc, clf

    def logits(sc, clf, Xq):
        return clf.decision_function(sc.transform(Xq))

    inner_splits = max(3, n_folds - 1)
    accs, aucs, fprs, tprs, coverages = [], [], [], [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pred = np.zeros(len(y), dtype=int)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            sc, clf = fit_head(X[tr], y[tr])
            z_va = logits(sc, clf, X[va])
            # nested inner CV over the TRAINING fold -> transferable score scale
            inner = np.zeros(len(tr))
            iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                   random_state=100 + fold)
            for itr, iva in iskf.split(X[tr], y[tr]):
                s1, c1 = fit_head(X[tr][itr], y[tr][itr])
                inner[iva] = logits(s1, c1, X[tr][iva])
            real = np.sort(inner[y[tr] == 0])[::-1]
            k = max(0, min(fp_budget, len(real) - 1))
            thr = real[k] if k == 0 else (real[k - 1] + real[k]) / 2.0
            oof[va] = z_va
            pred[va] = (z_va >= thr).astype(int)
        accs.append(float((pred == y).mean()))
        aucs.append(float(roc_auc_score(y, oof)))
        fprs.append(float(pred[y == 0].mean()))
        tprs.append(float(pred[y == 1].mean()))
        # coverage at the review band is not modelled here; report the raw operating point
        coverages.append(float(pred.mean()))
    return {"n_features": int(X.shape[1]),
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_min": round(float(np.min(accs)), 4),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "fpr_mean": round(float(np.mean(fprs)), 4),
            "tpr_mean": round(float(np.mean(tprs)), 4),
            "per_seed_acc": [round(a, 4) for a in accs],
            "per_seed_fpr": [round(a, 4) for a in fprs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports/tile_clip_sweep.json")
    ap.add_argument("--cache", default="models/tile_cache")
    ap.add_argument("--configs", default="C0,C1,C3,C5,C6,C7,C8")
    ap.add_argument("--aggregations", default="base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timing", action="store_true",
                    help="measure warm per-config tile latency with the resident backbone")
    ap.add_argument("--timing-samples", type=int, default=20)
    args = ap.parse_args()

    if args.list:
        for k, v in CONFIGS.items():
            scales = " + ".join(f"{g}x{g}@{o}/max{m}" for (g, o, m) in v)
            print(f"  {k}: {scales}")
        print("aggregations:", ", ".join(AGGS))
        return 0

    data_dir = Path(args.data_dir)
    cache = Path(args.cache)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    aggs = [a.strip() for a in args.aggregations.split(",") if a.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(","))
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; see --list")
    for a in aggs:
        if a not in AGGS:
            raise SystemExit(f"unknown aggregation {a}; see --list")

    extract(configs, data_dir, cache, args.device, args.batch_size, args.force_extract)
    seqs, labels = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            labels.append(1 if r["label"] == "ai" else 0)
    y = np.array(labels)

    # context features = L1(47) + L1b(20) + CLIP global(1024), identical for every cell
    X1 = np.loadtxt(ROOT / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(ROOT / "models" / "l1b_feats.npy")
    Fg = np.load(ROOT / "models" / "clip_feats.npy")
    X_ctx = np.concatenate([X1, Xb, Fg], axis=1).astype(np.float64)
    print(f"[context] {X_ctx.shape}")

    results = []
    for c in configs:
        tiles = load_tiles(cache, c)
        for a in aggs:
            agg = aggregate_multi(tiles, a).astype(np.float64)
            XC = np.concatenate([X_ctx, agg], axis=1)
            r = evaluate_cell(XC, y, args.n_folds, args.fp_budget, seeds)
            # headline protocol: forensic (+) combined fusion, both fit on the train fold
            fz = evaluate_fused(X_ctx, XC, y, args.n_folds, args.fp_budget,
                                {"combined": 0.60, "forensic": 0.25}, seeds)
            r["fused"] = fz
            r.update({"config": c, "aggregation": a,
                      "scales": [{"grid": g, "overlap": o, "max_side": m}
                                 for (g, o, m) in CONFIGS[c]],
                      "tile_dim": int(agg.shape[1])})
            results.append(r)
            print(f"[eval] {c:3s} agg={a:7s} dim={agg.shape[1]:5d} "
                  f"head auc={r['auc_mean']:.4f} acc={r['acc_mean']:.4f} | "
                  f"fused acc={fz['acc_mean']:.4f} (min {fz['acc_min']:.4f}) "
                  f"auc={fz['auc_mean']:.4f} fpr={fz['fpr_mean']:.4f} "
                  f"tpr={fz['tpr_mean']:.4f} | "
                  f"acc@2FP={fz['acc_at_fp_budget_mean']:.4f}", flush=True)

    timing = None
    if args.timing:
        timing = {}
        from l2_clip import CLIPFeatureExtractor as _E
        ext = _E(resolve_clip(""), device=args.device, input_size=224)
        paths = [str(data_dir / "images" / f"{s:03d}.png") for s in seqs[:args.timing_samples]]
        parsed = [parse_image(p, keep_full=True) for p in paths]
        for c in configs:
            scales = CONFIGS[c]
            for (g, o, m) in scales:                       # warm-up every scale
                ext.batch(tiles_for(parsed[0].bgr_full, g, o, m))
            t0 = time.perf_counter()
            n_tiles = 0
            for p in parsed:
                for (g, o, m) in scales:
                    ts = tiles_for(p.bgr_full, g, o, m)
                    n_tiles += len(ts)
                    ext.batch(ts, batch_size=len(ts))
            dt = (time.perf_counter() - t0) * 1000 / len(parsed)
            timing[c] = {"tile_ms_per_image": round(dt, 1),
                         "tiles_per_image": round(n_tiles / len(parsed), 1)}
            print(f"[timing] {c}: {dt:.0f} ms/image over {len(parsed)} images "
                  f"({n_tiles/len(parsed):.0f} tiles)", flush=True)

    payload = {
        "report": "tile_clip_sweep",
        "protocol": {
            "cv": f"stratified {args.n_folds}-fold, seeds={list(seeds)}",
            "threshold": f"order statistic of NESTED-CV out-of-fold training-fold real "
                         f"scores, fp_budget={args.fp_budget}",
            "context_features": "L1(47) + L1b(20) + CLIP global(1024)",
            "note": "no selection on held-out folds; every aggregation is scored by the same protocol",
        },
        "configs": {k: [{"grid": g, "overlap": o, "max_side": m} for (g, o, m) in v]
                    for k, v in CONFIGS.items() if k in configs},
        "results": results,
        "timing": timing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")

    best = max(results, key=lambda r: (r["fused"]["acc_mean"], r["fused"]["auc_mean"]))
    print(f"[best] {best['config']} / {best['aggregation']} -> fused acc="
          f"{best['fused']['acc_mean']} auc={best['fused']['auc_mean']} "
          f"fpr={best['fused']['fpr_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
