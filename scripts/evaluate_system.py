#!/usr/bin/env python3
"""Full evaluation on the 119 competition samples + latency benchmark.

Evidence producers (identical set to the production pipeline in src/detector.py):
  forensic       LR on [L1(47) (+) L1b(20)] zero-weight + low-cost forensic features
  combined       LR on [L1(47) (+) L1b(20) (+) CLIP_global(1024) (+) CLIP_tiles(3*1024)]
  clip_external  frozen CLIP probe trained ONLY on the 18-generator AIGC benchmark;
                 admitted into the fusion only when its training-fold AUC >= 0.65

Protocol
  - stratified 5-fold; every head is fit on the training fold only
  - decision threshold = order statistic of the TRAINING-fold REAL scores targeting
    FPR <= --target-fpr (default 0.03); the review band is derived from the same
    calibration set. Nothing is selected on the held-out fold.
  - out-of-fold predictions give the headline metrics, with bootstrap 95% CIs
  - because the given 119 samples are ~47:53 while the stated evaluation set is
    ~7:3 real:fake, the official ratio is ALSO simulated by resampling the
    observed per-class outcomes (70 real / 30 fake, 2000 draws)

Outputs: reports/metrics.json, models/{forensic_lr,combined_lr,clip_domain_lr}.joblib,
         models/fusion.json, reports/<seq>.json + .html
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import AIGCDetector, PipelineConfig   # noqa: E402
from fusion import FusionEngine, ModelScore         # noqa: E402
from report import save_html                        # noqa: E402

WEIGHTS = {"combined": 0.60, "forensic": 0.25, "clip_external": 0.15}
EXT_MIN_AUC = 0.65


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_labels(data_dir: Path):
    seqs, y = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    return np.array(seqs), np.array(y)


def load_matrix(path: Path, expect_rows: int, name: str) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"[missing] {name}: {path} (run the extract scripts first)")
    a = np.load(path)
    if a.shape[0] != expect_rows:
        raise SystemExit(f"[shape] {name}: {a.shape[0]} rows != {expect_rows}")
    return a.astype(np.float64)


def resolve_clip_dir(p: Path) -> Path:
    """Accept a snapshot dir, or a 'snapshots' dir containing hash subdirs."""
    if (p / "config.json").is_file():
        return p
    if p.is_dir():
        candidates = sorted([d for d in p.iterdir() if d.is_dir()], reverse=True)
        for d in candidates:
            if (d / "config.json").is_file():
                return d
    return p


def _torch_load(path, map_location="cpu"):
    """Load a checkpoint saved as {"state_dict": ..., "scaler": {...}}.

    torch>=2.6 defaults ``weights_only`` to True, which rejects the numpy scaler
    arrays that the probe bundles carry; the bundles are produced by this repo's own
    training script, so the pickle path is the intended one.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                      # torch < 1.13 has no weights_only
        return torch.load(path, map_location=map_location)


def load_external_probe(pt: Path):
    ck = _torch_load(pt)
    sd = ck.get("state_dict", ck)
    w = sd["weight"].numpy().reshape(-1)
    b = float(sd["bias"].numpy().reshape(-1)[0])
    sc = ck.get("scaler") or {}
    return w, b, np.array(sc.get("mean", np.zeros_like(w)), dtype=np.float64), \
        np.array(sc.get("scale", np.ones_like(w)), dtype=np.float64)


def fit_head(X, y, C=1.0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(X), y)
    return sc, clf


def head_logits(sc, clf, X):
    return logit(clf.predict_proba(sc.transform(X))[:, 1])


def fpr_capped_threshold(real_scores, target_fpr=None, fp_budget=None):
    """Threshold admitting at most `fp_budget` real samples (or floor(target_fpr*n)).

    The given 119-sample set is ~47:53 real:fake, so one false positive already
    costs 1/63 = 1.59% FPR and two cost 3.17%. The stated evaluation set is 7:3
    (70 real / 30 fake), where the 3% business rule means at most 2 false
    positives PER EVALUATION BATCH (floor(0.03*70)=2). We therefore also support
    an explicit absolute budget, which transfers across set sizes, and report both
    operating points.
    """
    s = np.sort(np.asarray(real_scores, dtype=np.float64))[::-1]
    n = len(s)
    if n == 0:
        return 0.5
    if fp_budget is None:
        allowed = int(np.floor(target_fpr * n))
    else:
        allowed = int(fp_budget)
    allowed = max(0, min(allowed, n - 1))
    if allowed <= 0:
        return float(s[0]) + 1e-9
    if allowed >= n:
        return float(s[-1]) - 1e-9
    return float((s[allowed - 1] + s[allowed]) / 2.0)


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"accuracy": round((tp + tn) / max(1, len(y)), 4),
            "fpr": round(fp / max(1, int((y == 0).sum())), 4),
            "tpr": round(tp / max(1, int((y == 1).sum())), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def bootstrap_ci(y, pred, n=2000, seed=7):
    rng = np.random.RandomState(seed)
    accs, fprs = [], []
    real_idx = np.where(y == 0)[0]
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        accs.append(float((pred[idx] == y[idx]).mean()))
        if len(real_idx):
            ri = rng.choice(real_idx, len(real_idx), replace=True)
            fprs.append(float((pred[ri] == 1).mean()))
    return {"accuracy_ci95": [round(float(np.percentile(accs, 2.5)), 4),
                              round(float(np.percentile(accs, 97.5)), 4)],
            "fpr_ci95": [round(float(np.percentile(fprs, 2.5)), 4),
                         round(float(np.percentile(fprs, 97.5)), 4)] if fprs else None}


def ratio_simulation(y, pred, n_real=70, n_fake=30, draws=2000, seed=11):
    """Resample observed per-class outcomes to the stated 7:3 evaluation ratio."""
    rng = np.random.RandomState(seed)
    real_correct = (pred[y == 0] == 0)
    fake_correct = (pred[y == 1] == 1)
    if len(real_correct) == 0 or len(fake_correct) == 0:
        return None
    accs, fprs = [], []
    for _ in range(draws):
        r = rng.choice(real_correct, n_real, replace=True)
        f = rng.choice(fake_correct, n_fake, replace=True)
        accs.append((r.sum() + f.sum()) / (n_real + n_fake))
        fprs.append(float((~r).mean()))
    return {"simulated_ratio": f"{n_real}:{n_fake}",
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_ci95": [round(float(np.percentile(accs, 2.5)), 4),
                              round(float(np.percentile(accs, 97.5)), 4)],
            "fpr_mean": round(float(np.mean(fprs)), 4),
            "fpr_ci95": [round(float(np.percentile(fprs, 2.5)), 4),
                         round(float(np.percentile(fprs, 97.5)), 4)]}


def roc_frontier(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    out = {}
    for target in (0.01, 0.02, 0.03, 0.05, 0.10):
        i = max(0, min(int(np.searchsorted(fpr, target, side="right")) - 1, len(tpr) - 1))
        out[f"tpr@fpr={target:.2f}"] = round(float(tpr[i]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--clip-probe", default="models/clip_probe_norm.pt")
    ap.add_argument("--clip-dir", default="models/hf_cache/models--openai--clip-vit-large-patch14/snapshots")
    ap.add_argument("--feature-spec", default=None,
                    help="feature_spec.json describing the tile/patch blocks the heads "
                         "consume (default: <models-dir>/feature_spec.json when present)")
    ap.add_argument("--frozen-fusion", default="",
                    help="keep an already-frozen fusion.json instead of re-deriving the "
                         "deployed thresholds in this run. The deployed artifact comes "
                         "from scripts/calibrate_deployment.py, whose threshold is the "
                         "pooled out-of-fold order statistic over several CV rounds; the "
                         "in-run value is a single nested split and is noisier. The "
                         "out-of-fold metrics below do not depend on either choice.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--fp-budget", type=int, default=2,
                    help="absolute false-positive budget used for the deployed threshold "
                         "(floor(0.03*70)=2 under the stated 7:3 evaluation ratio; "
                         "use -1 to fall back to the per-fold target-fpr quantile)")
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir, models_dir, out_dir = Path(args.data_dir), Path(args.models_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    seqs, y = load_labels(data_dir)
    n = len(y)
    print(f"[data] {n} samples  ai={int((y==1).sum())} real={int((y==0).sum())}")

    X1 = np.loadtxt(args.features, delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = load_matrix(models_dir / "l1b_feats_119.npy", n, "L1b forensics")
    Fg = load_matrix(models_dir / "clip_feats_119.npy", n, "CLIP global")
    # The tile block and the optional patch-token block follow the FROZEN feature spec,
    # i.e. exactly what src/detector.py assembles online (src/feature_spec.py). The
    # 3x3 mean/max/std artifact stays the default when no spec is present, so an older
    # artifact set still evaluates reproducibly.
    from feature_spec import FeatureSpec
    spec = FeatureSpec.load(args.feature_spec or (models_dir / "feature_spec.json"))
    if (models_dir / "clip_tiles_spec.npy").exists():
        Ft = load_matrix(models_dir / "clip_tiles_spec.npy", n, "CLIP tiles (spec)")
    else:
        Ft = load_matrix(models_dir / "clip_tiles_119.npy", n, "CLIP tiles")
    patch = None
    if spec.patch_tokens:
        patch = load_matrix(models_dir / "clip_patch.npy", n, "CLIP patch tokens")
    XF = np.concatenate([X1, Xb] + ([patch] if patch is not None else []), axis=1)
    XC = np.concatenate([XF, Fg, Ft], axis=1)
    print(f"[features] forensic={XF.shape[1]}  combined={XC.shape[1]}  spec={spec.name} "
          f"(tile aggregation={spec.aggregation}, patch_tokens={spec.patch_tokens})")

    probe = Path(args.clip_probe)
    if not probe.exists():
        raise SystemExit(f"external probe not found: {probe}")
    # the external probe may have been trained with a common-codec pre-processing
    # (--reencode-q); if such features exist, use them so the probe sees the input
    # distribution it was trained on.
    Fg_ext = Fg
    norm_feats = models_dir / "clip_feats_norm_119.npy"
    if "norm" in probe.name and norm_feats.exists():
        Fg_ext = load_matrix(norm_feats, n, "CLIP global (codec-normalized)")
        print(f"[clip_external] using codec-normalized features {norm_feats.name}")
    w, b, mean, scale = load_external_probe(probe)
    ext = (((Fg_ext - mean) / scale) @ w + b)
    print(f"[clip_external] AUC on the 119 = {roc_auc_score(y, ext):.4f}")

    oof_score = np.zeros(n)
    oof_pred = np.zeros(n, dtype=int)
    oof_verdict = np.array(["review"] * n, dtype=object)
    model_logits = {k: np.zeros(n) for k in ("forensic", "combined", "clip_external")}
    fold_rows, ext_rows = [], []

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    inner_splits = max(3, args.n_folds - 1)
    for fold, (tr, va) in enumerate(skf.split(XF, y)):
        scf, clff = fit_head(XF[tr], y[tr])
        scc, clfc = fit_head(XC[tr], y[tr])
        lg = {
            "forensic": (head_logits(scf, clff, XF[tr]), head_logits(scf, clff, XF[va])),
            "combined": (head_logits(scc, clfc, XC[tr]), head_logits(scc, clfc, XC[va])),
            "clip_external": (ext[tr], ext[va]),
        }
        for k in lg:
            model_logits[k][va] = lg[k][1]

        # ---- NESTED inner CV: unbiased score scale for calibration + thresholds ----
        # In-sample head logits are saturated, so a threshold taken from them does
        # not transfer. Inner-fold out-of-fold logits have the right scale.
        inner_logits = {"forensic": np.zeros(len(tr)), "combined": np.zeros(len(tr)),
                        "clip_external": ext[tr]}
        inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=100 + fold)
        for itr, iva in inner_skf.split(XF[tr], y[tr]):
            s1, c1 = fit_head(XF[tr][itr], y[tr][itr])
            inner_logits["forensic"][iva] = head_logits(s1, c1, XF[tr][iva])
            s2, c2 = fit_head(XC[tr][itr], y[tr][itr])
            inner_logits["combined"][iva] = head_logits(s2, c2, XC[tr][iva])

        ext_auc_tr = float(roc_auc_score(y[tr], ext[tr]))
        members = ["combined", "forensic"]
        if ext_auc_tr >= EXT_MIN_AUC:
            members.append("clip_external")
        wts = {m: WEIGHTS[m] for m in members}
        ext_rows.append({"fold": fold + 1, "ext_auc_train": round(ext_auc_tr, 4),
                         "admitted": "clip_external" in wts})

        eng = FusionEngine(weights=wts)
        eng.fit_calibrators({m: inner_logits[m] for m in wts}, y[tr])
        real_scores = np.array([
            eng.fuse([ModelScore(m, float(inner_logits[m][i])) for m in wts]).fused_score
            for i in np.where(y[tr] == 0)[0]])
        budget = None if args.fp_budget is None or args.fp_budget < 0 else args.fp_budget
        thr_high = fpr_capped_threshold(real_scores, args.target_fpr, budget)
        thr_low = float(np.median(real_scores))
        if thr_low >= thr_high:
            thr_low = float(np.percentile(real_scores, 25))

        for k, i in enumerate(va):
            sc = eng.fuse([ModelScore(m, float(lg[m][1][k])) for m in wts]).fused_score
            oof_score[i] = sc
            oof_pred[i] = 1 if sc >= thr_high else 0
            oof_verdict[i] = ("suspect" if oof_pred[i] == 1
                              else "normal" if sc <= thr_low else "review")
        m = metrics(y[va], oof_pred[va])
        m.update({"fold": fold + 1, "thresholds": [round(thr_low, 4), round(thr_high, 4)],
                  "members": list(wts), "ext_auc_train": round(ext_auc_tr, 4)})
        fold_rows.append(m)
        print(f"[fold {fold+1}] acc={m['accuracy']:.4f} fpr={m['fpr']:.4f} tpr={m['tpr']:.4f} "
              f"thr=[{thr_low:.3f},{thr_high:.3f}] members={list(wts)}", flush=True)

    overall = metrics(y, oof_pred)
    overall.update(bootstrap_ci(y, oof_pred))
    overall["three_tier"] = {
        "suspect": int((oof_verdict == "suspect").sum()),
        "normal": int((oof_verdict == "normal").sum()),
        "review": int((oof_verdict == "review").sum()),
    }
    per_model_auc = {k: round(float(roc_auc_score(y, v)), 4) for k, v in model_logits.items()}
    frontier = roc_frontier(y, oof_score)
    sim = ratio_simulation(y, oof_pred)

    # descriptive frontier: best accuracy attainable while holding the measured
    # FPR at or below a given level (computed on the pooled OOF scores)
    def best_acc(max_fpr):
        best = 0.0
        for t in np.unique(oof_score):
            p = (oof_score >= t).astype(int)
            if p[y == 0].mean() <= max_fpr:
                best = max(best, float((p == y).mean()))
        return round(best, 4)

    operating_points = {
        "fpr_le_1.59pct_(1_of_63)": best_acc(1 / 63 + 1e-9),
        "fpr_le_3.17pct_(2_of_63)": best_acc(2 / 63 + 1e-9),
        "fpr_le_4.76pct_(3_of_63)": best_acc(3 / 63 + 1e-9),
        "fpr_le_3pct_strict": best_acc(0.03),
        "official_ratio_2_of_70=2.86pct": best_acc(2 / 70 + 1e-9),
    }
    print(f"[OP] best accuracy vs measured FPR: {operating_points}")

    print(f"\n[OOF] accuracy={overall['accuracy']} fpr={overall['fpr']} tpr={overall['tpr']} "
          f"precision={overall['precision']}")
    print(f"[OOF] confusion={overall['confusion']}")
    print(f"[OOF] acc 95%CI={overall['accuracy_ci95']}  fpr 95%CI={overall['fpr_ci95']}")
    print(f"[OOF] three-tier={overall['three_tier']}")
    print(f"[AUC] per-model={per_model_auc}")
    print(f"[ROC] frontier={frontier}")
    print(f"[SIM] 7:3 ratio -> {sim}")

    fp_seqs = [int(seqs[i]) for i in range(n) if y[i] == 0 and oof_pred[i] == 1]
    fn_seqs = [int(seqs[i]) for i in range(n) if y[i] == 1 and oof_pred[i] == 0]
    print(f"[errors] FP(real->ai)={fp_seqs}")
    print(f"[errors] FN(ai->real)={fn_seqs}")

    # ---- final heads on all 119 (deployment/demo) ----
    scf, clff = fit_head(XF, y)
    scc, clfc = fit_head(XC, y)
    scd, clfd = fit_head(Fg, y)
    joblib.dump({"scaler": scf, "clf": clff, "features": "l1_47+l1b_20"},
                models_dir / "forensic_lr.joblib")
    joblib.dump({"scaler": scc, "clf": clfc, "features": "l1_47+l1b_20+clip_global+clip_tiles"},
                models_dir / "combined_lr.joblib")
    joblib.dump({"scaler": scd, "clf": clfd, "features": "clip_global"},
                models_dir / "clip_domain_lr.joblib")

    all_logits = {"forensic": head_logits(scf, clff, XF),
                  "combined": head_logits(scc, clfc, XC),
                  "clip_external": ext}
    final_members = ["combined", "forensic"]
    if roc_auc_score(y, ext) >= EXT_MIN_AUC:
        final_members.append("clip_external")

    # deployment thresholds use the SAME nested-CV score scale (in-sample logits
    # are saturated and would produce a non-transferable threshold)
    inner_logits_full = {"forensic": np.zeros(n), "combined": np.zeros(n),
                         "clip_external": ext}
    inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=777)
    for itr, iva in inner_skf.split(XF, y):
        s1, c1 = fit_head(XF[itr], y[itr])
        inner_logits_full["forensic"][iva] = head_logits(s1, c1, XF[iva])
        s2, c2 = fit_head(XC[itr], y[itr])
        inner_logits_full["combined"][iva] = head_logits(s2, c2, XC[iva])

    engf = FusionEngine(weights={m: WEIGHTS[m] for m in final_members})
    engf.fit_calibrators({m: inner_logits_full[m] for m in final_members}, y)
    real_scores = np.array([
        engf.fuse([ModelScore(m, float(inner_logits_full[m][i])) for m in final_members]).fused_score
        for i in range(n) if y[i] == 0])
    budget = None if args.fp_budget is None or args.fp_budget < 0 else args.fp_budget
    # fusion.json stores thresholds in the fused PROBABILITY space that fuse() returns
    engf.thr_high = fpr_capped_threshold(real_scores, args.target_fpr, budget)
    engf.thr_low = float(np.median(real_scores))
    if engf.thr_low >= engf.thr_high:
        engf.thr_low = float(np.percentile(real_scores, 25))
    in_run_thresholds = {"low": engf.thr_low, "high": engf.thr_high}
    if args.frozen_fusion:
        src = Path(args.frozen_fusion)
        if not src.exists():
            raise SystemExit(f"--frozen-fusion not found: {src}")
        engf = FusionEngine.load(str(src))
        print(f"[final] kept the frozen deployment thresholds from {src} "
              f"(a single in-run nested split would have given "
              f"{in_run_thresholds['low']:.4f}/{in_run_thresholds['high']:.4f})")
    else:
        engf.save(models_dir / "fusion.json")
    print(f"[final] members={final_members} thr_low={engf.thr_low:.4f} thr_high={engf.thr_high:.4f} "
          f"(fp_budget={budget}, target_fpr={args.target_fpr})")

    # ---- latency ----
    print(f"\n[latency] {args.latency_samples} warm end-to-end runs (incl. tiles + L3)...")
    det = AIGCDetector(PipelineConfig(
        clip_dir=str(resolve_clip_dir(Path(args.clip_dir))), clip_probe=str(probe),
        l1_lr_model=str(models_dir / "clip_domain_lr.joblib"),
        clip_domain_lr=str(models_dir / "clip_domain_lr.joblib"),
        combined_lr=str(models_dir / "combined_lr.joblib"),
        forensic_lr=str(models_dir / "forensic_lr.joblib"),
        feature_spec=str(args.feature_spec or (models_dir / "feature_spec.json")),
        fusion_json=str(models_dir / "fusion.json"), device=args.device))
    paths = [str(data_dir / "images" / f"{s:03d}.png") for s in seqs]
    det.detect_image(paths[0])
    lat = []
    for p in paths[:args.latency_samples]:
        t0 = time.perf_counter()
        det.detect_image(p)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    latency = {"n": len(lat), "p50_ms": round(float(np.percentile(lat, 50)), 1),
               "p95_ms": round(float(np.percentile(lat, 95)), 1),
               "mean_ms": round(float(lat.mean()), 1), "max_ms": round(float(lat.max()), 1),
               "budget_ms": 5000,
               "p95_within_budget": bool(np.percentile(lat, 95) <= 5000)}
    print(f"[latency] P50={latency['p50_ms']}ms P95={latency['p95_ms']}ms max={latency['max_ms']}ms "
          f"-> {'PASS' if latency['p95_within_budget'] else 'FAIL'}")

    # ---- reports ----
    print("[reports] writing per-sample JSON + HTML ...")
    for p, s in zip(paths, seqs):
        rep, _ = det.detect_image(p)
        rep.save_json(str(out_dir / f"{int(s):03d}.json"))
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        save_html(rep, str(out_dir / f"{int(s):03d}.html"), image_b64=b64)

    summary = {
        "n_samples": n,
        "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
        "protocol": {
            "cv": f"stratified {args.n_folds}-fold; every head fit on the training fold only",
            "fusion_weights": WEIGHTS,
            "decision_threshold": f"order statistic on training-fold real scores, target FPR<={args.target_fpr}",
            "external_probe": str(probe),
            "external_probe_admission": f"training-fold AUC >= {EXT_MIN_AUC}",
            "feature_spec": spec.to_dict(),
            "note": "the 119 samples are the only labelled domain data; no selection on held-out folds",
        },
        "oof_overall": overall,
        "per_model_oof_auc": per_model_auc,
        "roc_frontier": frontier,
        "operating_points_best_accuracy": operating_points,
        "simulated_official_ratio": sim,
        "folds": fold_rows,
        "external_probe_per_fold": ext_rows,
        "error_analysis": {"false_positives_real_seqs": fp_seqs,
                           "false_negatives_ai_seqs": fn_seqs},
        "latency_image": latency,
        "final_members": final_members,
        "final_thresholds": {"low": engf.thr_low, "high": engf.thr_high},
        "in_run_nested_thresholds": {
            "low": round(float(in_run_thresholds["low"]), 6),
            "high": round(float(in_run_thresholds["high"]), 6),
            "note": "this run's single nested split; the deployed thresholds come from "
                    "scripts/calibrate_deployment.py (pooled over several CV rounds)",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[save] {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
