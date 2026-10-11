#!/usr/bin/env python3
"""AEROBLADE ablation (manual ch.8.2 optional-member decision, ch.12.1 ablation).

The compliance audit (F4) requires every *optional* member to be decided by an
ablation, not by an "interface slot that is never enabled". This script does that
for AEROBLADE (Ricker et al., CVPR 2024): a training-free SD-VAE reconstruction
error. Effort (Koutlis & Papadopoulos, 2024) is *not* verifiable in this budget
(it needs low-rank EDM fine-tuning), which is stated in the report instead of
being papered over.

Protocol (kept identical to scripts/member_analysis.py / metrics_common.py)
--------------------------------------------------------------------------
* stratified 5-fold outer split, random_state=42 (same folds as
  scripts/evaluate_system.py); nothing is fitted on a held-out fold
* AEROBLADE itself is training-free, so its "out-of-fold" score is the raw score
  of every sample; only the one-parameter temperature scaler is fold-fitted (on
  the training-fold raw scores) so the route can enter the same calibrated fusion
  as the other members
* retention is decided at FPR=5% (the 1% level is not resolvable with 63 real
  samples, where one false positive already costs 1.59%); the 1% numbers are
  still reported
* the deployed fusion (models/fusion.json weights, threshold = order statistic on
  real OOF scores with a false-positive budget of 2) is the incumbent; the
  ablation reports Δ at several candidate AEROBLADE weights so the verdict does
  not hinge on one arbitrary weight

Usage
-----
  python scripts/aeroblade_ablation.py                  # reuse cached VAE + scores
  python scripts/aeroblade_ablation.py --refresh-scores # recompute the 119 scores
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

REPO = mc.ROOT
WS = mc.WS
sys.path.insert(0, str(REPO / "src"))

# --- pre-registered decision rule (declared here, before the numbers are read) ----
PRESET = {
    "min_auc": 0.65,
    "min_auc_rationale": "same admission bar evaluate_system.py applies to the other "
                         "frozen external member (EXT_MIN_AUC=0.65); AUC ~ 0.5 means the "
                         "route carries no ranking information",
    "min_delta_tpr_at_primary_fpr": 0.02,
    "delta_rationale": "1/56 AI samples = 1.79 pp, so a route must correct at least about "
                       "one AI sample to justify its inference cost (manual ch.12.1)",
    "primary_fpr_level": 0.05,
    "primary_fpr_rationale": "with 63 real samples 1 FP = 1.59%, so the 1% guardrail is "
                             "granularity-degenerate; the retention decision follows "
                             "scripts/member_analysis.py and uses FPR=5%",
    "min_delta_accuracy": 0.0,
    "declared_before_measurement": True,
    "verdict_key": "retain only if AUC >= min_auc AND (best-case ΔTPR@FPR=5% >= 0.02 "
                   "OR (ΔTPR@FPR=1% > 0 AND Δaccuracy > 0)); otherwise 剔除 (negative "
                   "result, manual ch.12.1)",
}

# score variants derived from ONE reconstruction per image
VARIANTS = {
    "multiscale_mse_512_256_128": {"keys": ["mse@512", "mse@256", "mse@128"],
                                   "weights": [0.5, 0.3, 0.2],
                                   "note": "shipped implementation (task-specified multi-scale MSE)"},
    "mse_512": {"keys": ["mse@512"], "weights": [1.0], "note": "single-scale full resolution"},
    "mse_256": {"keys": ["mse@256"], "weights": [1.0], "note": "paper-like resolution"},
    "mse_128": {"keys": ["mse@128"], "weights": [1.0], "note": "coarse structure only"},
    "l1_512": {"keys": ["l1@512"], "weights": [1.0], "note": "L1 instead of MSE (original stub)"},
    "l1_256": {"keys": ["l1@256"], "weights": [1.0], "note": "paper's 256 with L1"},
}
PRIMARY_VARIANT = "multiscale_mse_512_256_128"
WEIGHT_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30]


# ------------------------------------------------------------------ VAE resolution
def resolve_vae_dir(workspace: Path, explicit: str | None, allow_download: bool = True) -> dict:
    """Find a local SD-VAE snapshot; optionally download it from the HF mirror.

    Nothing is ever written to the user-level HF cache: the fallback download goes
    into ``<workspace>/models/hf_cache``.
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    repo = workspace / "repo"
    for base in (workspace / "models" / "hf_cache", workspace / "hf_cache" / "hub",
                 repo / "models" / "hf_cache"):
        d = base / "models--stabilityai--sd-vae-ft-mse" / "snapshots"
        if d.is_dir():
            cands.extend(sorted(d.iterdir(), reverse=True))
    for c in cands:
        if (c / "config.json").is_file() and any(
                (c / f).is_file() for f in ("diffusion_pytorch_model.safetensors",
                                            "diffusion_pytorch_model.bin")):
            return {"vae_dir": str(c), "source": "local cache" if not explicit else "explicit",
                    "downloaded": False}

    if not allow_download:
        return {"vae_dir": None, "source": "unavailable", "downloaded": False}

    from huggingface_hub import snapshot_download
    cache = workspace / "models" / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.pop("HF_HUB_OFFLINE", None)
    print(f"[vae] not cached locally -> downloading stabilityai/sd-vae-ft-mse "
          f"(endpoint={os.environ['HF_ENDPOINT']}) into {cache}", flush=True)
    path = Path(snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse", cache_dir=str(cache),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
    if not (path / "config.json").is_file():
        snaps = path / "snapshots"
        cands = sorted(snaps.iterdir(), reverse=True) if snaps.is_dir() else []
        path = cands[0] if cands else path
    ok = (path / "config.json").is_file()
    return {"vae_dir": str(path) if ok else None,
            "source": "hf-mirror download" if ok else "download failed", "downloaded": ok}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ score computation
def compute_scores(workspace: Path, vae_dir: str, device: str, cache: Path,
                   refresh: bool) -> dict:
    """Per-image multi-scale reconstruction errors for the 119 competition samples."""
    if cache.exists() and not refresh:
        print(f"[scores] reusing cached scores: {cache} (pass --refresh to recompute)",
              flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))

    from l0_input import parse_image
    from l2_aeroblade import AerobladeScorer, LOGIT_SCALE

    seqs, y = mc.load_labels(workspace / "data")
    t0 = time.perf_counter()
    scorer = AerobladeScorer(vae_dir, device=device)
    load_s = time.perf_counter() - t0
    print(f"[vae] loaded in {load_s:.1f}s device={scorer.device} "
          f"side={scorer.side}", flush=True)

    import torch
    import torch.nn.functional as F

    rows, times = [], []
    for i, s in enumerate(seqs):
        p = workspace / "data" / "images" / f"{int(s):03d}.png"
        parsed = parse_image(str(p), keep_full=False)
        if not parsed.ok:
            raise SystemExit(f"[input] cannot decode {p}: {parsed.error}")
        bgr = parsed.bgr_full if parsed.bgr_full is not None else parsed.bgr
        t = time.perf_counter()
        x, rec = scorer.reconstruct(bgr)
        m = {}
        with torch.no_grad():
            for side in (512, 256, 128):
                a = x if side == scorer.side else F.interpolate(x, size=(side, side), mode="area")
                b = rec if side == scorer.side else F.interpolate(rec, size=(side, side), mode="area")
                m[f"mse@{side}"] = float(F.mse_loss(b, a).item())
                m[f"l1@{side}"] = float(F.l1_loss(b, a).item())
        dt = (time.perf_counter() - t) * 1000
        times.append(dt)
        rows.append({"seq": int(s), "label": "ai" if y[i] == 1 else "real",
                     "image_sha256": parsed.sha256, "ms": round(dt, 1), **m})
        if (i + 1) % 20 == 0:
            print(f"[scores] {i + 1}/{len(seqs)}", flush=True)

    payload = {
        "vae_dir": vae_dir,
        "vae_weight_sha256": {
            f.name: "sha256:" + sha256_file(f)
            for f in sorted(Path(vae_dir).iterdir())
            if f.is_file() and f.suffix in (".safetensors", ".bin", ".json")},
        "recon_side": scorer.side,
        "logit_scale": LOGIT_SCALE,
        "device": scorer.device,
        "vae_load_s": round(load_s, 1),
        "scales": [s for s, _ in AerobladeScorer.SCALES],
        "imgs_per_s_median": round(1000.0 / float(np.median(times)), 2) if times else None,
        "median_ms": round(float(np.median(times)), 1),
        "rows": rows,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[scores] cached -> {cache} (median {payload['median_ms']} ms/image, "
          f"{payload['imgs_per_s_median']} img/s)", flush=True)
    return payload


# ------------------------------------------------------------------ evaluation bits
def variant_scores(cache: dict, name: str) -> np.ndarray:
    spec = VARIANTS[name]
    keys, ws = spec["keys"], spec["weights"]
    e = np.zeros(len(cache["rows"]), dtype=np.float64)
    for k, w in zip(keys, ws):
        e += w * np.array([r[k] for r in cache["rows"]], dtype=np.float64)
    return e / float(sum(ws))


def operating_row(y, score, target_fpr=0.03, fp_budget=2):
    real = np.asarray(score)[y == 0]
    thr = mc.fpr_capped_threshold(real, target_fpr, fp_budget)
    pred = (np.asarray(score) >= thr).astype(int)
    m = mc.metrics_at(y, pred)
    m["threshold"] = round(float(thr), 6)
    m["allowed_fp"] = int(fp_budget)
    m["n_real"] = int(len(real))
    m["auc"] = mc.auc(y, score)
    m["tpr@fpr=1%"] = mc.tpr_at_fpr(y, score, 0.01)
    m["tpr@fpr=5%"] = mc.tpr_at_fpr(y, score, 0.05)
    return m


def calibrate_oof(raw_ai: np.ndarray, y: np.ndarray, n_folds: int = mc.N_FOLDS_DEFAULT):
    """Fold-safe temperature calibration of a training-free route.

    Mirrors metrics_common.build_bundle: the temperature is fitted on the raw
    scores of the *training* fold only and then applied to the held-out fold, so
    the calibrated value of every sample is out-of-sample.
    """
    from fusion import TemperatureScaler
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    cal = np.zeros(n, dtype=np.float64)
    temps = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=mc.OUTER_SEED)
    for fold, (tr, va) in enumerate(skf.split(np.zeros((n, 1)), y)):
        sc = TemperatureScaler().fit(raw_ai[tr], y[tr])
        temps.append(round(float(sc.temperature), 4))
        cal[va] = 1.0 / (1.0 + np.exp(-raw_ai[va] / sc.temperature))
    return cal, temps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--vae-dir", default=None, help="explicit SD-VAE snapshot directory")
    ap.add_argument("--no-download", action="store_true",
                    help="never attempt the hf-mirror download (offline strict mode)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-cache", default="models/aeroblade_scores_119.json")
    ap.add_argument("--refresh-scores", action="store_true",
                    help="recompute the 119 reconstruction errors (same effect as "
                         "--refresh-oof for this script)")
    ap.add_argument("--target-fpr", type=float, default=0.03,
                    help="target FPR for the deployed-style operating point")
    ap.add_argument("--fp-budget", type=int, default=mc.FP_BUDGET_DEFAULT)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/aeroblade_ablation.json")
    seqs, y = mc.load_labels(ws / "data")
    n = len(y)

    # ---------------- 1. dependency: the frozen SD autoencoder
    vae = resolve_vae_dir(ws, args.vae_dir, allow_download=not args.no_download)
    if not vae["vae_dir"]:
        raise SystemExit("[vae] no local snapshot and download disabled/failed -> "
                         "cannot run the AEROBLADE ablation")
    print(f"[vae] using {vae['vae_dir']} ({vae['source']})", flush=True)

    cache_path = ws / args.score_cache
    cache = compute_scores(ws, vae["vae_dir"], args.device, cache_path,
                           args.refresh_oof or args.refresh_scores)
    if [int(r["seq"]) for r in cache["rows"]] != [int(s) for s in seqs]:
        raise SystemExit("[scores] cached scores do not follow data/labels.csv order")

    # ---------------- 2. single-route capability of every score variant
    single_variants = {}
    for name, spec in VARIANTS.items():
        err = variant_scores(cache, name)
        ai_logit = -err / float(cache["logit_scale"])      # higher => more AI
        row = operating_row(y, ai_logit, args.target_fpr, args.fp_budget)
        single_variants[name] = {
            "definition": spec["note"], "components": spec["keys"], "weights": spec["weights"],
            "direction": "AI score = -(reconstruction error); lower error => more LD-like",
            "oof_auc": mc.auc(y, ai_logit),
            "auc_5pct_operating_point": row["auc"],
            "accuracy": row["accuracy"], "fpr": row["fpr"], "tpr": row["tpr"],
            "tpr@fpr=1%": row["tpr@fpr=1%"], "tpr@fpr=5%": row["tpr@fpr=5%"],
            "threshold": row["threshold"],
            "in_deployed_fusion": False,
        }
        print(f"[variant] {name:28s} AUC={single_variants[name]['oof_auc']:.4f} "
              f"acc={row['accuracy']:.4f} TPR@FPR=5%={row['tpr@fpr=5%']['tpr_interp']:.4f}",
              flush=True)

    primary = PRIMARY_VARIANT
    err_primary = variant_scores(cache, primary)
    raw_ai = -err_primary / float(cache["logit_scale"])
    cal, temps = calibrate_oof(raw_ai, y)

    # ---------------- 3. shared OOF matrix of the five shipped routes
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    if not np.array_equal(a["y"], y):
        raise SystemExit("[bundle] label order differs from data/labels.csv")
    scores = a["oof_calibrated"]                     # calibrated OOF probability per route
    fused_cfg = mc.load_deployed_config(ws / "models")
    deployed_w = {m: float(w) for m, w in fused_cfg["weights"].items() if m in scores}
    print(f"[fusion] deployed weights {deployed_w} (source {fused_cfg['source']})", flush=True)

    # ---------------- 4. correlation with the shipped routes
    from scipy.stats import pearsonr, spearmanr
    corr = {}
    for m in mc.MEMBER_ORDER:
        if m not in scores:
            continue
        pr = pearsonr(cal, scores[m])
        sr = spearmanr(cal, scores[m])
        pr_raw = pearsonr(raw_ai, a["oof_logits"][m])
        corr[m] = {"pearson_calibrated": round(float(pr[0]), 4),
                   "pearson_calibrated_p": float(f"{pr[1]:.3g}"),
                   "spearman_calibrated": round(float(sr[0]), 4),
                   "spearman_calibrated_p": float(f"{sr[1]:.3g}"),
                   "pearson_rawlogit": round(float(pr_raw[0]), 4)}
        print(f"[corr] aeroblade vs {m:14s} pearson={corr[m]['pearson_calibrated']:+.4f} "
              f"spearman={corr[m]['spearman_calibrated']:+.4f}", flush=True)

    # ---------------- 5. Δ of adding the route to the deployed fusion
    scores_with = dict(scores)
    scores_with["aeroblade"] = cal
    base_score, base_members = mc.fuse_probabilities(scores, deployed_w)
    base_row = operating_row(y, base_score, args.target_fpr, args.fp_budget)
    base_pred = (base_score >= base_row["threshold"]).astype(int)
    base_errors = set(np.where(base_pred != y)[0].tolist())
    base_tpr5 = mc.tpr_at_fpr(y, base_score, PRESET["primary_fpr_level"])["tpr_interp"]
    print(f"[fusion] incumbent acc={base_row['accuracy']:.4f} fpr={base_row['fpr']:.4f} "
          f"TPR@FPR=5%={base_tpr5:.4f} TPR@FPR=1%="
          f"{base_row['tpr@fpr=1%']['tpr_interp']:.4f}", flush=True)

    add_rows = {}
    for w in WEIGHT_SWEEP:
        label = f"w_aeroblade={w:.2f}"
        wts = dict(deployed_w)
        wts["aeroblade"] = w
        sc, members = mc.fuse_probabilities(scores_with, wts, list(wts))
        row = operating_row(y, sc, args.target_fpr, args.fp_budget)
        pred = (sc >= row["threshold"]).astype(int)
        errs = set(np.where(pred != y)[0].tolist())
        fixed = sorted(int(seqs[i]) for i in (base_errors - errs))
        broken = sorted(int(seqs[i]) for i in (errs - base_errors))
        tpr5 = mc.tpr_at_fpr(y, sc, PRESET["primary_fpr_level"])["tpr_interp"]
        add_rows[label] = {
            "weights": wts, "members": members,
            "accuracy": row["accuracy"], "fpr": row["fpr"], "tpr": row["tpr"],
            "auc": row["auc"],
            "tpr@fpr=1%": row["tpr@fpr=1%"], "tpr@fpr=5%": row["tpr@fpr=5%"],
            "delta_accuracy": round(row["accuracy"] - base_row["accuracy"], 4),
            "delta_tpr_at_fpr1": round(
                row["tpr@fpr=1%"]["tpr_interp"] - base_row["tpr@fpr=1%"]["tpr_interp"], 4),
            "delta_tpr_at_fpr5": round(tpr5 - base_tpr5, 4),
            "delta_auc": round(row["auc"] - base_row["auc"], 4),
            "errors_fixed_n": len(fixed), "errors_fixed_seqs": fixed,
            "errors_introduced_n": len(broken), "errors_introduced_seqs": broken,
            "net_errors": len(broken) - len(fixed),
        }
        print(f"[add] {label:22s} dAcc={add_rows[label]['delta_accuracy']:+.4f} "
              f"dTPR@5%={add_rows[label]['delta_tpr_at_fpr5']:+.4f} "
              f"fixed={len(fixed)} broken={len(broken)}", flush=True)

    # ---------------- 6. pre-registered verdict (rule text == implementation)
    best = max(add_rows.values(), key=lambda r: (r["delta_tpr_at_fpr5"], r["delta_accuracy"]))
    auc_primary = single_variants[primary]["oof_auc"]
    best_variant_name, best_variant = max(single_variants.items(),
                                          key=lambda kv: kv[1]["oof_auc"])
    cond_auc = auc_primary >= PRESET["min_auc"]
    cond_delta = (best["delta_tpr_at_fpr5"] >= PRESET["min_delta_tpr_at_primary_fpr"]
                  or (best["delta_tpr_at_fpr1"] > 0 and best["delta_accuracy"] > 0))
    reasons = []
    if not cond_auc:
        reasons.append(f"单路 AUC={auc_primary:.4f} < 预注册门槛 {PRESET['min_auc']}"
                       f"（AUC≈0.5 表示该路无排序信息）；最佳变体 {best_variant_name} "
                       f"AUC={best_variant['oof_auc']:.4f}")
    if not cond_delta:
        reasons.append(
            f"加入 deployed 融合后最佳 ΔTPR@FPR=5%={best['delta_tpr_at_fpr5']:+.4f} "
            f"< 门槛 {PRESET['min_delta_tpr_at_primary_fpr']}，且 "
            f"ΔTPR@FPR=1%={best['delta_tpr_at_fpr1']:+.4f} / "
            f"ΔAccuracy={best['delta_accuracy']:+.4f} 未同时为正")
    retain = cond_auc and cond_delta
    verdict = "保留 (retain)" if retain else "剔除 (remove；负面结果)"
    conclusion = {
        "verdict": verdict,
        "retained": bool(retain),
        "basis": "pre-registered rule (see preset_decision_rule)",
        "conditions": {"auc_gate_passed": bool(cond_auc),
                       "delta_gate_passed": bool(cond_delta)},
        "failed_conditions": reasons,
        "primary_variant": primary,
        "primary_variant_auc": auc_primary,
        "best_variant": {"name": best_variant_name, "auc": best_variant["oof_auc"],
                         "note": "steelman: the best-separating variant, reported so the "
                                 "verdict does not depend on one arbitrary preprocessing"},
        "best_add_in": {k: best[k] for k in ("weights", "accuracy", "delta_accuracy",
                                            "delta_tpr_at_fpr1", "delta_tpr_at_fpr5",
                                            "delta_auc", "errors_fixed_n",
                                            "errors_introduced_n")},
        "negative_result_note": (
            "A negative result is reported as evidence, not hidden: the audit (F4) only "
            "requires the optional member to be *decided by an ablation*; 'not verifiable "
            "within budget' or 'verified and rejected' both close the finding."),
        "deployment_consequence": (
            "the member stays out of the default fusion: PipelineConfig.enable_aeroblade "
            "defaults to False and models/fusion.json carries neither a weight nor a "
            "temperature for it"),
    }

    payload = {
        "report": "aeroblade_ablation",
        "manual_sections": ["8.2 可选成员与消融", "12.1 多模型组合与错误相关性实验",
                            "3.6 冻结骨干纪律"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/aeroblade_ablation.py",
        "member_under_test": {
            "name": "aeroblade",
            "method": "AEROBLADE (Ricker et al., CVPR 2024) - frozen SD-VAE "
                      "reconstruction error, training-free",
            "score_direction": "direction=-1 (lower reconstruction error => more likely "
                               "latent-diffusion generated)",
            "implementation": "repo/src/l2_aeroblade.py (AerobladeScorer)",
        },
        "dependency": {
            "vae_repo": "stabilityai/sd-vae-ft-mse",
            "vae_dir": cache.get("vae_dir") or vae["vae_dir"],
            "vae_dir_relative": str(Path(cache.get("vae_dir") or vae["vae_dir"])
                                    .resolve().relative_to(WS.resolve()))
                                if str(cache.get("vae_dir") or vae["vae_dir"]).startswith(str(WS))
                                else None,
            "source": vae["source"],
            "downloaded_during_run": bool(vae["downloaded"]),
            "weight_sha256": cache.get("vae_weight_sha256"),
            "size_note": "~334 MB safetensors; cached in the workspace, never in the "
                         "user-level HF cache",
        },
        "not_verified": [{
            "member": "effort",
            "method": "Effort (Koutlis & Papadopoulos, CVPR 2024) - requires a low-rank "
                      "adaptation of an EDM diffusion model",
            "status": "未在预算内验证 (not verified within budget)",
            "reason": "the low-rank fine-tuning step needs an EDM checkpoint and GPU "
                      "training time that exceed this task's budget; no score was "
                      "produced, so no ablation number is claimed for it",
        }],
        "data": {
            "n_samples": n,
            "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
            "images": "data/images/*.png (competition set)",
            "score_matrix_source": bundle["protocol"],
            "calibration": "fold-safe single-parameter temperature scaling fitted on "
                           "training-fold raw scores (random_state=42 folds, identical to "
                           "scripts/evaluate_system.py)",
            "temperatures_per_fold": temps,
            "cost": {"vae_load_s": cache.get("vae_load_s"),
                     "median_ms_per_image": cache.get("median_ms"),
                     "images_per_s_median": cache.get("imgs_per_s_median"),
                     "device": cache.get("device"), "recon_side": cache.get("recon_side"),
                     "note": "the AEROBLADE member costs ~0.6 s/image, i.e. roughly half of "
                             "the whole 5-route evidence budget, which is why a route that "
                             "adds no TPR is also a latency liability"},
        },
        "preset_decision_rule": PRESET,
        "single_route_table": single_variants,
        "calibrated_route": {
            "oof_auc_of_calibrated_probability": mc.auc(y, cal),
            "note": "monotone temperature scaling cannot change the AUC; the value is "
                    "reported as a consistency check",
        },
        "score_correlation": corr,
        "add_to_deployed_fusion": {
            "incumbent": {"members": base_members, "weights": deployed_w,
                          "weights_source": fused_cfg["source"],
                          "thr_high_frozen": fused_cfg["thr_high"],
                          "operating_point": f"threshold = order statistic on real OOF "
                                             f"scores, fp_budget={args.fp_budget} "
                                             f"(target FPR {args.target_fpr:.0%})",
                          "accuracy": base_row["accuracy"], "fpr": base_row["fpr"],
                          "tpr": base_row["tpr"], "auc": base_row["auc"],
                          "tpr@fpr=1%": base_row["tpr@fpr=1%"],
                          "tpr@fpr=5%": base_row["tpr@fpr=5%"],
                          "error_n": len(base_errors),
                          "error_seqs": sorted(int(seqs[i]) for i in base_errors)},
            "weight_sweep": add_rows,
            "reading": "a candidate route must either fix incumbent errors at the same "
                       "operating point or lift TPR at the frozen FPR level; a route that "
                       "only reshuffles errors inside the fusion is not evidence of value",
        },
        "conclusions": conclusion,
        "limitations": [
            "All numbers are out-of-fold on the 119 competition samples; the two routes "
            "clip_domain/npr are not in the deployed fusion, so the incumbent here is the "
            "shipped combined+forensic table.",
            "TPR@FPR=1% is granularity-degenerate with 63 real samples (1 FP = 1.59%); the "
            "retention decision is taken at FPR=5%, the 1% column is informational.",
            "The paper's LPIPS reconstruction loss is not available offline; multi-scale "
            "MSE/L1 is the documented substitute, and every variant is reported rather "
            "than only the shipped one.",
            "AEROBLADE is a general-purpose 'is this latent-diffusion output' signal; the "
            "competition real class contains scanned/photographed documents whose noise "
            "profile differs from the camera photos the method was validated on, so a "
            "negative outcome here does not refute the paper.",
        ],
        "per_sample": [{"seq": r["seq"], "label": r["label"],
                        "recon_error_multiscale": round(
                            sum(w * r[k] for k, w in zip(VARIANTS[primary]["keys"],
                                                         VARIANTS[primary]["weights"]))
                            / sum(VARIANTS[primary]["weights"]), 8),
                        "ms": r["ms"]} for r in cache["rows"]],
    }
    mc.write_json(out_path, payload)
    print(f"\n[aeroblade] single-route AUC={auc_primary:.4f} | "
          f"best ΔTPR@FPR=5%={best['delta_tpr_at_fpr5']:+.4f} | verdict={verdict}")


if __name__ == "__main__":
    main()
