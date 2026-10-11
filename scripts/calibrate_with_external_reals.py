#!/usr/bin/env python3
"""R2 remedy: enlarge the pure-real calibration set with external real claim photos.

Why this exists
---------------
The 119 competition samples are ~47:53 and their "real" class is, by the manual's own
audit (ch.2.3), "almost entirely phone photos of paper documents" with only ~2 scene
photos. The decision boundary therefore encodes *camera photo of paper*, not *camera
photo*. Measured consequence (reports/external_holdout.json): on 41 real, network-
collected CLAIM photos (car damage, flooding, food, injury) the frozen threshold flags
**41/41 = 100%** - with the archived 3x3 configuration too, so it is not a regression
introduced here.

The manual's risk R2 mitigation is explicit: "扩充纯真实校准集，重点是数字原生的真实单据
…对单据类材料单独校准阈值". This script implements the calibration half of that remedy:

  * the decision threshold is re-derived as an order statistic over a LARGER pure-real
    pool (competition reals + external real claim photos), still targeting the same
    false-positive budget;
  * nothing is retrained, no head weight changes, and the reported in-domain out-of-fold
    metrics are recomputed with the same nested-CV protocol so the cost of the stricter
    threshold is visible.

A threshold can never separate two real populations that both score 0.97, so the script
also reports whether the pool is separable at all - if the external reals sit above the
competition fakes, no threshold can satisfy both and the honest output is that sentence.

Usage
  python scripts/calibrate_with_external_reals.py --external-reals <dir> [--dirs a,b]
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
sys.path.insert(0, str(ROOT / "scripts"))

from tile_clip_sweep import (_fit_head, _logits, _fit_temperature, _fused,  # noqa: E402
                            _fpr_threshold)

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def score_folder(paths, blocks_dir, cache, device, patch_max_side):
    """Fused OOF-scale logits for external images under the frozen heads.

    The heads are refit here on all 119 competition samples (they ship that way), so the
    scores are directly comparable with the deployed artefact; they are NOT used to
    report in-domain accuracy (that stays out-of-fold in deployment_metrics.json).
    """
    import joblib
    from l0_input import parse_image
    from l1_forensics import extract_l1_features, feature_vector
    from l1b_domain import extract_l1b_features, feature_vector_b
    from l2_clip import CLIPFeatureExtractor
    from l3_localization import patch_token_features
    from feature_spec import FeatureSpec
    from tile_clip_sweep import resolve_clip

    spec = FeatureSpec.load(blocks_dir / "feature_spec.json")
    fpack = joblib.load(blocks_dir / "forensic_lr.joblib")
    cpack = joblib.load(blocks_dir / "combined_lr.joblib")
    fusion = json.loads((blocks_dir / "fusion.json").read_text(encoding="utf-8"))
    temps = fusion["temperatures"]
    w = fusion["weights"]
    names = [n for n in w if w[n] > 0]
    wn = np.array([w[n] for n in names], dtype=np.float64)
    wn = wn / wn.sum()

    kwargs = {}
    if cache and Path(cache).exists():
        np.save  # noqa: B018 - placeholder to keep the import list explicit
    ext = CLIPFeatureExtractor(resolve_clip(""), device=device, input_size=224)
    out = []
    for p in paths:
        parsed = parse_image(str(p), keep_full=True)
        if not parsed.ok:
            continue
        v1 = feature_vector(extract_l1_features(parsed))
        vb = feature_vector_b(extract_l1b_features(parsed))
        cf = ext(parsed.bgr)
        bgr_full = parsed.bgr_full if parsed.bgr_full is not None else parsed.bgr
        pt = patch_token_features(ext, bgr_full, max_side=patch_max_side)
        tf = spec.aggregate_detail(spec.tile_detail(ext, bgr_full))
        zf = _logits(fpack["scaler"], fpack["clf"],
                     spec.forensic_vector(v1, vb, pt).reshape(1, -1))[0]
        zc = _logits(cpack["scaler"], cpack["clf"],
                     spec.combined_vector(v1, vb, cf, tf, pt).reshape(1, -1))[0]
        out.append((p.name, float(zf), float(zc)))
    return out, temps, w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external-reals", required=True,
                    help="folder (or comma-separated folders) of real claim photos")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default="reports/external_real_calibration.json")
    args = ap.parse_args()

    md = ROOT / args.models_dir
    fusion = json.loads((md / "fusion.json").read_text(encoding="utf-8"))
    temps = fusion["temperatures"]
    weights = fusion["weights"]
    spec_patch_side = int(fusion.get("feature_spec", {}).get("patch_max_side", 1600))

    folders = [Path(s.strip()) for s in args.external_reals.split(",") if s.strip()]
    files = []
    for folder in folders:
        files += [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in EXTS]
    print(f"[external] {len(files)} real photos from {[str(f) for f in folders]}")

    t0 = time.perf_counter()
    scored, _, _ = score_folder(files, md, None, args.device, spec_patch_side)
    print(f"[external] scored {len(scored)} in {time.perf_counter()-t0:.0f}s")

    # competition out-of-fold scores under the frozen weights/temperatures
    X1 = np.loadtxt(ROOT / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(md / "l1b_feats.npy")
    Fg = np.load(md / "clip_feats.npy")
    from feature_spec import FeatureSpec
    spec = FeatureSpec.load(md / "feature_spec.json")
    Xf = np.concatenate([X1, Xb] + ([np.load(md / "clip_patch.npy")]
                                    if spec.patch_tokens else []), axis=1)
    tiles = np.load(md / ("clip_tiles_spec.npy" if (md / "clip_tiles_spec.npy").exists()
                          else "clip_tiles.npy"))
    Xc = np.concatenate([Xf, Fg, tiles], axis=1)
    y = []
    with open(Path(args.data_dir) / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)

    from sklearn.model_selection import StratifiedKFold
    members = [m for m in ("combined", "forensic") if weights.get(m, 0.0) > 0]
    blocks = {"forensic": Xf, "combined": Xc}
    oof = np.zeros((args.rounds, len(y)))
    for r in range(args.rounds):
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42 + r * 13)
        for tr, va in skf.split(Xc, y):
            heads = {m: _fit_head(blocks[m][tr], y[tr]) for m in members}
            z = {m: _logits(*heads[m], blocks[m][va]) for m in members}
            oof[r, va] = _fused(z, temps, weights)

    ext_real = np.array([fused_pair(zf, zc, temps, weights) for _, zf, zc in scored])
    # Round-average the out-of-fold scores before taking the order statistic. Pooling the
    # raw per-round scores would multiply the real class by the round count and make the
    # budget mean "2 false positives across ALL rounds" instead of per evaluation batch.
    comp_real = oof[:, y == 0].mean(axis=0)
    comp_fake = oof[:, y == 1].mean(axis=0)

    def summarize(pool_real):
        """Threshold = order statistic over the pooled real scores."""
        flat = np.asarray(pool_real, dtype=np.float64).ravel()
        s = np.sort(flat)[::-1]
        k = max(0, min(args.fp_budget, len(s) - 1))
        thr = float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)
        per_round = []
        for r in range(args.rounds):
            pred = (oof[r] >= thr).astype(int)
            per_round.append({
                "accuracy": round(float((pred == y).mean()), 4),
                "fpr": round(float(pred[y == 0].mean()), 4),
                "tpr": round(float(pred[y == 1].mean()), 4)})
        return {
            "pool_real_n": int(len(flat)),
            "threshold": round(thr, 6),
            "competition_oof": {
                "accuracy_mean": round(float(np.mean([p["accuracy"] for p in per_round])), 4),
                "accuracy_min": round(float(np.min([p["accuracy"] for p in per_round])), 4),
                "fpr_mean": round(float(np.mean([p["fpr"] for p in per_round])), 4),
                "tpr_mean": round(float(np.mean([p["tpr"] for p in per_round])), 4),
            },
            "external_real_flag_rate": round(float((ext_real >= thr).mean()), 4),
            "external_real_n": int(len(ext_real)),
            "per_round": per_round,
        }

    before = summarize(comp_real)
    after = summarize(np.concatenate([comp_real, ext_real]))
    # separability: can ANY threshold satisfy both populations?
    ext_max = float(np.percentile(ext_real, 5))
    fake_min = float(np.percentile(comp_fake, 5))
    payload = {
        "report": "external_real_calibration",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motivation": ("the competition 'real' class is ~phone photos of paper; real claim "
                       "SCENE photos are out of distribution and were flagged 100% before "
                       "this remedy (manual ch.2.3 + risk R2)"),
        "external_folders": [str(f) for f in folders],
        "threshold_rule": f"order statistic over the pooled real scores, fp_budget={args.fp_budget}",
        "before_pooled_calibration": before,
        "after_pooled_calibration": after,
        "separability": {
            "external_real_p05": round(ext_max, 6),
            "competition_fake_p05": round(fake_min, 6),
            "overlap": bool(ext_max > fake_min),
            "note": ("overlap means real claim photos and AI-generated material occupy the "
                     "same score band, so no single threshold can satisfy both; the report "
                     "then has to choose which error it prefers"),
        },
        "external_real_scores": [{"file": n, "fused": round(fused_pair(zf, zc, temps, weights), 6)}
                                 for n, zf, zc in scored],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[before] thr={before['threshold']:.4f} official oof acc={before['competition_oof']['accuracy_mean']} "
          f"external real flagged={before['external_real_flag_rate']:.1%}")
    print(f"[after ] thr={after['threshold']:.4f} official oof acc={after['competition_oof']['accuracy_mean']} "
          f"external real flagged={after['external_real_flag_rate']:.1%}")
    print(f"[separability] external real p05={ext_max:.3f} vs competition fake p05={fake_min:.3f} "
          f"-> overlap={payload['separability']['overlap']}")
    print(f"[save] {out}")
    return 0


def fused_pair(zf, zc, temps, weights):
    names = [n for n in weights if weights[n] > 0]
    ps = np.array([1 / (1 + np.exp(-(zf if n == "forensic" else zc) / temps[n])) for n in names])
    wn = np.array([weights[n] for n in names], dtype=np.float64)
    wn = wn / wn.sum()
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    L = np.log(ps / (1 - ps))
    return float(1 / (1 + np.exp(-float((L * wn).sum()))))


if __name__ == "__main__":
    sys.exit(main())
