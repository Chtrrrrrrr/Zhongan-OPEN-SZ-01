#!/usr/bin/env python3
"""Train a frozen-CLIP linear probe for AI-generated image detection.

Data: AIGC-Detection-Benchmark parquet shards (125k images, 18 generators).
Backbone: openai/clip-vit-large-patch14 (frozen, image tower only).
Head: single linear layer -> BCEWithLogitsLoss.
Output: models/clip_probe.pt  (state dict + config + scaler)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import io

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from l2_clip import CLIPFeatureExtractor, LinearProbe  # noqa: E402


CACHE_DIR = Path(__file__).parent.parent / "models" / "hf_cache"


def stream_shard(path: Path, extractor, batch_size: int = 24, max_per_shard: int | None = None,
                 reencode_q: int = 0):
    """Stream a parquet shard: decode + extract features batch by batch.

    Keeps memory bounded (only raw parquet bytes + 1024-d features), instead of
    holding thousands of decoded RGB arrays.

    reencode_q > 0 applies a COMMON codec to every sample (decode -> JPEG(q) ->
    decode) so the probe cannot exploit the dataset shortcut "real=JPEG,
    fake=PNG" (measured: real 59% JPEG vs fake 94% PNG in the benchmark shards).
    """
    df = pd.read_parquet(path, columns=["image", "label"])
    n = len(df) if not max_per_shard else min(len(df), max_per_shard)
    # column-level numpy access: per-row df.iloc[i] is orders of magnitude slower
    raw_col = df["image"].to_numpy()
    lab_col = df["label"].to_numpy()
    feats, labels = [], []
    buf_imgs, buf_labs = [], []

    def flush():
        if not buf_imgs:
            return
        batch_bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in buf_imgs]
        feats.append(extractor.batch(batch_bgr, batch_size=len(batch_bgr)))
        labels.extend(buf_labs)
        buf_imgs.clear()
        buf_labs.clear()

    for i in tqdm(range(n), desc=f"extract {path.name}", mininterval=5):
        try:
            raw = raw_col[i]["bytes"]
        except Exception:
            continue
        lab = int(lab_col[i])
        try:
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            continue
        if reencode_q > 0:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(reencode_q)])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        buf_imgs.append(img)
        buf_labs.append(lab)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()
    return np.concatenate(feats, axis=0), np.array(labels, dtype=np.int64)


def train_one_split(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    device: str, epochs: int = 40, lr: float = 1e-2, weight_decay: float = 1e-3):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = LinearProbe(X_tr.shape[1]).to(device)
    ds_tr = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(torch.tensor(X_val_s, dtype=torch.float32).to(device)).cpu().numpy()
        auc = _auc(vlogits, y_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, scaler, best_auc


def _auc(logits: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, logits)


def resolve_clip_model(name_or_path: str) -> str:
    """Prefer a locally cached snapshot over a hub repo id (offline-friendly).

    Searches every plausible cache root: <repo>/models/hf_cache,
    <workspace>/models/hf_cache (repo's parent), and the HF default cache.
    """
    p = Path(name_or_path)
    if p.is_dir():
        return str(p)
    sub = "models--" + name_or_path.replace("/", "--")
    here = Path(__file__).resolve().parent          # repo/scripts
    roots = [
        here.parent / "models" / "hf_cache",        # repo/models/hf_cache
        here.parent.parent / "models" / "hf_cache",  # workspace/models/hf_cache
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        snaps = root / sub / "snapshots"
        if snaps.is_dir():
            for d in sorted(snaps.iterdir(), reverse=True):
                if (d / "config.json").is_file():
                    return str(d)
    return name_or_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=None,
                    help="parquet shard paths; default=data/external/test-*.parquet")
    ap.add_argument("--max-per-shard", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="models/clip_probe.pt")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--reencode-q", type=int, default=0,
                    help="if >0, apply this JPEG quality to ALL samples before feature "
                         "extraction (removes the real=JPEG / fake=PNG dataset shortcut)")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    shards = args.shards or sorted((Path(__file__).parent.parent / "data" / "external").glob("*.parquet"))
    if not shards:
        raise SystemExit("No parquet shards found. Put AIGC-Detection-Benchmark shards in data/external/")
    print(f"[data] {len(shards)} shards; common-codec re-encode: "
          f"{'JPEG q=' + str(args.reencode_q) if args.reencode_q > 0 else 'OFF'}", flush=True)

    extractor = CLIPFeatureExtractor(resolve_clip_model(args.clip_model),
                                     device=args.device, input_size=224)

    feat_list, label_list = [], []
    for p in shards:
        p = Path(p)
        print(f"[load] {p}", flush=True)
        f, labs = stream_shard(p, extractor, batch_size=24,
                               max_per_shard=args.max_per_shard,
                               reencode_q=args.reencode_q)
        feat_list.append(f)
        label_list.append(labs)
        print(f"[load] {p.name}: {f.shape[0]} samples", flush=True)
    feats = np.concatenate(feat_list, axis=0)
    all_labels = np.concatenate(label_list)
    print(f"[data] total {len(all_labels)} images; real={int((all_labels==0).sum())} "
          f"fake={int((all_labels==1).sum())}", flush=True)
    print("[feature] shape", feats.shape, flush=True)

    # Cross-validate
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(feats, all_labels)):
        _, _, auc = train_one_split(feats[tr_idx], all_labels[tr_idx],
                                    feats[val_idx], all_labels[val_idx],
                                    args.device, args.epochs, args.lr)
        print(f"[fold {fold + 1}/{args.n_folds}] val AUC = {auc:.4f}")
        aucs.append(auc)
    print(f"[cv] mean AUC = {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    # Final model on all data (for deployment)
    print("[final] training final probe on all data...")
    # Use last 10% as val for early stopping, but final weights from all data
    n = len(feats)
    idx = np.random.RandomState(2024).permutation(n)
    split = int(n * 0.9)
    tr_idx, val_idx = idx[:split], idx[split:]
    model, scaler, val_auc = train_one_split(feats[tr_idx], all_labels[tr_idx],
                                             feats[val_idx], all_labels[val_idx],
                                             args.device, args.epochs, args.lr)
    print(f"[final] hold-out AUC = {val_auc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_sd = model.state_dict()
    # flatten keys to match detector.py expectation: weight, bias
    flat_sd = {}
    for k, v in raw_sd.items():
        if k == "fc.weight":
            flat_sd["weight"] = v
        elif k == "fc.bias":
            flat_sd["bias"] = v
        else:
            flat_sd[k] = v
    torch.save({
        "state_dict": flat_sd,
        "config": {"in_dim": feats.shape[1], "clip_model": args.clip_model},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "cv_aucs": aucs,
        "val_auc": val_auc,
    }, out_path)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
