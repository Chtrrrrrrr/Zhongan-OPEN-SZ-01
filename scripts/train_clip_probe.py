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


def load_shard(path: Path, max_per_shard: int | None = None):
    df = pd.read_parquet(path)
    images, labels = [], []
    for _, row in df.iterrows():
        try:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            images.append(np.array(img))
            labels.append(int(row["label"]))
        except Exception:
            continue
        if max_per_shard and len(images) >= max_per_shard:
            break
    return images, np.array(labels, dtype=np.int64)


def extract_features(extractor, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(images), batch_size), desc="extract"):
        batch = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images[i:i + batch_size]]
        f = extractor.batch(batch, batch_size=batch_size)
        feats.append(f)
    return np.concatenate(feats, axis=0)


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
    args = ap.parse_args()

    global cv2, io
    import cv2
    import io

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    shards = args.shards or sorted((Path(__file__).parent.parent / "data" / "external").glob("*.parquet"))
    if not shards:
        raise SystemExit("No parquet shards found. Put AIGC-Detection-Benchmark shards in data/external/")
    print(f"[data] {len(shards)} shards")

    extractor = CLIPFeatureExtractor(args.clip_model, device=args.device, input_size=224)

    all_images, all_labels = [], []
    for p in shards:
        print(f"[load] {p}")
        imgs, labs = load_shard(Path(p), args.max_per_shard)
        all_images.extend(imgs)
        all_labels.append(labs)
    all_labels = np.concatenate(all_labels)
    print(f"[data] total {len(all_images)} images; real={sum(all_labels==0)} fake={sum(all_labels==1)}")

    print("[feature] extracting CLIP features...")
    feats = extract_features(extractor, all_images, batch_size=24)
    print("[feature] shape", feats.shape)

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
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"in_dim": feats.shape[1], "clip_model": args.clip_model},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "cv_aucs": aucs,
        "val_auc": val_auc,
    }, out_path)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
