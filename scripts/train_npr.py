#!/usr/bin/env python3
"""Train the NPR (Neighboring Pixel Relationships) detector from scratch.

Backbone: ResNet-18 whose first conv consumes the 4-direction NPR relation map
(12 channels) instead of RGB. Trained on AIGC-Detection-Benchmark shards.

Reference: Tan et al., CVPR 2024 (github.com/chuangchuangtan/NPR-DeepfakeDetection).
We re-implement the front-end from the paper and train our own weights; no
original checkpoints are used.

Output: models/npr.pt  (raw state_dict, loadable by src/l2_npr.py)
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from l2_npr import NPRNet  # noqa: E402


class ShardDataset(Dataset):
    """Streams decoded images from one or more parquet shards (no full preload).

    Columns are converted to numpy arrays once; per-row pandas indexing on a
    multi-hundred-MB frame is two orders of magnitude slower.
    """

    def __init__(self, shard_paths: list[Path], input_size: int = 256, max_per_shard: int | None = None):
        self.raws, self.labs, self.sizes = [], [], []
        for p in shard_paths:
            df = pd.read_parquet(p, columns=["image", "label"])
            n = len(df) if not max_per_shard else min(len(df), max_per_shard)
            self.raws.append(df["image"].to_numpy()[:n])
            self.labs.append(df["label"].to_numpy(dtype=np.int64)[:n])
            self.sizes.append(n)
        self.input_size = input_size

    def __len__(self):
        return sum(self.sizes)

    @property
    def labels(self) -> np.ndarray:
        """Labels straight from the parquet columns - no image decoding."""
        return np.concatenate(self.labs)

    def _locate(self, idx):
        for k, n in enumerate(self.sizes):
            if idx < n:
                return k, idx
            idx -= n
        raise IndexError(idx)

    def __getitem__(self, idx):
        k, i = self._locate(idx)
        img = np.array(Image.open(io.BytesIO(self.raws[k][i]["bytes"])).convert("RGB"))
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(float(self.labs[k][i]))
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--max-per-shard", type=int, default=None)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/npr.pt")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    shards = args.shards or sorted((Path(__file__).resolve().parent.parent.parent /
                                    "data" / "external").glob("*.parquet"))
    if not shards:
        raise SystemExit("no parquet shards found")
    print(f"[data] {len(shards)} shards", flush=True)

    ds = ShardDataset([Path(p) for p in shards], args.input_size, args.max_per_shard)
    n = len(ds)
    idx = np.arange(n)
    labels = ds.labels
    tr_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=42, stratify=labels)
    print(f"[data] total={n} train={len(tr_idx)} val={len(val_idx)}", flush=True)

    tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=args.batch_size,
                           shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    device = args.device
    net = NPRNet(pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        net.train()
        tot, seen = 0.0, 0
        for xb, yb in tqdm(tr_loader, desc=f"epoch {ep+1}/{args.epochs}", mininterval=10):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            seen += len(yb)
        sched.step()
        net.eval()
        outs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                outs.append(net(xb.to(device)).cpu().numpy())
                ys.append(yb.numpy())
        outs = np.concatenate(outs)
        ys = np.concatenate(ys)
        auc = roc_auc_score(ys, outs)
        acc = float(((outs >= 0) == (ys >= 0.5)).mean())
        print(f"[epoch {ep+1}] loss={tot/max(1,seen):.4f} val_auc={auc:.4f} val_acc={acc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(net.state_dict(), out_path)
            print(f"[save] {out_path} (val_auc={auc:.4f})", flush=True)

    print(f"[done] best val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
