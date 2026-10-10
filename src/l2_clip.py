"""L2 Core member: frozen CLIP visual backbone + linear probe (Ojha et al., CVPR 2023).

The backbone is NEVER fine-tuned (frozen-backbone + lightweight head strategy,
manual ch.3.6); only a linear layer on the pooled feature is trained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def default_cache_dir() -> str:
    """Workspace-local HF cache so weights stay inside the reproducible package."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "models", "hf_cache")


class CLIPFeatureExtractor:
    """Frozen CLIP ViT feature extractor (image tower only)."""

    def __init__(self, model_dir: str, device: str | None = None, input_size: int = 224,
                 cache_dir: str | None = None):
        from transformers import CLIPVisionModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cd = cache_dir or default_cache_dir()
        try:
            self.model = CLIPVisionModel.from_pretrained(
                model_dir, cache_dir=cd).to(self.device).eval()
        except Exception:
            # offline fallback: use whatever is already in the cache
            self.model = CLIPVisionModel.from_pretrained(
                model_dir, cache_dir=cd, local_files_only=True).to(self.device).eval()
        self.input_size = input_size
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """bgr uint8 HxWx3 -> L2-normalized feature vector."""
        x = self.preprocess(bgr).unsqueeze(0).to(self.device)
        feat = self.model(x).pooler_output  # [1, D]
        feat = nn.functional.normalize(feat.float(), dim=-1)
        return feat[0].cpu().numpy()

    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def batch(self, bgr_list: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(bgr_list), batch_size):
            xs = torch.stack([self.preprocess(b) for b in bgr_list[i:i + batch_size]]).to(self.device)
            f = self.model(xs).pooler_output
            f = nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)
