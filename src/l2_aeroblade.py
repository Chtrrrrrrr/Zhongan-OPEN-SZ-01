"""L2 Optional member: AEROBLADE (Ricker et al., CVPR 2024) - training-free.

Latent-diffusion images live close to the SD-VAE manifold: their VAE
reconstruction error is much lower than that of real photographs. Needs no
training data - only the frozen SD autoencoder.

Score returned = reconstruction error (higher => more likely REAL).
The decision layer inverts/standardizes it into an "AI score".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class AerobladeScorer:
    INPUT = 256  # paper evaluates at 256; 512 also possible (VRAM permitting)

    def __init__(self, vae_dir: str, device: str | None = None):
        from diffusers import AutoencoderKL  # optional dep
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(vae_dir).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.INPUT, self.INPUT), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        return x * 2.0 - 1.0  # [-1, 1]

    @torch.no_grad()
    def recon_error(self, bgr: np.ndarray) -> float:
        x = self.preprocess(bgr).unsqueeze(0).to(self.device)
        lat = self.vae.encode(x).latent_dist.mode()
        rec = self.vae.decode(lat).sample
        return float(F.l1_loss(rec, x).item())
