"""L2 Optional member: AEROBLADE (Ricker et al., CVPR 2024) - training-free.

Latent-diffusion (LD) images live close to the SD-VAE manifold: encoded and
decoded by the frozen Stable-Diffusion autoencoder, their reconstruction error is
much lower than that of real photographs, which carry sensor noise, demosaicing
and camera ISP processing the VAE never learned to invert. No training data is
needed - only the frozen SD autoencoder.

Score convention (manual ch.9/12.1)
----------------------------------
``recon_error()`` returns the raw reconstruction error: LOWER => more likely
LD-generated, which is why the decision layer registers this member with
``direction=-1`` (see ``detector.detect_image``).

``logit()`` returns a monotone re-scaling of that error onto a logit-like axis
(negated, divided by a fixed a-priori reference scale) so the route can be
temperature-calibrated exactly like the other members. Any monotone re-scaling
leaves the AUC of the route unchanged; the scale only has to bring the values
into a range where the one-parameter temperature fit is numerically well posed.

Implementation notes
--------------------
* input: BGR uint8 -> RGB, resized to :data:`RECON_SIDE` (512; the task-specified
  operating size, the paper's 256 is available as ``side=256``), scaled to [-1, 1]
* error: multi-scale MSE between the image and its VAE round trip at 512/256/128
  with weights (0.5, 0.3, 0.2). Multi-scale is the offline stand-in for the
  paper's LPIPS (no LPIPS weights available in this environment); the coarse
  scales keep the signal from being dominated by resampling/JPEG noise, and
  ``recon_error_breakdown()`` exposes every scale so the ablation report can
  show which one carries the separation.
* the autoencoder is frozen and never trained on the competition data
  (manual ch.3.6 frozen-backbone discipline).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# fixed a-priori reference scale that maps a typical reconstruction error
# (~0.01-0.05 MSE) onto a logit-like range (~-1 .. -5). Chosen once, from the
# definition of the loss, and never fitted on the 119 samples.
LOGIT_SCALE = 1e-2


class AerobladeScorer:
    """Frozen SD-VAE reconstruction-error scorer (training-free member)."""

    RECON_SIDE = 512                     # task-specified operating size
    SCALES = ((512, 0.5), (256, 0.3), (128, 0.2))   # (side, weight) for multi-scale MSE
    VAE_REPO = "stabilityai/sd-vae-ft-mse"

    def __init__(self, vae_dir: str, device: str | None = None, side: int | None = None):
        from diffusers import AutoencoderKL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_dir = str(vae_dir)
        self.side = int(side or self.RECON_SIDE)
        self.vae = AutoencoderKL.from_pretrained(self.vae_dir).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 HxWx3 -> float tensor [3, side, side] in [-1, 1]."""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.side, self.side), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
        return x * 2.0 - 1.0

    @torch.no_grad()
    def reconstruct(self, bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(input, reconstruction) tensors at ``side`` resolution, both in [-1, 1]."""
        x = self.preprocess(bgr).unsqueeze(0).to(self.device)
        lat = self.vae.encode(x).latent_dist.mode()
        rec = self.vae.decode(lat).sample
        return x, rec.clamp(-1.0, 1.0)

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def recon_error_breakdown(self, bgr: np.ndarray) -> dict:
        """Multi-scale MSE per scale + the weighted total (lower => more LD-like)."""
        x, rec = self.reconstruct(bgr)
        per_scale, total, wsum = {}, 0.0, 0.0
        for s, w in self.SCALES:
            if s > self.side:
                continue
            if s == self.side:
                a, b = x, rec
            else:
                a = F.interpolate(x, size=(s, s), mode="area")
                b = F.interpolate(rec, size=(s, s), mode="area")
            e = float(F.mse_loss(b, a).item())
            per_scale[f"mse@{s}"] = e
            total += w * e
            wsum += w
        per_scale["recon_error"] = total / (wsum or 1.0)
        per_scale["side"] = self.side
        return per_scale

    @torch.no_grad()
    def recon_error(self, bgr: np.ndarray) -> float:
        """Raw reconstruction error. LOWER => more likely latent-diffusion generated."""
        return float(self.recon_error_breakdown(bgr)["recon_error"])

    @torch.no_grad()
    def logit(self, bgr: np.ndarray) -> float:
        """AI-side logit: negated, fixed-scale reconstruction error (monotone in -error)."""
        return -self.recon_error(bgr) / LOGIT_SCALE
