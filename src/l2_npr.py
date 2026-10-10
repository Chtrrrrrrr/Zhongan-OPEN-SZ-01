"""L2 Core member: NPR - Neighboring Pixel Relationships (Tan et al., CVPR 2024).

Captures up-sampling artifacts left by CNN-based generators: the strong local
correlation pattern between a pixel and its neighbors. We implement the NPR
front-end (4 relation maps from 3x3 neighborhoods) feeding a ResNet backbone.

Reference: github.com/chuangchuangtan/NPR-DeepfakeDetection (code re-implemented
from the paper; no original weights are used - trained from scratch on our data).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NPRLayer(nn.Module):
    """Compute 4 neighbor-relation maps (Eq.1-2 in the NPR paper).

    For pixel x(i,j), relation r_d = x(i,j) - x(i+di, j+dj) for the four
    directions d in {(0,1), (1,1), (1,0), (1,-1)} (right, down-right, down, down-left).
    Output: 4-channel map, same spatial size (zero-padded).
    """

    OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [0,1]
        outs = []
        for di, dj in self.OFFSETS:
            shifted = F.pad(x, (dj, -dj if dj else 0, di, -di if di else 0)) if False else None
            # simpler: roll-free slicing with zero pad
            p = F.pad(x, (1, 1, 1, 1))
            h, w = x.shape[2], x.shape[3]
            nb = p[:, :, 1 + di:1 + di + h, 1 + dj:1 + dj + w]
            outs.append(x - nb)
        return torch.cat(outs, dim=1)  # [B, 4*3? no: 4*C] -> here C=3 => 12


class NPRNet(nn.Module):
    """NPR front-end + ResNet-18 backbone + binary head."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.npr = NPRLayer()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rel = self.npr(x)
        return self.backbone(rel).squeeze(-1)


class NPRFeatureExtractor:
    """Inference wrapper mirroring CLIPFeatureExtractor's interface."""

    INPUT = 256

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = NPRNet().to(self.device).eval()
        sd = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(sd)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def preprocess(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.INPUT, self.INPUT), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(x).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def logit(self, bgr) -> float:
        x = self.preprocess(bgr).unsqueeze(0).to(self.device)
        return float(self.net(x).item())
