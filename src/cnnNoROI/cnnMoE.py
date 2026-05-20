#!/usr/bin/env python3
"""CNN Expert model for MoE mask refinement (Alt 3: no RoI crops).

Input:  (B, c_in, H, W)  — MedSAM embedding + resized binary mask + SDF
Output: (B, 1,    H, W)  — refined mask probabilities in [0, 1]

Two instances of CNNExpert are trained independently, one per expert (small/large).
"""
import torch
import torch.nn as nn


class CNNExpert(nn.Module):
    """Lightweight spatial decoder for per-fragment mask refinement.

    Three conv layers act as a spatial decoder, not a feature extractor —
    MedSAM already extracted features. No backbone needed.

    Args:
        c_in: number of input channels. Default 258 = 256 (embedding)
              + 1 (binary mask) + 1 (SDF).
    """

    def __init__(self, c_in: int = 258):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
