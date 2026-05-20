#!/usr/bin/env python3
"""SDF transform utilities for MoE dataset loading.

Input:
    MedSAM binary mask, shape (H, W) or (1, H, W), uint8/bool-like.

Output:
    SDF channel with the same shape as the input, float32, range [-1, 1].

SDF convention:
    negative inside the object, positive outside, zero near the boundary.

Formula:
    sdf = dist_transform(~mask) - dist_transform(mask)
    sdf_norm = sdf / max(abs(sdf))
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import torch


def mask_to_sdf(mask: np.ndarray) -> np.ndarray:
    """Convert one binary mask to a per-instance normalized SDF.

    Args:
        mask: Binary mask with shape (H, W) or a single-channel mask with shape
            (1, H, W). Non-zero values are treated as foreground.

    Returns:
        float32 SDF with the same shape as ``mask`` and values in [-1, 1].
        Values are negative inside the foreground object and positive outside.
    """
    mask_array = np.asarray(mask)
    squeeze_channel = False

    if mask_array.ndim == 3:
        if mask_array.shape[0] != 1:
            raise ValueError(
                f"expected single-channel mask shape (1, H, W), got {mask_array.shape}"
            )
        mask_2d = mask_array[0]
        squeeze_channel = True
    elif mask_array.ndim == 2:
        mask_2d = mask_array
    else:
        raise ValueError(f"expected mask shape (H, W) or (1, H, W), got {mask_array.shape}")

    mask_bool = mask_2d.astype(bool)

    outside = distance_transform_edt(~mask_bool)
    inside = distance_transform_edt(mask_bool)
    sdf = outside - inside

    max_abs = float(np.max(np.abs(sdf)))
    if max_abs > 0.0:
        sdf = sdf / max_abs

    sdf = sdf.astype(np.float32, copy=False)
    if squeeze_channel:
        sdf = sdf[np.newaxis, ...]

    return sdf


def sdf_channel_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return an SDF channel for Dataset code that expects shape (1, H, W).

    A 2D input mask is promoted to (1, H, W). A single-channel input mask keeps
    its shape.
    """
    sdf = mask_to_sdf(mask)
    if sdf.ndim == 2:
        sdf = sdf[np.newaxis, ...]
    return sdf

def mask_to_flowsdf(mask: torch.Tensor, threshold: float = 15.0) -> torch.Tensor:
    """Convert a binary mask to FlowSDF notebook-style signed distance field.

    FlowSDF convention:
        negative inside foreground, positive outside foreground.
        distances are clipped to [-threshold, threshold] and divided by threshold.

    Args:
        mask: Binary mask tensor with shape (H, W), values 0/1.
        threshold: Distance clipping threshold in pixels.

    Returns:
        SDF tensor with shape (H, W), values in [-1, 1].
    """
    mask_np = mask.detach().cpu().numpy().astype(bool)

    boundary = np.abs(binary_erosion(mask_np).astype(np.float32) - mask_np.astype(np.float32))
    distance = distance_transform_edt(boundary == 0)

    sdf = np.where(mask_np, -distance, distance).astype(np.float32)
    sdf = np.clip(sdf, -threshold, threshold) / threshold

    return torch.from_numpy(sdf).to(mask.device, dtype=torch.float32)

__all__ = ["mask_to_sdf", "sdf_channel_from_mask", "mask_to_flowsdf"]
