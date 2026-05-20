#!/usr/bin/env python3
"""RoI crop/paste utilities for cnnROI mask refinement.

All spatial operations are defined relative to the 1024×1024 mask grid.
The embedding grid (64×64) and GT grid (448×448) are addressed by
projecting bbox coordinates with scale_bbox().
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def extract_bbox(
    mask: np.ndarray,
    pad: int = 64,
) -> tuple[int, int, int, int] | None:
    """Bounding box of non-zero pixels in a binary mask, with padding.

    Args:
        mask: (H, W) binary array at 1024×1024 scale.
        pad:  padding in pixels at mask scale.

    Returns:
        (y0, y1, x0, x1) clipped to image bounds, or None if mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def scale_bbox(
    bbox: tuple[int, int, int, int],
    from_size: int,
    to_size: int,
) -> tuple[int, int, int, int]:
    """Project bbox coordinates from one square grid to another.

    Args:
        bbox:      (y0, y1, x0, x1) in from_size coordinate space.
        from_size: source grid side length (e.g. 1024).
        to_size:   target grid side length (e.g. 64 or 448).

    Returns:
        (y0, y1, x0, x1) in to_size space, clamped to [0, to_size].
    """
    y0, y1, x0, x1 = bbox
    s = to_size / from_size
    return (
        max(0, int(y0 * s)),
        min(to_size, int(y1 * s) + 1),
        max(0, int(x0 * s)),
        min(to_size, int(x1 * s) + 1),
    )


def crop_to_roi(
    tensor: torch.Tensor,
    bbox: tuple[int, int, int, int],
    roi_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Crop a (C, H, W) tensor to bbox and resize to (C, roi_size, roi_size).

    Args:
        tensor:   (C, H, W) tensor.
        bbox:     (y0, y1, x0, x1) in tensor's coordinate space.
        roi_size: output spatial size.
        mode:     "bilinear" for embedding/GT, "nearest" for binary masks.

    Returns:
        (C, roi_size, roi_size) tensor.
    """
    y0, y1, x0, x1 = bbox
    # Guard against degenerate crops when bbox projection collapses to a point
    y1 = max(y1, y0 + 1)
    x1 = max(x1, x0 + 1)
    crop = tensor[:, y0:y1, x0:x1].unsqueeze(0)            # (1, C, h, w)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(
        crop, size=(roi_size, roi_size), mode=mode, **kwargs
    ).squeeze(0)                                             # (C, roi_size, roi_size)


def paste_into_canvas(
    pred_crop: torch.Tensor,
    bbox: tuple[int, int, int, int],
    canvas_size: int = 1024,
) -> torch.Tensor:
    """Upsample a crop to bbox dimensions and place it on a blank canvas.

    Args:
        pred_crop:   (1, roi_size, roi_size) prediction tensor.
        bbox:        (y0, y1, x0, x1) in canvas coordinates (1024 scale).
        canvas_size: side length of the output canvas.

    Returns:
        (1, canvas_size, canvas_size) float tensor, zeros outside bbox.
    """
    y0, y1, x0, x1 = bbox
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    resized = F.interpolate(
        pred_crop.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)                                             # (1, h, w)
    canvas = torch.zeros(
        1, canvas_size, canvas_size,
        dtype=pred_crop.dtype, device=pred_crop.device,
    )
    canvas[:, y0:y1, x0:x1] = resized
    return canvas
