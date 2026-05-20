#!/usr/bin/env python3
"""Shared utilities for PENGWIN MedSAM/MoE dataloading and evaluation.

This module contains data/path/GT-decoding helpers that should be reused by:
- evaluation/evaluate_medsam_pengwin.py
- src/moe_dataset.py
- future MoE training/inference code

Important:
PENGWIN X-ray GT labels are stored as bit-packed .tif files.
The matching is metadata-based:

    pred_masks[i] <-> fragments[i] <-> decoded GT fragment from fragments[i]

No overlap-based matching is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


CATEGORY_NAMES = {
    1: "SA",
    2: "LI",
    3: "RI",
}


def resolve_existing_path(path: str | Path) -> Path:
    """Resolve common Snellius path variants.

    Metadata sometimes stores paths as /gpfs/home5/..., while interactive shells
    may show /home/... . This helper tries both.
    """
    path = Path(path)

    if path.exists():
        return path

    text = str(path)

    if text.startswith("/gpfs/home5/"):
        alt = Path(text.replace("/gpfs/home5/", "/home/", 1))
        if alt.exists():
            return alt

    if text.startswith("/home/"):
        alt = Path(text.replace("/home/", "/gpfs/home5/", 1))
        if alt.exists():
            return alt

    return path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dict records."""
    path = resolve_existing_path(path)

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def load_box_metadata_map(path: str | Path) -> dict[str, dict]:
    """Load bounding-box metadata as sample_name -> record.

    The bounding-box metadata is useful because it contains original_label_path,
    while MedSAM prediction metadata may not.
    """
    path = resolve_existing_path(path)

    if not path.exists():
        return {}

    return {record["sample_name"]: record for record in load_jsonl(path)}


def infer_label_path_from_image_path(image_path: str | None) -> Path | None:
    """Infer GT label path from raw X-ray image path.

    Example:
        train/input/images/x-ray/001_0000.tif
    becomes:
        train/output/images/x-ray/001_0000.tif
    """
    if not image_path:
        return None

    text = str(image_path)

    marker = "/train/input/images/x-ray/"
    if marker not in text:
        return None

    return Path(text.replace(marker, "/train/output/images/x-ray/"))


def get_label_path(
    record: dict,
    box_metadata: dict[str, dict] | None = None,
) -> Path | None:
    """Return original_label_path for a MedSAM/MoE prediction record.

    Priority:
    1. record["original_label_path"], if present
    2. box_metadata[sample_name]["original_label_path"]
    3. infer from record["original_image_path"]
    """
    box_metadata = box_metadata or {}

    label_path = record.get("original_label_path")
    if label_path:
        return resolve_existing_path(label_path)

    sample_name = record.get("sample_name")
    if sample_name in box_metadata:
        label_path = box_metadata[sample_name].get("original_label_path")
        if label_path:
            return resolve_existing_path(label_path)

    inferred = infer_label_path_from_image_path(record.get("original_image_path"))
    if inferred is not None:
        return resolve_existing_path(inferred)

    return None


def load_pengwin_label(label_path: str | Path) -> np.ndarray:
    """Load a PENGWIN X-ray GT label .tif as uint32.

    This is not the raw X-ray image. It is the bit-packed segmentation label.
    """
    label_path = resolve_existing_path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"missing GT label file: {label_path}")

    return np.array(Image.open(label_path)).astype(np.uint32, copy=False)


def get_category_name(fragment: dict) -> str:
    """Return category name from fragment metadata."""
    category_id = int(fragment["category_id"])
    return fragment.get("category_name", CATEGORY_NAMES.get(category_id, "unknown"))


def pengwin_bit_shift(category_id: int, fragment_id: int) -> int:
    """Return the bit shift used to decode a PENGWIN fragment.

    Project convention currently used by the box-prep/evaluation pipeline:

        shift = 10 * (category_id - 1) + fragment_id

    category_id:
        1 = SA
        2 = LI
        3 = RI

    fragment_id:
        1-based fragment id within that category.
    """
    return 10 * (int(category_id) - 1) + int(fragment_id)


def decode_pengwin_fragment_from_array(
    segmentation: np.ndarray,
    category_id: int,
    fragment_id: int,
) -> np.ndarray:
    """Decode one binary GT fragment from a bit-packed PENGWIN label array.

    Returns:
        uint8 mask with shape equal to the original GT label shape.
    """
    shift = pengwin_bit_shift(category_id, fragment_id)
    return ((segmentation >> shift) & 1).astype(np.uint8)


def decode_pengwin_fragment_from_record(
    segmentation: np.ndarray,
    fragment: dict,
) -> np.ndarray:
    """Decode one binary GT fragment using a fragment metadata record."""
    return decode_pengwin_fragment_from_array(
        segmentation=segmentation,
        category_id=int(fragment["category_id"]),
        fragment_id=int(fragment["fragment_id"]),
    )


def load_prediction_masks(mask_path: str | Path) -> np.ndarray:
    """Load predicted masks from .npz or .npy.

    Expected MedSAM/MoE format:
        .npz with key "masks"
        shape: (N, H, W)
    """
    mask_path = resolve_existing_path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"missing predicted mask file: {mask_path}")

    loaded = np.load(mask_path)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "masks" in loaded.files:
                masks = loaded["masks"]
            else:
                masks = loaded[loaded.files[0]]
        finally:
            loaded.close()
    else:
        masks = loaded

    if masks.ndim != 3:
        raise ValueError(f"expected masks with shape (N, H, W), got {masks.shape}")

    return masks


def resize_binary_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation.

    Args:
        mask: binary-like array
        shape: target shape as (height, width)

    Returns:
        uint8 binary mask.
    """
    if mask.shape == shape:
        return mask.astype(np.uint8)

    target_h, target_w = shape

    image = Image.fromarray((mask.astype(np.uint8) * 255))
    image = image.resize((target_w, target_h), Image.NEAREST)

    return (np.array(image) > 0).astype(np.uint8)


def bbox_area_1024(fragment: dict) -> float:
    """Return bbox area on the 1024 MedSAM grid."""
    box = fragment.get("bbox_xyxy_1024")

    if box is None:
        return float("nan")

    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def bbox_width_height_1024(fragment: dict) -> tuple[float, float]:
    """Return bbox width and height on the 1024 MedSAM grid."""
    box = fragment.get("bbox_xyxy_1024")

    if box is None:
        return float("nan"), float("nan")

    x1, y1, x2, y2 = [float(v) for v in box]

    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Compute xyxy bbox from a binary mask.

    Useful for sanity-checking decoded GT masks against metadata["bbox_xyxy"].
    """
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]