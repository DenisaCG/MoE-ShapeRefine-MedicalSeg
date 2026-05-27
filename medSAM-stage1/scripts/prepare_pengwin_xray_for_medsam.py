#!/usr/bin/env python3
"""Convert PENGWIN Task 2 X-ray data into MedSAM's imgs/gts training format.

The output layout matches MedSAM's 2D training convention:

    <output_root>/
      imgs/
      gts/
      metadata.jsonl

Each sample stores:
- imgs/*.npy: float32 array of shape (1024, 1024, 3), normalized to [0, 1]
- gts/*.npy: uint8 array of shape (N, 1024, 1024), where each channel is one
  fragment instance mask. This preserves overlapping fragments in the original
  X-ray projections.

MedSAM's training code can then sample one channel at a time and derive
bounding boxes from each binary instance mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PENGWIN_ROOT = WORKSPACE_ROOT / "data" / "pengwin"
XRAY_IMAGE_ROOT = PENGWIN_ROOT / "original" / "task2_xray" / "train" / "input" / "images" / "x-ray"
XRAY_LABEL_ROOT = PENGWIN_ROOT / "original" / "task2_xray" / "train" / "output" / "images" / "x-ray"
DEFAULT_OUTPUT_ROOT = PENGWIN_ROOT / "derived" / "medsam" / "xray"

CATEGORY_NAMES = {
    1: "SA",
    2: "LI",
    3: "RI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=XRAY_IMAGE_ROOT)
    parser.add_argument("--label-root", type=Path, default=XRAY_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images with no positive fragments. Disabled by default because "
        "MedSAM's training loader expects at least one foreground label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a smaller debug conversion run.",
    )
    return parser.parse_args()


def neglog_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    image = -np.log(image + 1e-2)

    lower = np.percentile(image, 1.0)
    upper = np.percentile(image, 99.0)
    image = np.clip(image, lower, upper)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    return image


def decode_segmentation(
    segmentation: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, int | str]]]:
    masks: list[np.ndarray] = []
    fragments: list[dict[str, int | str]] = []
    for category_id in sorted(CATEGORY_NAMES):
        for fragment_id in range(1, 11):
            shift = 10 * (category_id - 1) + fragment_id
            mask = ((segmentation >> shift) & 1).astype(np.uint8)
            if np.any(mask):
                masks.append(mask)
                fragments.append(
                    {
                        "category_id": category_id,
                        "category_name": CATEGORY_NAMES[category_id],
                        "fragment_id": fragment_id,
                    }
                )

    for label_id, fragment in enumerate(fragments, start=1):
        fragment["medsam_instance_id"] = label_id
    return masks, fragments


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    image_3c = np.repeat(image[:, :, None], 3, axis=-1)
    resized = resize(
        image_3c,
        (image_size, image_size, 3),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
        mode="constant",
    ).astype(np.float32)
    resized = np.clip(resized, 0.0, 1.0)
    return resized


def resize_masks(masks: list[np.ndarray], image_size: int) -> np.ndarray:
    if not masks:
        return np.zeros((0, image_size, image_size), dtype=np.uint8)

    resized_masks = []
    for mask in masks:
        resized = resize(
            mask,
            (image_size, image_size),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
            mode="constant",
        )
        resized_masks.append(resized.astype(np.uint8))
    return np.stack(resized_masks, axis=0)


def iter_case_ids(image_root: Path, label_root: Path) -> list[str]:
    image_case_ids = {path.stem for path in image_root.glob("*.tif")}
    label_case_ids = {path.stem for path in label_root.glob("*.tif")}
    return sorted(image_case_ids & label_case_ids)


def main() -> int:
    args = parse_args()

    imgs_root = args.output_root / "imgs"
    gts_root = args.output_root / "gts"
    imgs_root.mkdir(parents=True, exist_ok=True)
    gts_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "metadata.jsonl"

    case_ids = iter_case_ids(args.image_root, args.label_root)
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    kept = 0
    skipped_empty = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for case_id in tqdm(case_ids, desc="Converting PENGWIN X-ray"):
            image = np.array(Image.open(args.image_root / f"{case_id}.tif"))
            segmentation = np.array(Image.open(args.label_root / f"{case_id}.tif"))

            masks, fragments = decode_segmentation(segmentation)
            if not fragments and not args.keep_empty:
                skipped_empty += 1
                continue

            image_norm = neglog_normalize(image)
            image_resized = resize_image(image_norm, args.image_size)
            masks_resized = resize_masks(masks, args.image_size)

            sample_name = f"XRAY_PENGWIN_{case_id}.npy"
            np.save(imgs_root / sample_name, image_resized)
            np.save(gts_root / sample_name, masks_resized)

            record = {
                "sample_name": sample_name,
                "case_id": case_id,
                "original_image_path": str((args.image_root / f"{case_id}.tif").resolve()),
                "original_label_path": str((args.label_root / f"{case_id}.tif").resolve()),
                "image_shape": list(image.shape),
                "image_size": args.image_size,
                "num_fragments": len(fragments),
                "fragments": fragments,
            }
            metadata_file.write(json.dumps(record) + "\n")
            kept += 1

    print(f"saved {kept} samples to {args.output_root}")
    if skipped_empty:
        print(f"skipped {skipped_empty} empty-label samples")
    print(f"imgs dir: {imgs_root}")
    print(f"gts dir: {gts_root}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
