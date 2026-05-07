#!/usr/bin/env python3
"""Evaluate PENGWIN X-ray MedSAM first-stage predictions.

Evaluation unit:
    one fragment = one bounding box = one MedSAM mask slice = one GT fragment mask

Outputs one CSV row per fragment and prints overall/per-class summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import transform
from tqdm import tqdm


CATEGORY_NAMES = {1: "SA", 2: "LI", 3: "RI"}
PROJECT_DIR_NAME = "MoE-ShapeRefine-MedicalSeg"


def script_root() -> Path:
    return Path(__file__).resolve().parent


def find_project_root(root: Path) -> Path:
    candidates = [
    root,
    root.parent,
    root.parent / PROJECT_DIR_NAME,
    root / PROJECT_DIR_NAME,
    Path.cwd(),
    Path.cwd() / PROJECT_DIR_NAME,
    ]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate.resolve()
    return (root.parent / PROJECT_DIR_NAME).resolve()


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (project_root / path).resolve()


def parse_args() -> argparse.Namespace:
    root = script_root()
    project_root = find_project_root(root)
    default_pred_root = project_root / "data" / "medsam-predictions"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-metadata",
        type=Path,
        default=default_pred_root / "metadata.jsonl",
        help="MedSAM prediction metadata JSONL.",
    )
    parser.add_argument(
        "--pred-mask-root",
        type=Path,
        default=default_pred_root / "binary_masks",
        help="Directory containing MedSAM .npz mask files.",
    )
    parser.add_argument(
        "--box-metadata",
        type=Path,
        default=project_root / "data" / "bounding-boxes-xrays" / "metadata.jsonl",
        help="Bounding-box metadata JSONL with original_label_path when available.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "data" / "medsam-predictions" / "evaluation_pengwin.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of image records to evaluate for a smoke test.",
    )
    args = parser.parse_args()
    args.project_root = project_root
    args.pred_metadata = resolve_path(args.pred_metadata, project_root)
    args.pred_mask_root = resolve_path(args.pred_mask_root, project_root)
    args.box_metadata = resolve_path(args.box_metadata, project_root)
    args.output_csv = resolve_path(args.output_csv, project_root)
    return args


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_box_metadata_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record["sample_name"]: record for record in load_jsonl(path)}


def infer_label_path_from_image_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None
    text = str(image_path)
    if "/train/input/images/x-ray/" not in text:
        return None
    return Path(text.replace("/train/input/images/x-ray/", "/train/output/images/x-ray/"))


def get_label_path(pred_record: dict, box_metadata: dict[str, dict]) -> Path | None:
    sample_name = pred_record.get("sample_name")
    box_record = box_metadata.get(sample_name, {})
    label_path = box_record.get("original_label_path")
    if label_path:
        return Path(label_path)

    return infer_label_path_from_image_path(pred_record.get("original_image_path"))


def decode_pengwin_fragment(
    label_path: Path,
    category_id: int,
    fragment_id: int,
) -> np.ndarray:
    """Decode one PENGWIN X-ray fragment from the bit-packed label image.

    Bits 1-10 are SA, bits 11-20 are LI, and bits 21-30 are RI.
    """
    segmentation = np.array(Image.open(label_path))
    shift = 10 * (category_id - 1) + fragment_id
    return ((segmentation >> shift) & 1).astype(np.uint8)


def resize_binary_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.uint8)
    return transform.resize(
        mask,
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    pred_area = int(pred_bool.sum())
    gt_area = int(gt_bool.sum())
    denom = pred_area + gt_area
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_bool, gt_bool).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_bool, gt_bool).sum() / union)


def surface_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    eroded = ndimage.binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool))
    return np.logical_xor(mask_bool, eroded)


def surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_surface = surface_mask(source)
    target_surface = surface_mask(target)
    if not source_surface.any() or not target_surface.any():
        return np.array([], dtype=np.float32)
    distance_map = ndimage.distance_transform_edt(~target_surface)
    return distance_map[source_surface].astype(np.float32)


def hd95_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        return np.nan
    distances = np.concatenate(
        [surface_distances(pred_bool, gt_bool), surface_distances(gt_bool, pred_bool)]
    )
    if distances.size == 0:
        return np.nan
    return float(np.percentile(distances, 95))


def assd_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        return np.nan
    distances = np.concatenate(
        [surface_distances(pred_bool, gt_bool), surface_distances(gt_bool, pred_bool)]
    )
    if distances.size == 0:
        return np.nan
    return float(distances.mean())


def pred_mask_path(record: dict, pred_mask_root: Path) -> Path:
    metadata_path = record.get("binary_masks_path")
    if metadata_path and Path(metadata_path).exists():
        return Path(metadata_path)
    return pred_mask_root / f"{record['sample_name']}.npz"


def evaluate_record(
    record: dict,
    box_metadata: dict[str, dict],
    pred_mask_root: Path,
) -> list[dict]:
    sample_name = record["sample_name"]
    case_id = record.get("case_id") or sample_name.removeprefix("XRAY_PENGWIN_")
    label_path = get_label_path(record, box_metadata)
    if label_path is None or not label_path.exists():
        raise FileNotFoundError(f"missing GT label for {sample_name}: {label_path}")

    mask_path = pred_mask_path(record, pred_mask_root)
    if not mask_path.exists():
        raise FileNotFoundError(f"missing predicted masks for {sample_name}: {mask_path}")

    pred_masks = np.load(mask_path)["masks"]
    fragments = record.get("fragments", [])
    if len(fragments) != pred_masks.shape[0]:
        raise ValueError(
            f"{sample_name}: {len(fragments)} fragments but {pred_masks.shape[0]} predicted masks"
        )

    rows = []
    for fragment_index, fragment in enumerate(fragments):
        category_id = int(fragment["category_id"])
        fragment_id = int(fragment["fragment_id"])
        category_name = fragment.get("category_name", CATEGORY_NAMES.get(category_id, "unknown"))

        pred = pred_masks[fragment_index].astype(np.uint8)
        gt = decode_pengwin_fragment(label_path, category_id, fragment_id)
        gt = resize_binary_nearest(gt, pred.shape)

        pred_area = int(pred.astype(bool).sum())
        gt_area = int(gt.astype(bool).sum())
        rows.append(
            {
                "sample_name": sample_name,
                "case_id": case_id,
                "fragment_index": fragment_index,
                "category_name": category_name,
                "fragment_id": fragment_id,
                "dice": dice_score(pred, gt),
                "iou": iou_score(pred, gt),
                "hd95": hd95_score(pred, gt),
                "assd": assd_score(pred, gt),
                "pred_area": pred_area,
                "gt_area": gt_area,
            }
        )
    return rows


def print_summary(frame: pd.DataFrame) -> None:
    metric_cols = ["dice", "iou", "hd95", "assd"]
    print("\nOverall mean metrics:")
    print(frame[metric_cols].mean(numeric_only=True).to_string())

    print("\nPer-class mean metrics:")
    per_class = frame.groupby("category_name", sort=True)[metric_cols].mean(numeric_only=True)
    for category in ["SA", "LI", "RI"]:
        if category in per_class.index:
            print(f"\n{category}")
            print(per_class.loc[category].to_string())


def main() -> int:
    args = parse_args()
    if not args.pred_metadata.exists():
        raise FileNotFoundError(f"missing prediction metadata: {args.pred_metadata}")

    pred_records = load_jsonl(args.pred_metadata)
    if args.limit is not None:
        pred_records = pred_records[: args.limit]
    if not pred_records:
        raise RuntimeError("no prediction records found")

    box_metadata = load_box_metadata_map(args.box_metadata)
    print(f"prediction metadata: {args.pred_metadata}")
    print(f"prediction mask root: {args.pred_mask_root}")
    print(f"box metadata: {args.box_metadata}")
    print(f"output csv: {args.output_csv}")
    print(f"records: {len(pred_records)}")
    print(f"limit: {args.limit if args.limit is not None else 'no'}")

    rows = []
    for record in tqdm(pred_records, desc="Evaluating MedSAM predictions"):
        rows.extend(evaluate_record(record, box_metadata, args.pred_mask_root))

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    print_summary(frame)
    print(f"\nSaved evaluation CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
