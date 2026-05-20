#!/usr/bin/env python3
"""Evaluate PENGWIN X-ray MedSAM or MoE per-fragment predictions.

Evaluation unit:
    one fragment = one bounding box = one predicted mask slice = one GT fragment mask

Outputs one CSV row per fragment and prints:
    - overall metrics
    - per-class metrics: SA, LI, RI
    - per-size metrics: small, large

Fast mode:
    Use --skip-boundary to compute only Dice and IoU.

Boundary mode:
    Computes Dice, IoU, HD95, and ASSD.
"""

from __future__ import annotations
import csv
from collections import defaultdict
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader_utils import (
    CATEGORY_NAMES,
    bbox_area_1024,
    bbox_width_height_1024,
    decode_pengwin_fragment_from_record,
    get_category_name,
    get_label_path,
    load_box_metadata_map,
    load_jsonl,
    load_pengwin_label,
    load_prediction_masks,
    resize_binary_nearest,
    resolve_existing_path,
)

PROJECT_DIR_NAME = "MoE-ShapeRefine-MedicalSeg"
METRIC_COLS = ["dice", "iou", "hd95", "assd"]

# Globals used only for multiprocessing workers.
_WORKER_BOX_METADATA: dict[str, dict] | None = None
_WORKER_PRED_MASK_ROOT: Path | None = None
_WORKER_SKIP_BOUNDARY: bool = False
_WORKER_METRIC_SPACE: str = "original"


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


#def resolve_existing_path(path: Path | str) -> Path:
    """Handle common /gpfs/home5/... vs /home/... path differences."""
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


def parse_args() -> argparse.Namespace:
    root = script_root()
    project_root = find_project_root(root)
    default_pred_root = project_root / "data" / "medsam-predictions"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-metadata",
        type=Path,
        default=default_pred_root / "metadata.jsonl",
        help="Prediction metadata JSONL. For MoE, this can still be the MedSAM metadata.",
    )
    parser.add_argument(
        "--pred-mask-root",
        type=Path,
        default=default_pred_root / "binary_masks",
        help=(
            "Directory containing predicted .npz mask files. "
            "For MedSAM: data/medsam-predictions/binary_masks. "
            "For MoE: data/moe-predictions/binary_masks."
        ),
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
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional case_id filter, e.g. 001_0000.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="Optional sample_name filter, e.g. XRAY_PENGWIN_001_0000.",
    )
    parser.add_argument(
        "--skip-boundary",
        action="store_true",
        help="Skip HD95 and ASSD. Much faster; computes Dice and IoU only.",
    )
    parser.add_argument(
        "--metric-space",
        choices=["original", "prediction"],
        default="original",
        help=(
            "original: resize prediction down to original GT label resolution before metrics. "
            "prediction: resize GT up to prediction resolution before metrics."
        ),
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=None,
        help=(
            "Bounding-box area threshold on the 1024 grid for small/large grouping. "
            "If not provided, the median bbox area of the evaluated fragments is used."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes. Use 0 for serial. "
            "Try 4 or 8 first on Snellius; too many workers may overload GPFS I/O."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Chunksize for multiprocessing map.",
    )

    args = parser.parse_args()
    args.project_root = project_root
    args.pred_metadata = resolve_path(args.pred_metadata, project_root)
    args.pred_mask_root = resolve_path(args.pred_mask_root, project_root)
    args.box_metadata = resolve_path(args.box_metadata, project_root)
    args.output_csv = resolve_path(args.output_csv, project_root)
    return args


#def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


#def load_box_metadata_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record["sample_name"]: record for record in load_jsonl(path)}


#def infer_label_path_from_image_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None
    text = str(image_path)
    if "/train/input/images/x-ray/" not in text:
        return None
    return Path(text.replace("/train/input/images/x-ray/", "/train/output/images/x-ray/"))


#def get_label_path(pred_record: dict, box_metadata: dict[str, dict]) -> Path | None:
    sample_name = pred_record.get("sample_name")
    box_record = box_metadata.get(sample_name, {})
    label_path = box_record.get("original_label_path")
    if label_path:
        return resolve_existing_path(label_path)

    inferred = infer_label_path_from_image_path(pred_record.get("original_image_path"))
    if inferred is not None:
        return resolve_existing_path(inferred)

    return None


#def decode_pengwin_fragment_from_array( segmentation: np.ndarray, category_id: int, fragment_id: int) -> np.ndarray:
    """Decode one PENGWIN X-ray fragment from the bit-packed label image.

    Current convention used here:
        bits 1-10  are SA
        bits 11-20 are LI
        bits 21-30 are RI

    This matches the team's current script convention:
        shift = 10 * (category_id - 1) + fragment_id
    """
    shift = 10 * (category_id - 1) + fragment_id
    return ((segmentation >> shift) & 1).astype(np.uint8)


#def resize_binary_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation using PIL."""
    if mask.shape == shape:
        return mask.astype(np.uint8)

    target_h, target_w = shape

    image = Image.fromarray((mask.astype(np.uint8) * 255))
    image = image.resize((target_w, target_h), Image.NEAREST)

    return (np.array(image) > 0).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    pred_area = int(pred_bool.sum())
    gt_area = int(gt_bool.sum())
    denom = pred_area + gt_area
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    if union == 0:
        return 1.0

    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    return float(intersection / union)


def surface_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)

    eroded = ndimage.binary_erosion(
        mask_bool,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    return np.logical_xor(mask_bool, eroded)


def surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_surface = surface_mask(source)
    target_surface = surface_mask(target)
    if not source_surface.any() or not target_surface.any():
        return np.array([], dtype=np.float32)
    distance_map = ndimage.distance_transform_edt(~target_surface)
    return distance_map[source_surface].astype(np.float32)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Compute HD95 and ASSD together.

    This avoids calculating the same surface distances twice.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    if not pred_bool.any() and not gt_bool.any():
        return 0.0, 0.0

    if not pred_bool.any() or not gt_bool.any():
        return np.nan, np.nan

    distances = np.concatenate(
        [
            surface_distances(pred_bool, gt_bool),
            surface_distances(gt_bool, pred_bool),
        ]
    )
    if distances.size == 0:
        return np.nan, np.nan

    hd95 = float(np.percentile(distances, 95))
    assd = float(distances.mean())

    return hd95, assd


#def bbox_area_1024(fragment: dict) -> float:
    box = fragment.get("bbox_xyxy_1024")

    if box is None:
        return np.nan

    x1, y1, x2, y2 = [float(v) for v in box]

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


#def bbox_width_height_1024(fragment: dict) -> tuple[float, float]:
    box = fragment.get("bbox_xyxy_1024")

    if box is None:
        return np.nan, np.nan

    x1, y1, x2, y2 = [float(v) for v in box]

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width, height


def pred_mask_path(record: dict, pred_mask_root: Path) -> Path:
    """Find predicted mask path from the selected prediction root.

    This prefers --pred-mask-root so the same evaluator can later evaluate MoE
    outputs by changing only the prediction directory.
    """
    return resolve_existing_path(pred_mask_root / f"{record['sample_name']}.npz")


def evaluate_record(
    record: dict,
    box_metadata: dict[str, dict],
    pred_mask_root: Path,
    skip_boundary: bool,
    metric_space: str,
) -> list[dict]:
    sample_name = record["sample_name"]
    case_id = record.get("case_id") or sample_name.removeprefix("XRAY_PENGWIN_")
    label_path = get_label_path(record, box_metadata)
    if label_path is None or not label_path.exists():
        raise FileNotFoundError(f"missing GT label for {sample_name}: {label_path}")

    mask_path = pred_mask_path(record, pred_mask_root)
    if not mask_path.exists():
        raise FileNotFoundError(f"missing predicted masks for {sample_name}: {mask_path}")

    pred_masks = load_prediction_masks(mask_path)
    fragments = record.get("fragments", [])
    if len(fragments) != pred_masks.shape[0]:
        raise ValueError(
            f"{sample_name}: {len(fragments)} fragments but "
            f"{pred_masks.shape[0]} predicted masks"
        )

    # Important speed fix:
    # Load the GT label only once per image, not once per fragment.
    segmentation = load_pengwin_label(label_path)

    rows: list[dict] = []

    for fragment_index, fragment in enumerate(fragments):
        category_id = int(fragment["category_id"])
        fragment_id = int(fragment["fragment_id"])
        category_name = get_category_name(fragment)

        pred_original_size = pred_masks[fragment_index].astype(np.uint8)
        gt_original_size = decode_pengwin_fragment_from_record(
            segmentation=segmentation,
            fragment=fragment,
        )

        if metric_space == "original":
            # Faster and more PENGWIN-like:
            # evaluate in the original GT label space, usually 448x448.
            gt = gt_original_size
            pred = resize_binary_nearest(pred_original_size, gt.shape)
        elif metric_space == "prediction":
            # Evaluate in MedSAM/MoE output space, usually 1024x1024.
            pred = pred_original_size
            gt = resize_binary_nearest(gt_original_size, pred.shape)
        else:
            raise ValueError(f"unknown metric_space: {metric_space}")

        pred_area = int(pred.astype(bool).sum())
        gt_area = int(gt.astype(bool).sum())

        dice = dice_score(pred, gt)
        iou = iou_score(pred, gt)

        if skip_boundary:
            hd95 = np.nan
            assd = np.nan
        else:
            hd95, assd = boundary_metrics(pred, gt)

        box_area = bbox_area_1024(fragment)
        box_width, box_height = bbox_width_height_1024(fragment)

        rows.append(
            {
                "sample_name": sample_name,
                "case_id": case_id,
                "fragment_index": fragment_index,
                "category_name": category_name,
                "category_id": category_id,
                "fragment_id": fragment_id,
                "medsam_instance_id": fragment.get("medsam_instance_id"),
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "assd": assd,
                "pred_area": pred_area,
                "gt_area": gt_area,
                "bbox_area_1024": box_area,
                "bbox_width_1024": box_width,
                "bbox_height_1024": box_height,
                "metric_space": metric_space,
                "eval_height": int(pred.shape[0]),
                "eval_width": int(pred.shape[1]),
                "pred_mask_path": str(mask_path),
                "original_label_path": str(label_path),
            }
        )
    return rows


def init_worker(
    box_metadata: dict[str, dict],
    pred_mask_root: str,
    skip_boundary: bool,
    metric_space: str,
) -> None:
    global _WORKER_BOX_METADATA
    global _WORKER_PRED_MASK_ROOT
    global _WORKER_SKIP_BOUNDARY
    global _WORKER_METRIC_SPACE

    _WORKER_BOX_METADATA = box_metadata
    _WORKER_PRED_MASK_ROOT = Path(pred_mask_root)
    _WORKER_SKIP_BOUNDARY = skip_boundary
    _WORKER_METRIC_SPACE = metric_space


def evaluate_record_worker(record: dict) -> list[dict]:
    if _WORKER_BOX_METADATA is None or _WORKER_PRED_MASK_ROOT is None:
        raise RuntimeError("worker was not initialised correctly")

    return evaluate_record(
        record=record,
        box_metadata=_WORKER_BOX_METADATA,
        pred_mask_root=_WORKER_PRED_MASK_ROOT,
        skip_boundary=_WORKER_SKIP_BOUNDARY,
        metric_space=_WORKER_METRIC_SPACE,
    )


def choose_area_threshold(records: list[dict], requested_threshold: float | None) -> float:
    """Choose the small/large threshold.

    If --area-threshold is provided, use it.
    Otherwise use the median bbox area from the evaluated records.
    """
    if requested_threshold is not None:
        return float(requested_threshold)

    areas: list[float] = []

    for record in records:
        for fragment in record.get("fragments", []):
            area = bbox_area_1024(fragment)
            if math.isfinite(float(area)):
                areas.append(float(area))

    if not areas:
        return math.nan

    return float(np.median(areas))


def add_size_group_to_row(row: dict, area_threshold: float) -> dict:
    """Add small/large grouping columns to one already-computed metric row.

    This does not affect Dice, IoU, HD95, or ASSD.
    It only labels the row for subgroup analysis.
    """
    area = float(row.get("bbox_area_1024", math.nan))
    row["area_threshold_1024"] = area_threshold

    if math.isnan(area_threshold) or not math.isfinite(area):
        row["is_large"] = False
        row["size_group"] = "unknown"
    else:
        is_large = area >= area_threshold
        row["is_large"] = bool(is_large)
        row["size_group"] = "large" if is_large else "small"

    return row


def make_summary_bucket() -> dict:
    return {
        "n": 0,
        "sums": defaultdict(float),
        "counts": defaultdict(int),
    }


def update_summary_bucket(bucket: dict, row: dict) -> None:
    bucket["n"] += 1

    for metric in METRIC_COLS:
        value = row.get(metric, np.nan)

        try:
            value = float(value)
        except Exception:
            continue

        if math.isfinite(value):
            bucket["sums"][metric] += value
            bucket["counts"][metric] += 1


def update_summaries(
    overall_summary: dict,
    class_summary: dict,
    size_summary: dict,
    class_size_summary: dict,
    row: dict,
) -> None:
    category_name = row.get("category_name", "unknown")
    size_group = row.get("size_group", "unknown")

    update_summary_bucket(overall_summary["overall"], row)
    update_summary_bucket(class_summary[category_name], row)
    update_summary_bucket(size_summary[size_group], row)
    update_summary_bucket(class_size_summary[(category_name, size_group)], row)


def print_summary_table(title: str, summary: dict, preferred_order: list | None = None) -> None:
    print(f"\n{title}")

    if not summary:
        print("No rows.")
        return

    if preferred_order is None:
        keys = sorted(summary.keys(), key=lambda x: str(x))
    else:
        keys = [key for key in preferred_order if key in summary]
        keys += [key for key in summary.keys() if key not in keys]

    header = f"{'group':<20} {'n':>10}"
    for metric in METRIC_COLS:
        header += f" {metric + '_mean':>14}"
    print(header)
    print("-" * len(header))

    for key in keys:
        bucket = summary[key]

        if isinstance(key, tuple):
            group_name = "/".join(str(part) for part in key)
        else:
            group_name = str(key)

        line = f"{group_name:<20} {bucket['n']:>10}"

        for metric in METRIC_COLS:
            count = bucket["counts"][metric]
            if count == 0:
                mean_text = "nan"
            else:
                mean_text = f"{bucket['sums'][metric] / count:.4f}"

            line += f" {mean_text:>14}"

        print(line)


def print_streaming_summary(
    overall_summary: dict,
    class_summary: dict,
    size_summary: dict,
    class_size_summary: dict,
    rows_written: int,
    records_evaluated: int,
    failed_records: int,
    empty_pred_count: int,
    empty_gt_count: int,
    area_threshold: float,
) -> None:
    print("\nEvaluation summary")
    print("=" * 90)
    print(f"Image records evaluated: {records_evaluated}")
    print(f"Fragment rows written: {rows_written}")
    print(f"Failed records: {failed_records}")
    print(f"Empty predictions: {empty_pred_count}")
    print(f"Empty GT masks: {empty_gt_count}")
    print(f"Area threshold on 1024 grid: {area_threshold}")

    print_summary_table(
        "Overall mean metrics:",
        overall_summary,
        preferred_order=["overall"],
    )

    print_summary_table(
        "Per-class mean metrics:",
        class_summary,
        preferred_order=["SA", "LI", "RI"],
    )

    print_summary_table(
        "Per-size mean metrics:",
        size_summary,
        preferred_order=["small", "large", "unknown"],
    )

    print_summary_table(
        "Per-class and per-size mean metrics:",
        class_size_summary,
        preferred_order=[
            ("SA", "small"),
            ("SA", "large"),
            ("LI", "small"),
            ("LI", "large"),
            ("RI", "small"),
            ("RI", "large"),
        ],
    )

    print("=" * 90)


def main() -> int:
    args = parse_args()

    if not args.pred_metadata.exists():
        raise FileNotFoundError(f"missing prediction metadata: {args.pred_metadata}")

    if not args.pred_mask_root.exists():
        raise FileNotFoundError(f"missing prediction mask root: {args.pred_mask_root}")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    pred_records = load_jsonl(args.pred_metadata)

    if args.case_id is not None:
        pred_records = [r for r in pred_records if r.get("case_id") == args.case_id]

    if args.sample_name is not None:
        pred_records = [r for r in pred_records if r.get("sample_name") == args.sample_name]

    if args.limit is not None:
        pred_records = pred_records[: args.limit]

    if not pred_records:
        raise RuntimeError("no prediction records found")

    box_metadata = load_box_metadata_map(args.box_metadata)

    # Important for streaming:
    # Decide the area threshold before writing rows, so each row can be labeled
    # small/large immediately.
    area_threshold = choose_area_threshold(pred_records, args.area_threshold)

    print(f"prediction metadata: {args.pred_metadata}")
    print(f"prediction mask root: {args.pred_mask_root}")
    print(f"box metadata: {args.box_metadata}")
    print(f"output csv: {args.output_csv}")
    print(f"records: {len(pred_records)}")
    print(f"limit: {args.limit if args.limit is not None else 'no'}")
    print(f"skip boundary: {args.skip_boundary}")
    print(f"metric space: {args.metric_space}")
    print(f"num workers: {args.num_workers}")
    print(f"area threshold: {area_threshold}")

    fieldnames = [
        "sample_name",
        "case_id",
        "fragment_index",
        "category_name",
        "category_id",
        "fragment_id",
        "medsam_instance_id",
        "dice",
        "iou",
        "hd95",
        "assd",
        "pred_area",
        "gt_area",
        "bbox_area_1024",
        "bbox_width_1024",
        "bbox_height_1024",
        "area_threshold_1024",
        "is_large",
        "size_group",
        "metric_space",
        "eval_height",
        "eval_width",
        "pred_mask_path",
        "original_label_path",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    overall_summary = defaultdict(make_summary_bucket)
    class_summary = defaultdict(make_summary_bucket)
    size_summary = defaultdict(make_summary_bucket)
    class_size_summary = defaultdict(make_summary_bucket)

    rows_written = 0
    records_evaluated = 0
    failed_records = 0
    empty_pred_count = 0
    empty_gt_count = 0

    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if args.num_workers == 0:
            iterator = pred_records

            for record in tqdm(iterator, desc="Evaluating predictions"):
                try:
                    record_rows = evaluate_record(
                        record=record,
                        box_metadata=box_metadata,
                        pred_mask_root=args.pred_mask_root,
                        skip_boundary=args.skip_boundary,
                        metric_space=args.metric_space,
                    )

                    records_evaluated += 1

                    for row in record_rows:
                        row = add_size_group_to_row(row, area_threshold)

                        writer.writerow(row)
                        rows_written += 1

                        if int(row.get("pred_area", 0)) == 0:
                            empty_pred_count += 1
                        if int(row.get("gt_area", 0)) == 0:
                            empty_gt_count += 1

                        update_summaries(
                            overall_summary=overall_summary,
                            class_summary=class_summary,
                            size_summary=size_summary,
                            class_size_summary=class_size_summary,
                            row=row,
                        )

                    if rows_written % 1000 == 0:
                        csv_file.flush()

                except Exception as exc:
                    failed_records += 1
                    sample_name = record.get("sample_name", "unknown")
                    print(f"WARNING: failed record {sample_name}: {exc}")

        else:
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_worker,
                initargs=(
                    box_metadata,
                    str(args.pred_mask_root),
                    args.skip_boundary,
                    args.metric_space,
                ),
            ) as executor:
                result_iterator = executor.map(
                    evaluate_record_worker,
                    pred_records,
                    chunksize=args.chunksize,
                )

                for record_rows in tqdm(
                    result_iterator,
                    total=len(pred_records),
                    desc="Evaluating predictions",
                ):
                    records_evaluated += 1

                    for row in record_rows:
                        row = add_size_group_to_row(row, area_threshold)

                        writer.writerow(row)
                        rows_written += 1

                        if int(row.get("pred_area", 0)) == 0:
                            empty_pred_count += 1
                        if int(row.get("gt_area", 0)) == 0:
                            empty_gt_count += 1

                        update_summaries(
                            overall_summary=overall_summary,
                            class_summary=class_summary,
                            size_summary=size_summary,
                            class_size_summary=class_size_summary,
                            row=row,
                        )

                    if rows_written % 1000 == 0:
                        csv_file.flush()

    print_streaming_summary(
        overall_summary=overall_summary,
        class_summary=class_summary,
        size_summary=size_summary,
        class_size_summary=class_size_summary,
        rows_written=rows_written,
        records_evaluated=records_evaluated,
        failed_records=failed_records,
        empty_pred_count=empty_pred_count,
        empty_gt_count=empty_gt_count,
        area_threshold=area_threshold,
    )

    print(f"\nSaved evaluation CSV: {args.output_csv}")

    if rows_written == 0:
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
