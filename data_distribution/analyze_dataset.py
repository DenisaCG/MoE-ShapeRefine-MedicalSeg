#!/usr/bin/env python3
"""Analyze PENGWIN X-ray segmentation shape distributions.

This script assumes the README pipeline outputs already exist:
- data/bounding-boxes-xrays/metadata.jsonl for GT fragments
- data/medsam-predictions/metadata.jsonl plus binary_masks/*.npz for MedSAM masks
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "moe_shape_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform
from tqdm import tqdm

try:
    from .plotting import generate_examples, generate_plots
    from .shape_features import dice_score, extract_mask_features
except ImportError:
    from plotting import generate_examples, generate_plots
    from shape_features import dice_score, extract_mask_features


THRESHOLD_METRICS = [
    "area",
    "relative_area",
    "elongation",
    "compactness",
    "connected_components",
]
THRESHOLD_PERCENTILES = {
    "p10": 0.10,
    "p25": 0.25,
    "p33": 0.33,
    "p50": 0.50,
    "p66": 0.66,
    "p75": 0.75,
    "p90": 0.90,
}


def script_root() -> Path:
    return Path(__file__).resolve().parent


def default_data_root() -> Path:
    """Prefer canonical data/ next to the current run, then next to this script."""
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data
    return script_root().parent / "data"


def resolve_path(path: Path | None) -> Path | None:
    """Resolve relative paths against CWD first, then the repository containing this script."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (script_root().parent / path).resolve()


def parse_args() -> argparse.Namespace:
    root = script_root()
    data_root = default_data_root()
    default_pred_root = data_root / "medsam-predictions"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-source",
        choices=("gt", "medsam"),
        default="gt",
        help="Which masks to analyze: decoded ground truth or MedSAM predictions.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=data_root / "bounding-boxes-xrays" / "metadata.jsonl",
        help="GT metadata JSONL produced by the README data-preparation pipeline.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=default_pred_root,
        help="Root directory containing MedSAM predictions.",
    )
    parser.add_argument(
        "--pred-mask-root",
        type=Path,
        default=None,
        help="Directory containing MedSAM binary mask .npz files.",
    )
    parser.add_argument(
        "--pred-metadata",
        type=Path,
        default=None,
        help="MedSAM prediction metadata JSONL.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to data_analysis_gt or data_analysis_medsam.",
    )
    parser.add_argument(
        "--include-medsam",
        action="store_true",
        help="Compute MedSAM Dice/failure trends when GT labels are available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of image records to process for a quick smoke test.",
    )

    args = parser.parse_args()
    args.script_root = root
    args.metadata = resolve_path(args.metadata)
    args.pred_root = resolve_path(args.pred_root)
    args.pred_mask_root = (
        resolve_path(args.pred_mask_root)
        if args.pred_mask_root
        else args.pred_root / "binary_masks"
    )
    args.pred_metadata = (
        resolve_path(args.pred_metadata)
        if args.pred_metadata
        else args.pred_root / "metadata.jsonl"
    )
    args.output_root = (
        resolve_path(args.output_root)
        if args.output_root
        else root / f"data_analysis_{args.mask_source}"
    )
    return args


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_required_jsonl(path: Path, description: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} metadata file does not exist: {path}\n"
            "Run the README pipeline first, or pass a custom metadata path."
        )
    return load_jsonl(path)


def decode_single_fragment(
    label_path: Path,
    category_id: int,
    fragment_id: int,
) -> np.ndarray:
    segmentation = np.array(Image.open(label_path))
    shift = 10 * (category_id - 1) + fragment_id
    return ((segmentation >> shift) & 1).astype(np.uint8)


def mask_path_for_record(record: dict, pred_mask_root: Path) -> Path:
    sample_name = record.get("sample_name")
    return Path(
        record.get("binary_masks_path")
        or pred_mask_root / f"{sample_name}.npz"
    )


def add_medsam_dice(rows: list[dict], pred_records: dict[str, dict], pred_mask_root: Path) -> None:
    """Mutate GT feature rows with MedSAM Dice scores using prediction metadata."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sample_name"], []).append(row)

    for sample_name, sample_rows in tqdm(grouped.items(), desc="Computing MedSAM Dice"):
        pred_record = pred_records.get(sample_name)
        if not pred_record:
            continue
        mask_path = mask_path_for_record(pred_record, pred_mask_root)
        if not mask_path.exists():
            continue
        predictions = np.load(mask_path)["masks"]
        for row in sample_rows:
            label_path = Path(str(row.get("label_path", "")))
            if not label_path.exists():
                continue
            pred_idx = int(row["mask_index"])
            if pred_idx < 0 or pred_idx >= predictions.shape[0]:
                continue
            gt = decode_single_fragment(
                label_path,
                int(row["category_id"]),
                int(row["fragment_id"]),
            )
            pred = predictions[pred_idx]
            if pred.shape != gt.shape:
                gt = transform.resize(
                    gt,
                    pred.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
            row["medsam_dice"] = dice_score(pred, gt)


def base_row(
    record: dict,
    image_shape: tuple[int, int],
    fragment: dict,
    mask_source: str,
) -> dict:
    case_id = record.get("case_id") or record.get("sample_name", "").removeprefix("XRAY_PENGWIN_")
    return {
        "mask_source": mask_source,
        "case_id": case_id,
        "sample_name": record.get("sample_name"),
        "split": record.get("split", "unknown"),
        "image_path": record.get("original_image_path"),
        "label_path": record.get("original_label_path"),
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        **fragment,
    }


def extract_gt_features(
    records: list[dict],
    include_medsam: bool,
    pred_records: dict[str, dict],
    pred_mask_root: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for record in tqdm(records, desc="Extracting GT mask features"):
        label_path = Path(record["original_label_path"])
        if not label_path.exists():
            continue
        image_shape = np.array(Image.open(label_path)).shape[:2]

        for mask_index, fragment in enumerate(record.get("fragments", []) or []):
            fragment = dict(fragment)
            fragment["mask_index"] = mask_index
            mask = decode_single_fragment(
                label_path,
                int(fragment["category_id"]),
                int(fragment["fragment_id"]),
            )
            row = base_row(record, image_shape, fragment, "gt")
            row.update(extract_mask_features(mask, image_shape))
            rows.append(row)

    if include_medsam:
        add_medsam_dice(rows, pred_records, pred_mask_root)

    return pd.DataFrame(rows)


def fragment_from_prediction(record: dict, mask_index: int) -> dict:
    fragments = record.get("fragments") or []
    fragment = dict(fragments[mask_index]) if mask_index < len(fragments) else {}
    fragment.setdefault("category_id", np.nan)
    fragment.setdefault("category_name", "unknown")
    fragment.setdefault("fragment_id", mask_index + 1)
    fragment.setdefault("medsam_instance_id", mask_index + 1)
    fragment["mask_index"] = mask_index
    return fragment


def extract_medsam_features(
    records: list[dict],
    pred_mask_root: Path,
    include_medsam: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for record in tqdm(records, desc="Extracting MedSAM mask features"):
        mask_path = mask_path_for_record(record, pred_mask_root)
        if not mask_path.exists():
            continue
        masks = np.load(mask_path)["masks"]

        for mask_index, mask in enumerate(masks):
            fragment = fragment_from_prediction(record, mask_index)
            image_shape = mask.shape[:2]
            row = base_row(record, image_shape, fragment, "medsam")
            row["mask_path"] = str(mask_path)
            row.update(extract_mask_features(mask.astype(np.uint8), image_shape))

            label_path = Path(str(row.get("label_path", "")))
            if include_medsam and label_path.exists():
                gt = decode_single_fragment(
                    label_path,
                    int(row["category_id"]),
                    int(row["fragment_id"]),
                )
                if gt.shape != mask.shape:
                    gt = transform.resize(
                        gt,
                        mask.shape,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(np.uint8)
                row["medsam_dice"] = dice_score(mask, gt)
            rows.append(row)

    return pd.DataFrame(rows)


def threshold_suggestions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in THRESHOLD_METRICS:
        values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
        row = {"metric": metric}
        for name, percentile in THRESHOLD_PERCENTILES.items():
            row[name] = float(values.quantile(percentile)) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_lookup(thresholds: pd.DataFrame, metric: str, percentile: str) -> float:
    value = thresholds.loc[thresholds["metric"] == metric, percentile]
    if value.empty:
        return np.nan
    return float(value.iloc[0])


def suggested_shape_group(row: pd.Series) -> str:
    if row["fragmentation_bin"] == "fragmented":
        return f"fragmented_{row['compactness_bin']}"
    if row["elongation_bin"] == "elongated":
        return f"{row['size_bin']}_elongated"
    if row["compactness_bin"] == "complex":
        return f"{row['size_bin']}_complex"
    return f"{row['size_bin']}_compact"


def add_gating_columns(frame: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    area_p33 = threshold_lookup(thresholds, "area", "p33")
    area_p66 = threshold_lookup(thresholds, "area", "p66")
    elongation_p75 = threshold_lookup(thresholds, "elongation", "p75")
    compactness_p75 = threshold_lookup(thresholds, "compactness", "p75")

    frame["size_bin"] = np.select(
        [frame["area"] <= area_p33, frame["area"] <= area_p66],
        ["small", "medium"],
        default="large",
    )
    frame["elongation_bin"] = np.where(
        frame["elongation"].notna() & (frame["elongation"] > elongation_p75),
        "elongated",
        "normal",
    )
    frame["compactness_bin"] = np.where(
        frame["compactness"].notna() & (frame["compactness"] > compactness_p75),
        "complex",
        "simple",
    )
    frame["fragmentation_bin"] = np.where(
        frame["connected_components"] == 1,
        "connected",
        "fragmented",
    )
    frame["suggested_shape_group"] = frame.apply(suggested_shape_group, axis=1)
    return frame


def write_failure_summaries(frame: pd.DataFrame, output_root: Path) -> None:
    if "medsam_dice" not in frame.columns:
        return
    scored = frame.dropna(subset=["medsam_dice"])
    if scored.empty:
        return
    scored = scored.copy()
    scored["category_name"] = scored["category_name"].fillna("unknown")

    per_class = (
        scored.groupby("category_name")["medsam_dice"]
        .agg(count="count", mean_dice="mean", median_dice="median", std_dice="std")
        .reset_index()
    )
    per_class.to_csv(output_root / "medsam_failure_summary_per_class.csv", index=False)

    by_shape_group = (
        scored.groupby("suggested_shape_group")["medsam_dice"]
        .agg(count="count", mean_dice="mean", median_dice="median")
        .reset_index()
    )
    by_shape_group.to_csv(output_root / "medsam_failure_summary_by_shape_group.csv", index=False)


def category_counts_from_records(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for record in records:
        for fragment in record.get("fragments", []) or []:
            counts[fragment.get("category_name", "unknown")] += 1
    return counts


def print_validation(
    args: argparse.Namespace,
    records: list[dict],
    stage: str,
    features: pd.DataFrame | None = None,
) -> None:
    unique_images = len({record.get("sample_name") for record in records})
    fragments = sum(len(record.get("fragments", []) or []) for record in records)
    if features is not None and not features.empty:
        unique_images = int(features["sample_name"].nunique())
        fragments = len(features)
        frame_counts = Counter(features["category_name"].fillna("unknown"))
    else:
        frame_counts = category_counts_from_records(records)

    print(f"\nValidation ({stage})")
    print(f"Mask source: {args.mask_source}")
    print(f"Unique images: {unique_images}")
    print(f"Fragments: {fragments}")
    print(f"Category counts: {dict(sorted(frame_counts.items()))}")
    print(f"Limit used: {args.limit if args.limit is not None else 'no'}")
    print("Records came from: metadata")
    if args.mask_source == "medsam":
        print(f"Predicted mask root: {args.pred_mask_root}")


def print_summary(frame: pd.DataFrame) -> None:
    component_counts = Counter(frame["connected_components"].astype(int))
    area_p33 = frame["area"].quantile(0.33)
    area_p66 = frame["area"].quantile(0.66)
    elongation_p75 = frame["elongation"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.75)
    compactness_p75 = frame["compactness"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.75)
    fragmented_masks = int((frame["connected_components"] != 1).sum())

    print("\nData analysis summary")
    print(f"Total mask samples: {len(frame)}")
    print(f"Total image samples: {frame['sample_name'].nunique()}")
    print(f"Mean area: {frame['area'].mean():.2f}")
    print(f"Median area: {frame['area'].median():.2f}")
    print(f"Area p33/p66: {area_p33:.2f} / {area_p66:.2f}")
    print(f"Elongation p75: {elongation_p75:.4f}")
    print(f"Compactness p75: {compactness_p75:.4f}")
    print(f"Fragmented masks: {fragmented_masks}")
    print(
        "Elongation range: "
        f"{frame['elongation'].min(skipna=True):.4f} - "
        f"{frame['elongation'].max(skipna=True):.4f}"
    )
    print("Connected components distribution:")
    for components, count in sorted(component_counts.items()):
        print(f"  {components}: {count}")
    if "medsam_dice" in frame.columns and frame["medsam_dice"].notna().any():
        scored = frame.dropna(subset=["medsam_dice"])
        worst_class = scored.groupby("category_name")["medsam_dice"].mean().idxmin()
        worst_shape_group = scored.groupby("suggested_shape_group")["medsam_dice"].mean().idxmin()
        print(f"Mean MedSAM Dice: {scored['medsam_dice'].mean():.4f}")
        print(f"Worst-performing class: {worst_class}")
        print(f"Worst-performing shape group: {worst_shape_group}")


def main() -> int:
    args = parse_args()

    if args.mask_source == "gt":
        records = load_required_jsonl(args.metadata, "GT")
    else:
        records = load_required_jsonl(args.pred_metadata, "MedSAM prediction")

    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No records found in the selected metadata file.")

    pred_records = {}
    if args.include_medsam and args.mask_source == "gt":
        pred_records = {
            record["sample_name"]: record
            for record in load_required_jsonl(args.pred_metadata, "MedSAM prediction")
        }

    print_validation(args, records, stage="before")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.mask_source == "gt":
        features = extract_gt_features(
            records=records,
            include_medsam=args.include_medsam,
            pred_records=pred_records,
            pred_mask_root=args.pred_mask_root,
        )
    else:
        features = extract_medsam_features(
            records=records,
            pred_mask_root=args.pred_mask_root,
            include_medsam=args.include_medsam,
        )

    if features.empty:
        raise RuntimeError("No masks were decoded or loaded from the selected source.")

    thresholds = threshold_suggestions(features)
    features = add_gating_columns(features, thresholds)
    thresholds_path = args.output_root / "threshold_suggestions.csv"
    thresholds.to_csv(thresholds_path, index=False)
    write_failure_summaries(features, args.output_root)

    features_path = args.output_root / "features.csv"
    features.to_csv(features_path, index=False)

    generate_plots(features, args.output_root / "plots")
    generate_examples(features, args.output_root / "examples")

    print_validation(args, records, stage="after", features=features)
    print_summary(features)
    print(f"\nSaved features: {features_path}")
    print(f"Saved thresholds: {thresholds_path}")
    print(f"Saved plots: {args.output_root / 'plots'}")
    print(f"Saved examples: {args.output_root / 'examples'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
