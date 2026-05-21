#!/usr/bin/env python3
"""Direct comparison of cnnROI vs cnnNoROI mask refinement.

Merges the two evaluation CSVs on (sample_name, fragment_index), computes
per-fragment delta (cnnROI − cnnNoROI), and generates 6-panel figures for
the most informative fragments:

    X-ray | MedSAM | cnnNoROI | cnnROI | GT | Diff(cnnROI vs cnnNoROI)

When FlowSDF inputs are provided, the figures include FlowSDF as an additional
panel:

    X-ray | MedSAM | cnnNoROI | cnnROI | FlowSDF | GT | Diff(cnnROI vs cnnNoROI)

Diff colour key:
    white = both correct   green = RoI fixed   red = RoI broke   dark = both wrong

Fragment selection (all relative to cnnROI − cnnNoROI delta):
    roi_wins   — largest positive delta_dice  (RoI crops helped most)
    roi_loses  — largest negative delta_dice  (RoI crops hurt most)
    small      — random sample from expert_small population
    large      — random sample from expert_large population

Also prints a per-group metrics table comparing cnnNoROI → cnnROI, and FlowSDF
when available.

Requires (run evaluation jobs first):
    data/moe-predictions/evaluation_moe.csv
    data/roi-predictions/evaluation_roi.csv
    data/medsam-predictions/evaluation_pengwin.csv   (for the table only)
    data/moe-predictions/binary_masks/
    data/roi-predictions/binary_masks/
    data/medsam-predictions/binary_masks/
    data/flowsdf-moe-predictions/test/binary_masks/  (optional)
    data/medsam-predictions/metadata.jsonl

Usage:
    python evaluation/compare_roi_noroi_pengwin.py
    python evaluation/compare_roi_noroi_pengwin.py --n-roi-wins 10 --n-roi-loses 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataloader_utils import (
    decode_pengwin_fragment_from_record,
    load_jsonl,
    load_pengwin_label,
    load_prediction_masks,
    resolve_existing_path,
)

# Overlay colours (R, G, B) in [0, 1]
COLOUR_MEDSAM = (0.2, 0.6, 1.0)    # blue
COLOUR_NOROI  = (0.8, 0.3, 0.9)    # purple
COLOUR_ROI    = (1.0, 0.5, 0.0)    # orange
COLOUR_FLOWSDF = (0.0, 0.85, 0.75)  # teal
COLOUR_GT     = (0.2, 0.9, 0.2)    # green
OVERLAY_ALPHA = 0.45
MASK_SIZE     = 1024
PAD_PX        = 40
MERGE_KEYS    = ["sample_name", "fragment_index"]


# ---------------------------------------------------------------------------
# Data helpers  (shared with visualize_moe_pengwin.py, duplicated to keep
# this script self-contained)
# ---------------------------------------------------------------------------

def build_image_map(metadata_path: Path) -> dict[str, Path]:
    records = load_jsonl(metadata_path)
    return {
        r["sample_name"]: resolve_existing_path(r["original_image_path"])
        for r in records
        if r.get("original_image_path")
    }


def load_xray(path: Path) -> np.ndarray | None:
    """Load X-ray as uint8 grayscale with CLAHE normalisation."""
    path = resolve_existing_path(path)
    if not path.exists():
        return None
    img = np.array(Image.open(path)).astype(np.float32)
    img += img.min() + 0.01
    img = -np.log(img)
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    return 255 - clahe.apply(img_u8)


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_hw:
        return mask
    h, w = target_hw
    pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)


def compute_bbox(mask: np.ndarray, pad: int = PAD_PX) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    return (
        max(0, int(ys.min()) - pad),
        min(h, int(ys.max()) + pad + 1),
        max(0, int(xs.min()) - pad),
        min(w, int(xs.max()) + pad + 1),
    )


def crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def xray_to_display(xray: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if xray is None:
        return np.full((h, w), 30, dtype=np.uint8)
    return np.array(Image.fromarray(xray).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def overlay(
    xray_gray: np.ndarray,
    mask: np.ndarray,
    colour: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    base   = np.stack([xray_gray / 255.0] * 3, axis=-1)
    fg     = np.array(colour, dtype=np.float32)
    mask_f = mask.astype(bool)[..., np.newaxis]
    return np.where(mask_f, (1 - alpha) * base + alpha * fg, base).astype(np.float32)


def diff_roi_vs_noroi(
    roi: np.ndarray,
    noroi: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """
    White = both correct   Green = RoI fixed   Red = RoI broke   Dark = both wrong
    """
    roi_ok   = (roi   == gt).astype(bool)
    noroi_ok = (noroi == gt).astype(bool)
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[ roi_ok &  noroi_ok] = [1.00, 1.00, 1.00]   # white — both correct
    rgb[ roi_ok & ~noroi_ok] = [0.10, 0.85, 0.10]   # green — RoI fixed
    rgb[~roi_ok &  noroi_ok] = [0.90, 0.10, 0.10]   # red   — RoI broke
    rgb[~roi_ok & ~noroi_ok] = [0.15, 0.15, 0.15]   # dark  — both wrong
    return rgb


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def print_comparison_table(merged: pd.DataFrame, medsam_csv: Path | None) -> None:
    """Print cnnNoROI → cnnROI, plus FlowSDF when available, per-group summary."""

    groups: list[tuple[str, pd.DataFrame]] = [("overall", merged)]
    has_flowsdf = "dice_flowsdf" in merged.columns

    size_col = "size_group_noroi" if "size_group_noroi" in merged.columns else None
    cat_col  = "category_name_noroi" if "category_name_noroi" in merged.columns else None

    if size_col:
        for val in ["small", "large"]:
            groups.append((val, merged[merged[size_col] == val]))
    if cat_col:
        for val in merged[cat_col].dropna().unique():
            groups.append((str(val), merged[merged[cat_col] == val]))

    header = (
        f"{'group':<16}{'n':>7}  {'dice_noroi':>10}  {'dice_roi':>8}  {'ΔR-N':>7}"
        f"  {'iou_noroi':>9}  {'iou_roi':>7}  {'ΔR-N':>7}"
    )
    if has_flowsdf:
        header += f"  {'dice_flow':>9}  {'ΔF-N':>7}  {'ΔF-R':>7}  {'iou_flow':>8}  {'ΔF-N':>7}  {'ΔF-R':>7}"
    sep    = "-" * len(header)
    title = "=== cnnROI vs cnnNoROI"
    if has_flowsdf:
        title += " vs FlowSDF"
    title += " — per-group comparison ==="
    print(f"\n{title}")
    print(header)
    print(sep)

    for label, sub in groups:
        if sub.empty:
            continue
        dn  = sub["dice_noroi"].mean()
        dr  = sub["dice_roi"].mean()
        in_ = sub["iou_noroi"].mean()
        ir  = sub["iou_roi"].mean()
        dd  = sub["delta_dice"].mean()
        di  = sub["delta_iou"].mean() if "delta_iou" in sub.columns else float("nan")
        line = (
            f"{label:<16}{len(sub):>7}  {dn:>10.4f}  {dr:>8.4f}  {dd:>+7.4f}  "
            f"{in_:>9.4f}  {ir:>7.4f}  {di:>+7.4f}"
        )
        if has_flowsdf:
            df_ = sub["dice_flowsdf"].mean()
            if_ = sub["iou_flowsdf"].mean()
            dfn = sub["delta_dice_flowsdf_vs_noroi"].mean()
            dfr = sub["delta_dice_flowsdf_vs_roi"].mean()
            ifn = sub["delta_iou_flowsdf_vs_noroi"].mean()
            ifr = sub["delta_iou_flowsdf_vs_roi"].mean()
            line += f"  {df_:>9.4f}  {dfn:>+7.4f}  {dfr:>+7.4f}  {if_:>8.4f}  {ifn:>+7.4f}  {ifr:>+7.4f}"
        print(line)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Per-fragment figure
# ---------------------------------------------------------------------------

def visualize_fragment(
    row: pd.Series,
    xray_map: dict[str, Path],
    medsam_root: Path,
    noroi_root: Path,
    roi_root: Path,
    flowsdf_root: Path | None,
    output_dir: Path,
    group_label: str,
) -> None:
    sample_name   = row["sample_name"]
    instance_idx  = int(row["medsam_instance_id_noroi"]) - 1
    category_id   = int(row["category_id_noroi"])
    fragment_id   = int(row["fragment_id_noroi"])
    category_name = str(row.get("category_name_noroi", ""))
    size_group    = str(row.get("size_group_noroi", ""))
    dice_noroi    = float(row.get("dice_noroi", float("nan")))
    dice_roi      = float(row.get("dice_roi",   float("nan")))
    dice_flowsdf  = float(row.get("dice_flowsdf", float("nan")))
    delta_dice    = float(row.get("delta_dice", float("nan")))
    label_path    = resolve_existing_path(str(row.get("original_label_path_noroi", "")))
    has_flowsdf   = flowsdf_root is not None and "dice_flowsdf" in row.index

    # --- load masks ---
    medsam_file = resolve_existing_path(medsam_root / f"{sample_name}.npz")
    noroi_file  = resolve_existing_path(noroi_root  / f"{sample_name}.npz")
    roi_file    = resolve_existing_path(roi_root    / f"{sample_name}.npz")
    flowsdf_file = resolve_existing_path(flowsdf_root / f"{sample_name}.npz") if has_flowsdf else None

    for fpath, name in [(medsam_file, "medsam"), (noroi_file, "noroi"), (roi_file, "roi")]:
        if not fpath.exists():
            print(f"  SKIP {sample_name}: {name} mask file missing ({fpath})")
            return
    if flowsdf_file is not None and not flowsdf_file.exists():
        print(f"  SKIP {sample_name}: flowsdf mask file missing ({flowsdf_file})")
        return

    medsam_mask = load_prediction_masks(medsam_file)[instance_idx]
    noroi_mask  = load_prediction_masks(noroi_file)[instance_idx]
    roi_mask    = load_prediction_masks(roi_file)[instance_idx]
    flowsdf_mask = load_prediction_masks(flowsdf_file)[instance_idx] if flowsdf_file is not None else None

    # --- load GT ---
    if not label_path.exists():
        print(f"  SKIP {sample_name}: GT label missing at {label_path}")
        return
    seg    = load_pengwin_label(label_path)
    gt_448 = decode_pengwin_fragment_from_record(
        seg, {"category_id": category_id, "fragment_id": fragment_id}
    )
    gt_1024 = resize_mask(gt_448, (MASK_SIZE, MASK_SIZE))

    # --- bounding box from GT (fall back to roi or noroi mask) ---
    bbox = compute_bbox(gt_1024)
    if bbox is None:
        bbox = compute_bbox(roi_mask) or compute_bbox(noroi_mask) or compute_bbox(medsam_mask)
    if bbox is None:
        print(f"  SKIP {sample_name}: no foreground pixels found")
        return

    # --- X-ray ---
    xray_full = load_xray(xray_map.get(sample_name, Path("__missing__")))
    if xray_full is not None and xray_full.shape != (MASK_SIZE, MASK_SIZE):
        xray_full = np.array(
            Image.fromarray(xray_full).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR),
            dtype=np.uint8,
        )
    crop_hw   = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    xray_disp = xray_to_display(
        crop(xray_full, bbox) if xray_full is not None else None,
        crop_hw,
    )

    # --- crop all masks ---
    medsam_crop = crop(medsam_mask, bbox)
    noroi_crop  = crop(noroi_mask,  bbox)
    roi_crop    = crop(roi_mask,    bbox)
    flowsdf_crop = crop(flowsdf_mask, bbox) if flowsdf_mask is not None else None
    gt_crop     = crop(gt_1024,     bbox)

    # --- panels ---
    panel_xray   = np.stack([xray_disp / 255.0] * 3, axis=-1)
    panel_medsam = overlay(xray_disp, medsam_crop, COLOUR_MEDSAM, OVERLAY_ALPHA)
    panel_noroi  = overlay(xray_disp, noroi_crop,  COLOUR_NOROI,  OVERLAY_ALPHA)
    panel_roi    = overlay(xray_disp, roi_crop,    COLOUR_ROI,    OVERLAY_ALPHA)
    panel_flowsdf = overlay(xray_disp, flowsdf_crop, COLOUR_FLOWSDF, OVERLAY_ALPHA) if flowsdf_crop is not None else None
    panel_gt     = overlay(xray_disp, gt_crop,     COLOUR_GT,     OVERLAY_ALPHA)
    panel_diff   = diff_roi_vs_noroi(roi_crop, noroi_crop, gt_crop)

    if panel_flowsdf is None:
        panels = [panel_xray, panel_medsam, panel_noroi, panel_roi, panel_gt, panel_diff]
        titles = ["X-ray", "MedSAM", "cnnNoROI", "cnnROI", "Ground truth", "Diff (RoI vs NoRoI)"]
        fig_width = 24
    else:
        panels = [panel_xray, panel_medsam, panel_noroi, panel_roi, panel_flowsdf, panel_gt, panel_diff]
        titles = ["X-ray", "MedSAM", "cnnNoROI", "cnnROI", "FlowSDF", "Ground truth", "Diff (RoI vs NoRoI)"]
        fig_width = 28

    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 4))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    axes[2].set_xlabel(f"Dice {dice_noroi:.3f}", color="#cc88ff", fontsize=9)
    axes[3].set_xlabel(f"Dice {dice_roi:.3f}  (Δ {delta_dice:+.3f})", color="#ffaa44", fontsize=9)
    if panel_flowsdf is not None:
        axes[4].set_xlabel(f"Dice {dice_flowsdf:.3f}", color="#66ffee", fontsize=9)
        diff_ax = axes[6]
    else:
        diff_ax = axes[5]

    legend_patches = [
        mpatches.Patch(color=(0.10, 0.85, 0.10), label="RoI fixed"),
        mpatches.Patch(color=(0.90, 0.10, 0.10), label="RoI broke"),
        mpatches.Patch(color=(1.00, 1.00, 1.00), label="Both correct"),
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="Both wrong"),
    ]
    diff_ax.legend(
        handles=legend_patches, loc="lower center",
        bbox_to_anchor=(0.5, -0.22), ncol=2,
        fontsize=8, framealpha=0.3, labelcolor="white",
        facecolor="#1a1a1a",
    )

    sign = "+" if delta_dice >= 0 else ""
    flow_text = f" · FlowSDF {dice_flowsdf:.3f}" if panel_flowsdf is not None else ""
    fig.suptitle(
        f"{sample_name}  ·  {category_name}  ·  {size_group}  ·  "
        f"NoRoI {dice_noroi:.3f} → RoI {dice_roi:.3f}  ({sign}{delta_dice:.3f})"
        f"{flow_text}  [{group_label}]",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout()

    fname = (
        f"{group_label}__{sample_name}__frag{instance_idx:03d}"
        f"__{category_name}__{size_group}__delta{delta_dice:+.3f}.png"
    )
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ---------------------------------------------------------------------------
# CSV merge and fragment selection
# ---------------------------------------------------------------------------

def suffix_non_key_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rename = {col: f"{col}_{suffix}" for col in df.columns if col not in MERGE_KEYS}
    return df.rename(columns=rename)


def build_comparison_df(noroi_csv: Path, roi_csv: Path, flowsdf_csv: Path | None = None) -> pd.DataFrame:
    """Merge evaluation CSVs and compute per-fragment deltas."""
    noroi = pd.read_csv(noroi_csv)
    roi   = pd.read_csv(roi_csv)

    merged = noroi.merge(roi, on=MERGE_KEYS, suffixes=("_noroi", "_roi"))

    if flowsdf_csv is not None:
        flowsdf = suffix_non_key_columns(pd.read_csv(flowsdf_csv), "flowsdf")
        merged = merged.merge(flowsdf, on=MERGE_KEYS)

    for metric in ["dice", "iou", "hd95", "assd"]:
        col_roi   = f"{metric}_roi"
        col_noroi = f"{metric}_noroi"
        if col_roi in merged.columns and col_noroi in merged.columns:
            merged[f"delta_{metric}"] = merged[col_roi] - merged[col_noroi]
        col_flowsdf = f"{metric}_flowsdf"
        if col_flowsdf in merged.columns:
            if col_noroi in merged.columns:
                merged[f"delta_{metric}_flowsdf_vs_noroi"] = merged[col_flowsdf] - merged[col_noroi]
            if col_roi in merged.columns:
                merged[f"delta_{metric}_flowsdf_vs_roi"] = merged[col_flowsdf] - merged[col_roi]

    return merged


def select_fragments(
    df: pd.DataFrame,
    n_roi_wins: int,
    n_roi_loses: int,
    n_random_small: int,
    n_random_large: int,
    seed: int = 42,
) -> list[tuple[pd.Series, str]]:
    rng     = np.random.default_rng(seed)
    valid   = df.dropna(subset=["delta_dice"])
    selected: list[tuple[pd.Series, str]] = []

    def pick(subset: pd.DataFrame, n: int, label: str, ascending: bool) -> None:
        if subset.empty or n == 0:
            return
        for _, row in subset.sort_values("delta_dice", ascending=ascending).head(n).iterrows():
            selected.append((row, label))

    pick(valid, n_roi_wins,  "roi_wins",  ascending=False)
    pick(valid, n_roi_loses, "roi_loses", ascending=True)

    size_col = "size_group_noroi" if "size_group_noroi" in df.columns else None
    if size_col:
        for sz, n, label in [("small", n_random_small, "random_small"),
                              ("large", n_random_large, "random_large")]:
            sub = valid[valid[size_col] == sz]
            if not sub.empty and n > 0:
                idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
                for i in idx:
                    selected.append((sub.iloc[i], label))

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--noroi-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_moe.csv",
                        help="cnnNoROI evaluation CSV.")
    parser.add_argument("--roi-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "roi-predictions" / "evaluation_roi.csv",
                        help="cnnROI evaluation CSV.")
    parser.add_argument("--flowsdf-csv", type=Path, default=None,
                        help="Optional FlowSDF evaluation CSV.")
    parser.add_argument("--medsam-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "evaluation_pengwin.csv",
                        help="MedSAM baseline CSV (used for the table only).")
    parser.add_argument("--medsam-mask-root", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "binary_masks")
    parser.add_argument("--noroi-mask-root", type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "binary_masks")
    parser.add_argument("--roi-mask-root", type=Path,
                        default=PROJECT_ROOT / "data" / "roi-predictions" / "binary_masks")
    parser.add_argument("--flowsdf-mask-root", type=Path, default=None,
                        help="Optional FlowSDF binary mask directory.")
    parser.add_argument("--metadata", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "roi-predictions" / "comparisons")
    parser.add_argument("--n-roi-wins",     type=int, default=5,
                        help="Fragments where cnnROI improves most over cnnNoROI.")
    parser.add_argument("--n-roi-loses",    type=int, default=5,
                        help="Fragments where cnnROI degrades most vs cnnNoROI.")
    parser.add_argument("--n-random-small", type=int, default=5)
    parser.add_argument("--n-random-large", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path, name in [(args.noroi_csv, "--noroi-csv"), (args.roi_csv, "--roi-csv")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.flowsdf_csv is not None and not args.flowsdf_csv.exists():
        raise FileNotFoundError(f"--flowsdf-csv not found: {args.flowsdf_csv}")
    if args.flowsdf_csv is not None and args.flowsdf_mask_root is None:
        raise ValueError("--flowsdf-mask-root is required when --flowsdf-csv is provided")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluation CSVs …")
    merged = build_comparison_df(args.noroi_csv, args.roi_csv, args.flowsdf_csv)
    print(f"  {len(merged)} matched fragments")

    print_comparison_table(merged, args.medsam_csv if args.medsam_csv.exists() else None)

    print("Building X-ray image map …")
    xray_map = build_image_map(args.metadata)

    fragments = select_fragments(
        merged,
        n_roi_wins=args.n_roi_wins,
        n_roi_loses=args.n_roi_loses,
        n_random_small=args.n_random_small,
        n_random_large=args.n_random_large,
        seed=args.seed,
    )
    print(f"Selected {len(fragments)} fragments — generating figures …\n")

    for row, group_label in fragments:
        print(f"[{group_label}] {row['sample_name']}")
        visualize_fragment(
            row=row,
            xray_map=xray_map,
            medsam_root=args.medsam_mask_root,
            noroi_root=args.noroi_mask_root,
            roi_root=args.roi_mask_root,
            flowsdf_root=args.flowsdf_mask_root,
            output_dir=args.output_dir,
            group_label=group_label,
        )

    print(f"\nDone. {len(fragments)} figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
