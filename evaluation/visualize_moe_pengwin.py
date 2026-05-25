#!/usr/bin/env python3
"""Visualize MoE refinement vs MedSAM baseline for selected fragments.

Requires (run evaluate_moe_pengwin.py --delta first):
    data/moe-predictions/evaluation_delta.csv
    data/moe-predictions/binary_masks/
    data/medsam-predictions/binary_masks/
    data/medsam-predictions/metadata.jsonl

Selects fragments by four criteria and saves one PNG per fragment:
    best   — largest Dice improvement  (MoE >> MedSAM)
    worst  — largest Dice regression   (MoE << MedSAM)
    small  — random from expert_small population
    large  — random from expert_large population

Each figure shows 5 panels, all cropped to the fragment bounding box:
    X-ray  |  MedSAM overlay  |  MoE overlay  |  GT overlay  |  Diff

Diff colour key:
    white = both correct   green = MoE fixed   red = MoE broke   dark = both wrong

Usage:
    python evaluation/visualize_moe_pengwin.py
    python evaluation/visualize_moe_pengwin.py --n-best 10 --n-worst 5 --output-dir my_dir
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
COLOUR_MEDSAM = (0.2, 0.6, 1.0)   # blue
COLOUR_MOE    = (1.0, 0.5, 0.0)   # orange
COLOUR_GT     = (0.2, 0.9, 0.2)   # green
OVERLAY_ALPHA = 0.45
MASK_SIZE     = 1024               # native resolution of predicted masks
PAD_PX        = 40                 # padding around bounding box (pixels at MASK_SIZE)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_image_map(metadata_path: Path) -> dict[str, Path]:
    """sample_name → raw X-ray path."""
    records = load_jsonl(metadata_path)
    result: dict[str, Path] = {}
    for r in records:
        img = r.get("original_image_path")
        if img:
            result[r["sample_name"]] = resolve_existing_path(img)
    return result


def load_xray(path: Path) -> np.ndarray | None:
    """Load X-ray as uint8 grayscale (H, W) with CLAHE normalisation.

    Raw PENGWIN X-rays are float32 DRRs — direct uint8 cast gives a black
    image.  The pipeline mirrors pengwin_utils.visualize_drr:
      1. negative-log transform + normalise to [0, 1]
      2. convert to uint8
      3. CLAHE (clipLimit=4, tileGridSize=8×8)
      4. invert (bone → bright)
    """
    path = resolve_existing_path(path)
    if not path.exists():
        return None

    img = np.array(Image.open(path)).astype(np.float32)

    # Negative-log transform + normalise to [0, 1]
    img += img.min() + 0.01
    img = -np.log(img)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)

    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)

    # Invert so bone is bright on dark background
    return 255 - img_u8


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize binary mask with nearest-neighbour interpolation."""
    if mask.shape == target_hw:
        return mask
    h, w = target_hw
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)


def compute_bbox(mask: np.ndarray, pad: int = PAD_PX) -> tuple[int, int, int, int] | None:
    """Return (y0, y1, x0, x1) bounding box of non-zero pixels with padding."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def xray_to_display(xray: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray:
    """Return uint8 (H, W) grayscale, resized to hw. Blank if unavailable."""
    h, w = hw
    if xray is None:
        return np.full((h, w), 30, dtype=np.uint8)
    pil = Image.fromarray(xray).resize((w, h), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def overlay(xray_gray: np.ndarray, mask: np.ndarray, colour: tuple[float, float, float], alpha: float) -> np.ndarray:
    """Overlay binary mask on grayscale X-ray. Returns RGB float32 in [0,1]."""
    base = np.stack([xray_gray / 255.0] * 3, axis=-1)   # (H, W, 3)
    fg   = np.array(colour, dtype=np.float32)
    mask_f = mask.astype(bool)[..., np.newaxis]
    return np.where(mask_f, (1 - alpha) * base + alpha * fg, base).astype(np.float32)


def diff_image(moe: np.ndarray, medsam: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    White = both correct   Green = MoE fixed   Red = MoE broke   Dark = both wrong
    All inputs are binary (0/1) at the same resolution.
    """
    moe_ok  = (moe  == gt).astype(bool)
    sam_ok  = (medsam == gt).astype(bool)

    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[ moe_ok &  sam_ok] = [1.00, 1.00, 1.00]   # white
    rgb[ moe_ok & ~sam_ok] = [0.10, 0.85, 0.10]   # green  — MoE fixed
    rgb[~moe_ok &  sam_ok] = [0.90, 0.10, 0.10]   # red    — MoE broke
    rgb[~moe_ok & ~sam_ok] = [0.15, 0.15, 0.15]   # dark   — both wrong
    return rgb


def annotate_panel_score(ax: plt.Axes, text: str, color: str) -> None:
    """Draw a compact score label inside the lower-right corner of a panel."""
    ax.text(
        0.985,
        0.035,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=color,
        fontsize=9,
        fontweight="bold",
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#000000",
            "edgecolor": "none",
            "alpha": 0.68,
        },
    )


# ---------------------------------------------------------------------------
# Per-fragment figure
# ---------------------------------------------------------------------------

def visualize_fragment(
    row: pd.Series,
    xray_map: dict[str, Path],
    medsam_root: Path,
    moe_root: Path,
    output_dir: Path,
    group_label: str,
    method_label: str,
) -> None:
    sample_name      = row["sample_name"]
    instance_idx     = int(row["medsam_instance_id_moe"]) - 1   # 0-based
    category_id      = int(row["category_id_moe"])
    fragment_id      = int(row["fragment_id_moe"])
    category_name    = str(row.get("category_name_moe", ""))
    size_group       = str(row.get("size_group_moe", ""))
    dice_medsam      = float(row.get("dice_medsam", float("nan")))
    dice_moe         = float(row.get("dice_moe",    float("nan")))
    delta_dice       = float(row.get("delta_dice",  float("nan")))
    label_path       = resolve_existing_path(str(row.get("original_label_path_moe", "")))

    # --- load masks ---
    medsam_file = resolve_existing_path(medsam_root / f"{sample_name}.npz")
    moe_file    = resolve_existing_path(moe_root    / f"{sample_name}.npz")
    if not medsam_file.exists() or not moe_file.exists():
        print(f"  SKIP {sample_name}: mask file missing")
        return

    medsam_mask = load_prediction_masks(medsam_file)[instance_idx]   # (1024, 1024)
    moe_mask    = load_prediction_masks(moe_file)[instance_idx]      # (1024, 1024)

    # --- load GT ---
    if not label_path.exists():
        print(f"  SKIP {sample_name}: GT label missing at {label_path}")
        return
    seg    = load_pengwin_label(label_path)
    gt_448 = decode_pengwin_fragment_from_record(
        seg, {"category_id": category_id, "fragment_id": fragment_id}
    )                                                                 # (448, 448)
    gt_1024 = resize_mask(gt_448, (MASK_SIZE, MASK_SIZE))            # (1024, 1024)

    # --- bounding box (computed from GT at 1024 scale) ---
    bbox = compute_bbox(gt_1024)
    if bbox is None:
        # Fall back to predicted mask if GT is empty
        bbox = compute_bbox(moe_mask) or compute_bbox(medsam_mask)
    if bbox is None:
        print(f"  SKIP {sample_name}: no foreground pixels found")
        return

    # --- load and crop X-ray ---
    xray_full  = load_xray(xray_map.get(sample_name, Path("__missing__")))
    crop_hw    = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    xray_crop  = xray_to_display(
        crop(xray_full, bbox) if xray_full is not None and xray_full.shape == (MASK_SIZE, MASK_SIZE)
        else xray_full,
        crop_hw,
    )

    # If X-ray is a different size, resize to MASK_SIZE first
    if xray_full is not None and xray_full.shape != (MASK_SIZE, MASK_SIZE):
        pil = Image.fromarray(xray_full).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR)
        xray_full = np.array(pil, dtype=np.uint8)
        xray_crop = crop(xray_full, bbox)

    # --- crop masks ---
    medsam_crop = crop(medsam_mask, bbox)
    moe_crop    = crop(moe_mask,    bbox)
    gt_crop     = crop(gt_1024,     bbox)

    # --- build panels ---
    xray_disp   = xray_crop if xray_full is not None else np.full(crop_hw, 30, dtype=np.uint8)
    panel_xray  = np.stack([xray_disp / 255.0] * 3, axis=-1)
    panel_medsam = overlay(xray_disp, medsam_crop, COLOUR_MEDSAM, OVERLAY_ALPHA)
    panel_moe    = overlay(xray_disp, moe_crop,    COLOUR_MOE,    OVERLAY_ALPHA)
    panel_gt     = overlay(xray_disp, gt_crop,     COLOUR_GT,     OVERLAY_ALPHA)
    panel_diff   = diff_image(moe_crop, medsam_crop, gt_crop)

    # --- figure ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("#1a1a1a")

    panels = [panel_xray, panel_medsam, panel_moe, panel_gt, panel_diff]
    titles = ["X-ray", "MedSAM", method_label, "Ground truth", "Diff"]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11, pad=4)
        ax.axis("off")

    # Metric annotations inside method panels, so they remain visible per image.
    annotate_panel_score(axes[1], f"Dice {dice_medsam:.3f}\nΔ {0.0:+.3f}", "#88bbff")
    annotate_panel_score(axes[2], f"Dice {dice_moe:.3f}\nΔ {delta_dice:+.3f}", "#ffaa44")

    # Diff legend
    legend_patches = [
        mpatches.Patch(color=(0.10, 0.85, 0.10), label="MoE fixed"),
        mpatches.Patch(color=(0.90, 0.10, 0.10), label="MoE broke"),
        mpatches.Patch(color=(1.00, 1.00, 1.00), label="Both correct"),
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="Both wrong"),
    ]
    axes[4].legend(
        handles=legend_patches, loc="lower center",
        bbox_to_anchor=(0.5, -0.22), ncol=2,
        fontsize=8, framealpha=0.3, labelcolor="white",
        facecolor="#1a1a1a",
    )

    suptitle = (
        f"{sample_name}  ·  {category_name}  ·  {size_group}  ·  "
        f"[{group_label}]"
    )
    fig.suptitle(suptitle, color="white", fontsize=11, y=1.01)
    plt.tight_layout()

    fname = f"{group_label}__{sample_name}__frag{instance_idx:03d}__{category_name}__{size_group}__delta{delta_dice:+.3f}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Fragment selection
# ---------------------------------------------------------------------------

def select_fragments(
    df: pd.DataFrame,
    n_best: int,
    n_worst: int,
    n_random_small: int,
    n_random_large: int,
    seed: int = 42,
) -> list[tuple[pd.Series, str]]:
    rng = np.random.default_rng(seed)
    selected: list[tuple[pd.Series, str]] = []

    valid = df.dropna(subset=["delta_dice"])

    def pick(subset: pd.DataFrame, n: int, label: str, sort_col: str = "delta_dice", ascending: bool = True) -> None:
        if subset.empty or n == 0:
            return
        ordered = subset.sort_values(sort_col, ascending=ascending).head(n)
        for _, row in ordered.iterrows():
            selected.append((row, label))

    # Best improvements
    pick(valid, n_best,  "best",  "delta_dice", ascending=False)
    # Worst regressions
    pick(valid, n_worst, "worst", "delta_dice", ascending=True)

    # Random small
    size_col = "size_group_moe" if "size_group_moe" in df.columns else None
    if size_col and n_random_small > 0:
        small = valid[valid[size_col] == "small"]
        if not small.empty:
            idx = rng.choice(len(small), size=min(n_random_small, len(small)), replace=False)
            for i in idx:
                selected.append((small.iloc[i], "random_small"))

    # Random large
    if size_col and n_random_large > 0:
        large = valid[valid[size_col] == "large"]
        if not large.empty:
            idx = rng.choice(len(large), size=min(n_random_large, len(large)), replace=False)
            for i in idx:
                selected.append((large.iloc[i], "random_large"))

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delta-csv",      type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_delta.csv")
    parser.add_argument("--medsam-mask-root", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "binary_masks")
    parser.add_argument("--moe-mask-root",  type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "binary_masks")
    parser.add_argument("--metadata",       type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl")
    parser.add_argument("--output-dir",     type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "visualizations")
    parser.add_argument("--method-label",   default="MoE refined",
                        help="Title for the refined-mask panel.")
    parser.add_argument("--n-best",         type=int, default=5,
                        help="Number of best-improvement fragments.")
    parser.add_argument("--n-worst",        type=int, default=5,
                        help="Number of worst-regression fragments.")
    parser.add_argument("--n-random-small", type=int, default=5,
                        help="Number of random small fragments.")
    parser.add_argument("--n-random-large", type=int, default=5,
                        help="Number of random large fragments.")
    parser.add_argument("--seed",           type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.delta_csv.exists():
        raise FileNotFoundError(
            f"Delta CSV not found: {args.delta_csv}\n"
            "Run: python evaluation/evaluate_moe_pengwin.py --delta"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading delta CSV: {args.delta_csv}")
    df = pd.read_csv(args.delta_csv)
    print(f"  {len(df)} fragments")

    print("Building X-ray image map …")
    xray_map = build_image_map(args.metadata)

    fragments = select_fragments(
        df,
        n_best=args.n_best,
        n_worst=args.n_worst,
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
            moe_root=args.moe_mask_root,
            output_dir=args.output_dir,
            group_label=group_label,
            method_label=args.method_label,
        )

    print(f"\nDone. {len(fragments)} figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
