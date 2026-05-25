#!/usr/bin/env python3
"""Create 5-panel figures: X-ray | Ground Truth | MedSAM | MoE Refined | FlowSDF.

Each figure is cropped to the fragment bounding box and saved as a PNG.
FlowSDF panel is skipped (falls back to 4 panels) if no FlowSDF CSV/masks are available.

Usage:
    # Visualize a specific sample
    python evaluation/visualize_medsam_simple.py --sample XRAY_PENGWIN_001_0026

    # Visualize N random samples
    python evaluation/visualize_medsam_simple.py --n-random 5

    # Visualize the N best/worst MedSAM samples by Dice
    python evaluation/visualize_medsam_simple.py --n-best 5 --n-worst 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
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

COLOUR_GT       = (0.2, 0.9, 0.2)   # green
COLOUR_MEDSAM   = (0.2, 0.6, 1.0)   # blue
COLOUR_MOE      = (1.0, 0.5, 0.0)   # orange
COLOUR_FLOWSDF  = (0.9, 0.2, 0.8)   # purple
OVERLAY_ALPHA   = 0.45
MASK_SIZE       = 1024
PAD_PX          = 40


def build_image_map(metadata_path: Path) -> dict[str, Path]:
    records = load_jsonl(metadata_path)
    return {r["sample_name"]: resolve_existing_path(r["original_image_path"])
            for r in records if r.get("original_image_path")}


def load_xray(path: Path) -> np.ndarray | None:
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


def overlay(xray_gray: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float) -> np.ndarray:
    base = np.stack([xray_gray / 255.0] * 3, axis=-1)
    fg = np.array(colour, dtype=np.float32)
    return np.where(mask.astype(bool)[..., np.newaxis], (1 - alpha) * base + alpha * fg, base).astype(np.float32)


def visualize_fragment(
    row: pd.Series,
    xray_map: dict[str, Path],
    medsam_root: Path,
    moe_root: Path,
    output_dir: Path,
    flowsdf_root: Path | None = None,
    flowsdf_dice: float | None = None,
    label: str = "",
) -> None:
    sample_name   = row["sample_name"]
    instance_idx  = int(row["medsam_instance_id_medsam"]) - 1
    category_id   = int(row["category_id_medsam"])
    fragment_id   = int(row["fragment_id_medsam"])
    category_name = str(row.get("category_name_medsam", ""))
    dice_medsam   = float(row.get("dice_medsam", float("nan")))
    dice_moe      = float(row.get("dice_moe", float("nan")))
    label_path    = resolve_existing_path(str(row.get("original_label_path_medsam", "")))

    medsam_file = resolve_existing_path(medsam_root / f"{sample_name}.npz")
    moe_file    = resolve_existing_path(moe_root    / f"{sample_name}.npz")
    if not medsam_file.exists():
        print(f"  SKIP {sample_name}: MedSAM mask file missing")
        return
    if not moe_file.exists():
        print(f"  SKIP {sample_name}: MoE mask file missing")
        return
    if not label_path.exists():
        print(f"  SKIP {sample_name}: GT label missing")
        return

    medsam_mask = load_prediction_masks(medsam_file)[instance_idx]
    moe_mask    = load_prediction_masks(moe_file)[instance_idx]

    # Optional FlowSDF mask
    flowsdf_mask = None
    if flowsdf_root is not None:
        flowsdf_file = resolve_existing_path(flowsdf_root / f"{sample_name}.npz")
        if flowsdf_file.exists():
            flowsdf_mask = load_prediction_masks(flowsdf_file)[instance_idx]

    seg    = load_pengwin_label(label_path)
    gt_448 = decode_pengwin_fragment_from_record(
        seg, {"category_id": category_id, "fragment_id": fragment_id}
    )
    gt_1024 = resize_mask(gt_448, (MASK_SIZE, MASK_SIZE))

    bbox = compute_bbox(gt_1024) or compute_bbox(medsam_mask)
    if bbox is None:
        print(f"  SKIP {sample_name}: no foreground pixels")
        return

    xray_full = load_xray(xray_map.get(sample_name, Path("__missing__")))
    if xray_full is not None and xray_full.shape != (MASK_SIZE, MASK_SIZE):
        pil = Image.fromarray(xray_full).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR)
        xray_full = np.array(pil, dtype=np.uint8)

    xray_crop = crop(xray_full, bbox) if xray_full is not None else np.full((bbox[1]-bbox[0], bbox[3]-bbox[2]), 30, dtype=np.uint8)
    gt_crop   = crop(gt_1024, bbox)
    sam_crop  = crop(medsam_mask, bbox)
    moe_crop  = crop(moe_mask, bbox)

    panels = [
        np.stack([xray_crop / 255.0] * 3, axis=-1),
        overlay(xray_crop, gt_crop,  COLOUR_GT,     OVERLAY_ALPHA),
        overlay(xray_crop, sam_crop, COLOUR_MEDSAM, OVERLAY_ALPHA),
        overlay(xray_crop, moe_crop, COLOUR_MOE,    OVERLAY_ALPHA),
    ]
    titles = ["X-ray", "Ground Truth", "MedSAM", "MoE Refined"]
    xlabels: list[str | None] = [None, None, f"Dice {dice_medsam:.3f}", f"Dice {dice_moe:.3f}"]

    if flowsdf_mask is not None:
        panels.append(overlay(xray_crop, crop(flowsdf_mask, bbox), COLOUR_FLOWSDF, OVERLAY_ALPHA))
        titles.append("FlowSDF")
        xlabels.append(f"Dice {flowsdf_dice:.3f}" if flowsdf_dice is not None else None)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("white")

    for ax, panel, title, xlabel in zip(axes, panels, titles, xlabels):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="black", fontsize=15, pad=6)
        ax.axis("off")
        if xlabel:
            ax.set_xlabel(xlabel, color="black", fontsize=13)

    prefix = f"{label}__" if label else ""
    fig.suptitle(f"{sample_name}  ·  {category_name}", color="black", fontsize=13, y=1.02)
    plt.tight_layout(w_pad=0.0)
    plt.subplots_adjust(wspace=0.02)

    fname = f"{prefix}{sample_name}__frag{instance_idx:03d}__{category_name}__dice{dice_medsam:.3f}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delta-csv",         type=Path, default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_delta.csv")
    p.add_argument("--flowsdf-csv",       type=Path, default=PROJECT_ROOT / "data" / "flowsdf-moe-predictions" / "test" / "evaluation_delta.csv")
    p.add_argument("--medsam-mask-root",  type=Path, default=PROJECT_ROOT / "data" / "medsam-predictions" / "binary_masks")
    p.add_argument("--moe-mask-root",     type=Path, default=PROJECT_ROOT / "data" / "moe-predictions"    / "binary_masks")
    p.add_argument("--flowsdf-mask-root", type=Path, default=PROJECT_ROOT / "data" / "flowsdf-moe-predictions" / "test" / "binary_masks")
    p.add_argument("--metadata",          type=Path, default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl")
    p.add_argument("--output-dir",        type=Path, default=PROJECT_ROOT / "data" / "medsam-predictions" / "visualizations")
    p.add_argument("--sample",            type=str,  default=None, help="Visualize a specific sample name (all its fragments).")
    p.add_argument("--n-best",            type=int,  default=0,    help="N fragments with highest MedSAM Dice.")
    p.add_argument("--n-worst",           type=int,  default=0,    help="N fragments with lowest MedSAM Dice.")
    p.add_argument("--n-random",          type=int,  default=0,    help="N random fragments.")
    p.add_argument("--seed",              type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.delta_csv)
    xray_map = build_image_map(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load FlowSDF dice scores keyed by (sample_name, medsam_instance_id)
    flowsdf_dice_map: dict[tuple[str, int], float] = {}
    flowsdf_root: Path | None = None
    if args.flowsdf_csv.exists() and args.flowsdf_mask_root.exists():
        fdf = pd.read_csv(args.flowsdf_csv)
        for _, r in fdf.iterrows():
            key = (r["sample_name"], int(r["medsam_instance_id_moe"]))
            flowsdf_dice_map[key] = float(r["dice_moe"])
        flowsdf_root = args.flowsdf_mask_root
        print(f"FlowSDF: loaded {len(flowsdf_dice_map)} entries from {args.flowsdf_csv}")
    else:
        print("FlowSDF CSV/masks not found — will render 4 panels only.")

    selected: list[tuple[pd.Series, str]] = []

    if args.sample:
        rows = df[df["sample_name"] == args.sample]
        if rows.empty:
            print(f"Sample '{args.sample}' not found in CSV.")
            return
        for _, row in rows.iterrows():
            selected.append((row, ""))

    if args.n_best > 0:
        for _, row in df.nlargest(args.n_best, "dice_medsam").iterrows():
            selected.append((row, "best"))

    if args.n_worst > 0:
        for _, row in df.nsmallest(args.n_worst, "dice_medsam").iterrows():
            selected.append((row, "worst"))

    if args.n_random > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(df), size=min(args.n_random, len(df)), replace=False)
        for i in idx:
            selected.append((df.iloc[i], "random"))

    if not selected:
        print("Nothing selected — use --sample, --n-best, --n-worst, or --n-random.")
        return

    print(f"Generating {len(selected)} figures → {args.output_dir}\n")
    for row, label in selected:
        instance_id = int(row["medsam_instance_id_medsam"])
        flowsdf_dice = flowsdf_dice_map.get((row["sample_name"], instance_id))
        visualize_fragment(
            row, xray_map, args.medsam_mask_root, args.moe_mask_root, args.output_dir,
            flowsdf_root=flowsdf_root,
            flowsdf_dice=flowsdf_dice,
            label=label,
        )

    print(f"\nDone. {len(selected)} figures saved.")


if __name__ == "__main__":
    main()
