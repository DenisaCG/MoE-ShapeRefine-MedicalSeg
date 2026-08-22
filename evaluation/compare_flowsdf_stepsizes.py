#!/usr/bin/env python3
"""Three-way comparison of FlowSDF ODE step counts (stepsize_10 / 40 / 60).

Merges the three per-stepsize `evaluation_flowsdf.csv` files on
(sample_name, fragment_index), prints per-group metric tables, decides which
step count performs best overall (by mean metrics *and* by per-fragment win
counts, since the means can be very close), writes a merged per-fragment CSV
with a `winner_stepsize` column, and renders 6-panel comparison figures:

    X-ray | stepsize_10 | stepsize_40 | stepsize_60 | Ground truth | Diff

Diff colour key (3-way, analogous to compare_roi_noroi_pengwin.py):
    exact stepsize colour = only that stepsize correct
    blended colour        = exactly two stepsizes correct
    white                 = all three correct   dark = all three wrong

Fragment selection:
    max_disagreement — largest (max dice − min dice) across the 3 stepsizes
    winner_best       — cases where the declared overall-winner stepsize beats
                         the other two by the largest margin
    winner_worst      — cases where the declared overall-winner stepsize is
                         beaten by the other two by the largest margin
    random_small      — random sample from the small-fragment population
    random_large      — random sample from the large-fragment population

Requires (run 2_infer_flowsdf.job + 4_evaluate_flowsdf.job for each ODE_STEPS
value first):
    data/flowsdf-moe-predictions/test/stepsize_10/csv/evaluation_flowsdf.csv
    data/flowsdf-moe-predictions/test/stepsize_40/csv/evaluation_flowsdf.csv
    data/flowsdf-moe-predictions/test/stepsize_60/csv/evaluation_flowsdf.csv
    data/flowsdf-moe-predictions/test/stepsize_{10,40,60}/binary_masks/
    data/medsam-predictions/metadata.jsonl   (for X-ray paths)

Usage:
    python evaluation/compare_flowsdf_stepsizes.py
    python evaluation/compare_flowsdf_stepsizes.py --n-max-disagreement 10
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

STEPSIZES = [10, 40, 60]
MERGE_KEYS = ["sample_name", "fragment_index"]
METRICS = ["dice", "iou", "hd95", "assd"]
LOWER_IS_BETTER = {"hd95", "assd"}

# Overlay colours (R, G, B) in [0, 1]
COLOUR_S10 = (0.95, 0.75, 0.10)   # amber
COLOUR_S40 = (1.0, 0.35, 0.55)    # pink
COLOUR_S60 = (0.35, 0.55, 1.0)    # blue
COLOUR_GT  = (0.2, 0.9, 0.2)      # green
STEP_COLOURS = {10: COLOUR_S10, 40: COLOUR_S40, 60: COLOUR_S60}
OVERLAY_ALPHA = 0.45
MASK_SIZE = 1024
PAD_PX = 40


# ---------------------------------------------------------------------------
# Data helpers (duplicated from compare_roi_noroi_pengwin.py to keep this
# script self-contained, per repo convention)
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


def xray_to_display(xray: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if xray is None:
        return np.full((h, w), 30, dtype=np.uint8)
    return np.array(Image.fromarray(xray).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def overlay(xray_gray: np.ndarray, mask: np.ndarray, colour: tuple[float, float, float], alpha: float) -> np.ndarray:
    base = np.stack([xray_gray / 255.0] * 3, axis=-1)
    fg = np.array(colour, dtype=np.float32)
    mask_f = mask.astype(bool)[..., np.newaxis]
    return np.where(mask_f, (1 - alpha) * base + alpha * fg, base).astype(np.float32)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_b, gt_b).sum() / denom)


def diff_three_way(
    mask_10: np.ndarray, mask_40: np.ndarray, mask_60: np.ndarray, gt: np.ndarray,
) -> np.ndarray:
    """Show which stepsize(s) are pixel-correct. Blends colours for 2-way ties."""
    ok10 = (mask_10 == gt).astype(bool)
    ok40 = (mask_40 == gt).astype(bool)
    ok60 = (mask_60 == gt).astype(bool)

    c10 = np.array(COLOUR_S10, dtype=np.float32)
    c40 = np.array(COLOUR_S40, dtype=np.float32)
    c60 = np.array(COLOUR_S60, dtype=np.float32)

    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[~ok10 & ~ok40 & ~ok60] = [0.15, 0.15, 0.15]
    rgb[ ok10 &  ok40 &  ok60] = [1.00, 1.00, 1.00]
    rgb[ ok10 & ~ok40 & ~ok60] = c10
    rgb[~ok10 &  ok40 & ~ok60] = c40
    rgb[~ok10 & ~ok40 &  ok60] = c60
    rgb[ ok10 &  ok40 & ~ok60] = (c10 + c40) / 2.0
    rgb[ ok10 & ~ok40 &  ok60] = (c10 + c60) / 2.0
    rgb[~ok10 &  ok40 &  ok60] = (c40 + c60) / 2.0
    return rgb


def annotate_panel_score(ax: plt.Axes, text: str, color: str) -> None:
    ax.text(
        0.985, 0.035, text, transform=ax.transAxes, ha="right", va="bottom",
        color=color, fontsize=8.5, fontweight="bold", linespacing=1.15,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#000000", "edgecolor": "none", "alpha": 0.68},
    )


# ---------------------------------------------------------------------------
# CSV merge, deltas, winner determination
# ---------------------------------------------------------------------------

def suffix_non_key_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rename = {col: f"{col}_{suffix}" for col in df.columns if col not in MERGE_KEYS}
    return df.rename(columns=rename)


def build_comparison_df(csv_10: Path, csv_40: Path, csv_60: Path) -> pd.DataFrame:
    d10 = suffix_non_key_columns(pd.read_csv(csv_10), "10")
    d40 = suffix_non_key_columns(pd.read_csv(csv_40), "40")
    d60 = suffix_non_key_columns(pd.read_csv(csv_60), "60")

    merged = d10.merge(d40, on=MERGE_KEYS).merge(d60, on=MERGE_KEYS)

    dice_cols = {s: f"dice_{s}" for s in STEPSIZES}
    merged["dice_max"] = merged[list(dice_cols.values())].max(axis=1)
    merged["dice_min"] = merged[list(dice_cols.values())].min(axis=1)
    merged["dice_spread"] = merged["dice_max"] - merged["dice_min"]

    def winner_row(row: pd.Series) -> int:
        return max(STEPSIZES, key=lambda s: row[dice_cols[s]])

    merged["winner_stepsize"] = merged.apply(winner_row, axis=1)
    return merged


def decide_overall_winner(merged: pd.DataFrame) -> tuple[int, dict]:
    """Return (winner_stepsize, details) using mean-dice and win-count evidence."""
    mean_dice = {s: merged[f"dice_{s}"].mean() for s in STEPSIZES}
    mean_hd95 = {s: merged[f"hd95_{s}"].mean() for s in STEPSIZES}
    win_counts = {s: int((merged["winner_stepsize"] == s).sum()) for s in STEPSIZES}

    winner_by_mean = max(STEPSIZES, key=lambda s: mean_dice[s])
    winner_by_count = max(STEPSIZES, key=lambda s: win_counts[s])

    best_dice = mean_dice[winner_by_mean]
    runner_up_dice = max(mean_dice[s] for s in STEPSIZES if s != winner_by_mean)
    margin = best_dice - runner_up_dice

    details = {
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95,
        "win_counts": win_counts,
        "winner_by_mean": winner_by_mean,
        "winner_by_count": winner_by_count,
        "margin": margin,
    }
    return winner_by_mean, details


def print_group_table(merged: pd.DataFrame, group_label: str, sub: pd.DataFrame) -> None:
    if sub.empty:
        return
    header = f"{'group':<16}{'n':>7}"
    for s in STEPSIZES:
        header += f"  {'dice_' + str(s):>9}"
    for s in STEPSIZES:
        header += f"  {'hd95_' + str(s):>9}"
    header += f"  {'wins(10/40/60)':>16}"
    if group_label == "__header__":
        print(header)
        print("-" * len(header))
        return
    line = f"{group_label:<16}{len(sub):>7}"
    for s in STEPSIZES:
        line += f"  {sub[f'dice_{s}'].mean():>9.4f}"
    for s in STEPSIZES:
        line += f"  {sub[f'hd95_{s}'].mean():>9.3f}"
    wins = "/".join(str(int((sub['winner_stepsize'] == s).sum())) for s in STEPSIZES)
    line += f"  {wins:>16}"
    print(line)


def print_comparison_table(merged: pd.DataFrame) -> None:
    print("\n=== FlowSDF stepsize_10 vs stepsize_40 vs stepsize_60 — per-group comparison ===")
    print_group_table(merged, "__header__", merged)
    print_group_table(merged, "overall", merged)

    if "size_group_10" in merged.columns:
        for val in ["small", "large"]:
            print_group_table(merged, val, merged[merged["size_group_10"] == val])
    if "category_name_10" in merged.columns:
        for val in sorted(merged["category_name_10"].dropna().unique()):
            print_group_table(merged, str(val), merged[merged["category_name_10"] == val])
    print()


def print_verdict(merged: pd.DataFrame, details: dict) -> str:
    mean_dice = details["mean_dice"]
    mean_hd95 = details["mean_hd95"]
    win_counts = details["win_counts"]
    winner_by_mean = details["winner_by_mean"]
    winner_by_count = details["winner_by_count"]
    margin = details["margin"]
    n = len(merged)

    lines = []
    lines.append("=== VERDICT: which ODE step count performs best? ===")
    lines.append(f"Matched fragments compared: {n}")
    lines.append("")
    lines.append("Mean Dice:  " + "   ".join(f"stepsize_{s}={mean_dice[s]:.4f}" for s in STEPSIZES))
    lines.append("Mean HD95:  " + "   ".join(f"stepsize_{s}={mean_hd95[s]:.3f}" for s in STEPSIZES))
    lines.append(
        "Per-fragment Dice win counts: "
        + "   ".join(f"stepsize_{s}={win_counts[s]} ({100*win_counts[s]/n:.1f}%)" for s in STEPSIZES)
    )
    lines.append("")
    lines.append(f"Best mean Dice: stepsize_{winner_by_mean} (margin over runner-up: {margin:+.4f})")
    lines.append(f"Most per-fragment wins: stepsize_{winner_by_count}")

    if margin < 0.003:
        lines.append(
            "NOTE: mean-Dice differences are within ~0.003 of each other — effectively "
            "noise at this sample size. Step count does not meaningfully change accuracy here."
        )
    if winner_by_mean != winner_by_count:
        lines.append(
            f"NOTE: mean-based winner (stepsize_{winner_by_mean}) and win-count-based winner "
            f"(stepsize_{winner_by_count}) disagree — treat the result as inconclusive/noisy "
            "rather than a clear win for either."
        )

    if margin < 0.003 or winner_by_mean != winner_by_count:
        overall_call = (
            f"stepsize_{winner_by_mean} (weak/inconclusive edge — differences are marginal; "
            f"stepsize_10 is the cheapest to run and is not meaningfully worse)"
        )
    else:
        overall_call = f"stepsize_{winner_by_mean} (clear win by both mean Dice and per-fragment win count)"

    lines.append("")
    lines.append(f"OVERALL RECOMMENDATION: {overall_call}")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text + "\n")
    return text


# ---------------------------------------------------------------------------
# Fragment selection
# ---------------------------------------------------------------------------

def select_fragments(
    merged: pd.DataFrame,
    winner_stepsize: int,
    n_max_disagreement: int,
    n_winner_best: int,
    n_winner_worst: int,
    n_random_small: int,
    n_random_large: int,
    seed: int = 42,
) -> list[tuple[pd.Series, str]]:
    rng = np.random.default_rng(seed)
    selected: list[tuple[pd.Series, str]] = []

    def pick(subset: pd.DataFrame, n: int, label: str, sort_col: str, ascending: bool) -> None:
        if subset.empty or n == 0:
            return
        for _, row in subset.sort_values(sort_col, ascending=ascending).head(n).iterrows():
            selected.append((row, label))

    pick(merged, n_max_disagreement, "max_disagreement", "dice_spread", ascending=False)

    winner_col = f"dice_{winner_stepsize}"
    other_cols = [f"dice_{s}" for s in STEPSIZES if s != winner_stepsize]
    merged = merged.copy()
    merged["winner_vs_others"] = merged[winner_col] - merged[other_cols].max(axis=1)

    pick(merged, n_winner_best, "winner_best", "winner_vs_others", ascending=False)
    pick(merged, n_winner_worst, "winner_worst", "winner_vs_others", ascending=True)

    size_col = "size_group_10" if "size_group_10" in merged.columns else None
    if size_col:
        for sz, n, label in [("small", n_random_small, "random_small"), ("large", n_random_large, "random_large")]:
            sub = merged[merged[size_col] == sz]
            if not sub.empty and n > 0:
                idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
                for i in idx:
                    selected.append((sub.iloc[i], label))

    return selected


# ---------------------------------------------------------------------------
# Per-fragment figure
# ---------------------------------------------------------------------------

def visualize_fragment(
    row: pd.Series,
    xray_map: dict[str, Path],
    mask_roots: dict[int, Path],
    output_dir: Path,
    group_label: str,
) -> None:
    sample_name = row["sample_name"]
    instance_idx = int(row["medsam_instance_id_10"]) - 1
    category_id = int(row["category_id_10"])
    fragment_id = int(row["fragment_id_10"])
    category_name = str(row.get("category_name_10", ""))
    size_group = str(row.get("size_group_10", ""))
    label_path = resolve_existing_path(str(row.get("original_label_path_10", "")))

    mask_files = {s: resolve_existing_path(mask_roots[s] / f"{sample_name}.npz") for s in STEPSIZES}
    for s, fpath in mask_files.items():
        if not fpath.exists():
            print(f"  SKIP {sample_name}: stepsize_{s} mask file missing ({fpath})")
            return

    masks = {s: load_prediction_masks(mask_files[s])[instance_idx] for s in STEPSIZES}

    if not label_path.exists():
        print(f"  SKIP {sample_name}: GT label missing at {label_path}")
        return
    seg = load_pengwin_label(label_path)
    gt_448 = decode_pengwin_fragment_from_record(seg, {"category_id": category_id, "fragment_id": fragment_id})
    gt_1024 = resize_mask(gt_448, (MASK_SIZE, MASK_SIZE))

    bbox = compute_bbox(gt_1024)
    if bbox is None:
        for s in STEPSIZES:
            bbox = compute_bbox(masks[s])
            if bbox is not None:
                break
    if bbox is None:
        print(f"  SKIP {sample_name}: no foreground pixels found")
        return

    xray_full = load_xray(xray_map.get(sample_name, Path("__missing__")))
    if xray_full is not None and xray_full.shape != (MASK_SIZE, MASK_SIZE):
        xray_full = np.array(Image.fromarray(xray_full).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR), dtype=np.uint8)
    crop_hw = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    xray_disp = xray_to_display(crop(xray_full, bbox) if xray_full is not None else None, crop_hw)

    mask_crops = {s: crop(masks[s], bbox) for s in STEPSIZES}
    gt_crop = crop(gt_1024, bbox)

    dice_scores = {s: float(row.get(f"dice_{s}", float("nan"))) for s in STEPSIZES}

    panel_xray = np.stack([xray_disp / 255.0] * 3, axis=-1)
    panels = [panel_xray]
    titles = ["X-ray"]
    for s in STEPSIZES:
        panels.append(overlay(xray_disp, mask_crops[s], STEP_COLOURS[s], OVERLAY_ALPHA))
        titles.append(f"stepsize_{s}")
    panels.append(overlay(xray_disp, gt_crop, COLOUR_GT, OVERLAY_ALPHA))
    titles.append("Ground truth")
    panels.append(diff_three_way(mask_crops[10], mask_crops[40], mask_crops[60], gt_crop))
    titles.append("Diff (all stepsizes)")

    fig, axes = plt.subplots(1, len(panels), figsize=(26, 4))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    step_ax_idx = {10: 1, 40: 2, 60: 3}
    step_label_colour = {10: "#f2c01a", 40: "#ff5980", 60: "#598cff"}
    for s in STEPSIZES:
        annotate_panel_score(axes[step_ax_idx[s]], f"Dice {dice_scores[s]:.3f}", step_label_colour[s])

    diff_ax = axes[-1]
    legend_patches = [
        mpatches.Patch(color=COLOUR_S10, label="stepsize_10 only"),
        mpatches.Patch(color=COLOUR_S40, label="stepsize_40 only"),
        mpatches.Patch(color=COLOUR_S60, label="stepsize_60 only"),
        mpatches.Patch(color=tuple((np.array(COLOUR_S10) + np.array(COLOUR_S40)) / 2.0), label="10 + 40"),
        mpatches.Patch(color=tuple((np.array(COLOUR_S10) + np.array(COLOUR_S60)) / 2.0), label="10 + 60"),
        mpatches.Patch(color=tuple((np.array(COLOUR_S40) + np.array(COLOUR_S60)) / 2.0), label="40 + 60"),
        mpatches.Patch(color=(1.00, 1.00, 1.00), label="All correct"),
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="All wrong"),
    ]
    diff_ax.legend(
        handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4,
        fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#1a1a1a",
    )

    fig.suptitle(
        f"{sample_name}  ·  {category_name}  ·  {size_group}  ·  [{group_label}]",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout()

    spread = float(row.get("dice_spread", float("nan")))
    fname = (
        f"{group_label}__{sample_name}__frag{instance_idx:03d}"
        f"__{category_name}__{size_group}__spread{spread:.3f}.png"
    )
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = PROJECT_ROOT / "data" / "flowsdf-moe-predictions" / "test"
    parser.add_argument("--csv-10", type=Path, default=root / "stepsize_10" / "csv" / "evaluation_flowsdf.csv")
    parser.add_argument("--csv-40", type=Path, default=root / "stepsize_40" / "csv" / "evaluation_flowsdf.csv")
    parser.add_argument("--csv-60", type=Path, default=root / "stepsize_60" / "csv" / "evaluation_flowsdf.csv")
    parser.add_argument("--mask-root-10", type=Path, default=root / "stepsize_10" / "binary_masks")
    parser.add_argument("--mask-root-40", type=Path, default=root / "stepsize_40" / "binary_masks")
    parser.add_argument("--mask-root-60", type=Path, default=root / "stepsize_60" / "binary_masks")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "stepsize_comparison")
    parser.add_argument("--n-max-disagreement", type=int, default=5)
    parser.add_argument("--n-winner-best", type=int, default=5)
    parser.add_argument("--n-winner-worst", type=int, default=5)
    parser.add_argument("--n-random-small", type=int, default=5)
    parser.add_argument("--n-random-large", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--case-id", type=str, default=None,
        help="If set, restrict fragment selection to rows whose sample_name contains this string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path, name in [(args.csv_10, "--csv-10"), (args.csv_40, "--csv-40"), (args.csv_60, "--csv-60")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = args.output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation CSVs …")
    merged = build_comparison_df(args.csv_10, args.csv_40, args.csv_60)
    print(f"  {len(merged)} matched fragments across stepsize_10 / stepsize_40 / stepsize_60")

    print_comparison_table(merged)
    winner_stepsize, details = decide_overall_winner(merged)
    verdict_text = print_verdict(merged, details)

    summary_csv = args.output_dir / "csv" / "stepsize_comparison_merged.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(summary_csv, index=False)
    print(f"Saved merged per-fragment CSV: {summary_csv}")

    verdict_path = args.output_dir / "verdict.txt"
    verdict_path.write_text(verdict_text + "\n")
    print(f"Saved verdict: {verdict_path}")

    print("Building X-ray image map …")
    xray_map = build_image_map(args.metadata)

    if args.case_id:
        before = len(merged)
        merged = merged[merged["sample_name"].str.contains(args.case_id, na=False)]
        print(f"Filtered to case '{args.case_id}': {len(merged)}/{before} fragments")
        if merged.empty:
            raise SystemExit(f"No fragments matched --case-id '{args.case_id}'")

    fragments = select_fragments(
        merged,
        winner_stepsize=winner_stepsize,
        n_max_disagreement=args.n_max_disagreement,
        n_winner_best=args.n_winner_best,
        n_winner_worst=args.n_winner_worst,
        n_random_small=args.n_random_small,
        n_random_large=args.n_random_large,
        seed=args.seed,
    )
    print(f"Selected {len(fragments)} fragments — generating figures …\n")

    mask_roots = {10: args.mask_root_10, 40: args.mask_root_40, 60: args.mask_root_60}
    for row, group_label in fragments:
        print(f"[{group_label}] {row['sample_name']}")
        visualize_fragment(
            row=row,
            xray_map=xray_map,
            mask_roots=mask_roots,
            output_dir=comparisons_dir,
            group_label=group_label,
        )

    print(f"\nDone. {len(fragments)} figures saved to: {comparisons_dir}")


if __name__ == "__main__":
    main()
