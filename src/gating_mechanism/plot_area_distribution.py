#!/usr/bin/env python3
"""Plot fragment mask area distributions for the gating mechanism."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataloader_utils import load_pengwin_label, pengwin_bit_shift

DEFAULT_OUT = PROJECT_ROOT / "figures" / "gating_area_distribution.png"
DEFAULT_THRESHOLD = 5402
MEDSAM_SIZE = 1024
GT_DIR = Path("/gpfs/home5/scur0509/projects/data/pengwin/original/task2_xray/train/output/images/x-ray")

CLASS_COLOURS = {
    "SA": "#7fb7ff",
    "LI": "#9c86ff",
    "RI": "#d986ff",
}


def load_records(split: str) -> pd.DataFrame:
    if split == "all":
        frames = []
        for name in ("train", "val", "test"):
            path = CSV_DIR / f"gated_{name}_records.csv"
            if path.exists():
                frame = pd.read_csv(path)
                frame["split"] = name
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No gated CSV files found in {CSV_DIR}")
        return pd.concat(frames, ignore_index=True)

    path = CSV_DIR / f"gated_{split}_records.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run gating_mechanism.py first.")
    frame = pd.read_csv(path)
    frame["split"] = split
    return frame


def add_gt_areas(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """Decode GT fragments and add a gt_area column."""
    df = df.copy()
    df["case_id_for_gt"] = df["sample_name"].astype(str).str.replace("XRAY_PENGWIN_", "", regex=False)
    df["gt_area"] = 0
    gt_pixels: int | None = None

    for case_id, case_df in df.groupby("case_id_for_gt", sort=False):
        gt = load_pengwin_label(GT_DIR / f"{case_id}.tif")
        gt_pixels = int(gt.shape[0] * gt.shape[1])

        fragments = case_df[["category_id", "fragment_id"]].drop_duplicates()
        for fragment in fragments.itertuples(index=False):
            shift = pengwin_bit_shift(
                category_id=int(fragment.category_id),
                fragment_id=int(fragment.fragment_id),
            )
            area = int(((gt >> shift) & 1).sum())
            matches = (
                (df["case_id_for_gt"] == case_id)
                & (df["category_id"].astype(int) == int(fragment.category_id))
                & (df["fragment_id"].astype(int) == int(fragment.fragment_id))
            )
            df.loc[matches, "gt_area"] = area

    df = df.drop(columns=["case_id_for_gt"])
    return df, gt_pixels


def plot_distribution(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    slide: bool,
    threshold: int,
    area_column: str,
    xlabel: str,
) -> None:
    df = df[df[area_column] > 0].copy()
    if df.empty:
        raise ValueError("No positive fragment areas found to plot.")

    bins = np.logspace(np.log10(df[area_column].min()), np.log10(df[area_column].max()), 55)

    if slide:
        fig = plt.figure(figsize=(14, 6.25), facecolor="#ddc6f2")
        fig.text(
            0.5,
            0.88,
            "Gating Mechanism",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color="black",
        )
        ax = fig.add_axes([0.52, 0.18, 0.42, 0.62], facecolor="white")
    else:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")

    for class_name in ("SA", "LI", "RI"):
        values = df.loc[df["category_name"] == class_name, area_column].to_numpy()
        if len(values) == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.62,
            label=class_name,
            color=CLASS_COLOURS[class_name],
            edgecolor="white",
            linewidth=0.25,
        )

    ax.set_xscale("log")
    ax.axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=f"Gating threshold = {threshold:,}",
        zorder=5,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of fragments")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Class", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which gated CSV split to plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--slide",
        action="store_true",
        help="Save with a lavender slide background and title.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Foreground-pixel area cutoff used by the gating mechanism.",
    )
    parser.add_argument(
        "--source",
        choices=("gt", "medsam"),
        default="gt",
        help="Plot ground-truth mask areas or MedSAM mask areas from the gated CSV.",
    )
    args = parser.parse_args()

    df = load_records(args.split)
    area_column = "area"
    threshold = args.threshold
    xlabel = "Mask area (foreground pixels, log scale)"
    title = "Distribution of Fragment Mask Area by Class"

    if args.source == "gt":
        df, gt_pixels = add_gt_areas(df)
        area_column = "gt_area"
        title = "Distribution of Ground-Truth Fragment Mask Area by Class"
        xlabel = "Ground-truth mask area (foreground pixels, log scale)"
        if gt_pixels is not None:
            threshold = round(args.threshold * gt_pixels / (MEDSAM_SIZE * MEDSAM_SIZE))

    plot_distribution(
        df,
        args.out,
        title=title,
        slide=args.slide,
        threshold=threshold,
        area_column=area_column,
        xlabel=xlabel,
    )
    print(f"Saved {args.split} {args.source} distribution plot -> {args.out}")


if __name__ == "__main__":
    main()
