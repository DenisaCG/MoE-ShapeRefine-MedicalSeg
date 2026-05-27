#!/usr/bin/env python3
"""Analyze how MedSAM Dice changes as a function of mask area.

This script uses an existing features.csv with at least:
- area
- medsam_dice

It does not recompute masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to features.csv containing area and medsam_dice.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where threshold analysis outputs will be saved.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Dice value below which a segmentation is considered failure.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5000,
        help="Rolling window size after sorting by area.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"area", "medsam_dice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["area", "medsam_dice"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["area"] > 0]
    df = df.sort_values("area").reset_index(drop=True)
    return df


def rolling_analysis(df: pd.DataFrame, window_size: int, failure_threshold: float) -> pd.DataFrame:
    out = pd.DataFrame()
    out["area"] = df["area"]

    out["rolling_mean_dice"] = df["medsam_dice"].rolling(
        window=window_size,
        center=True,
        min_periods=max(100, window_size // 10),
    ).mean()

    out["rolling_median_dice"] = df["medsam_dice"].rolling(
        window=window_size,
        center=True,
        min_periods=max(100, window_size // 10),
    ).median()

    failure = (df["medsam_dice"] < failure_threshold).astype(float)
    out["rolling_failure_rate"] = failure.rolling(
        window=window_size,
        center=True,
        min_periods=max(100, window_size // 10),
    ).mean()

    return out.dropna()


def threshold_candidates(rolling: pd.DataFrame) -> pd.DataFrame:
    """Find interpretable area cutoffs based on rolling Dice behaviour."""
    candidates = []

    checks = [
        ("mean_dice_above_0.70", "rolling_mean_dice", 0.70, "above"),
        ("mean_dice_above_0.75", "rolling_mean_dice", 0.75, "above"),
        ("median_dice_above_0.70", "rolling_median_dice", 0.70, "above"),
        ("failure_rate_below_0.25", "rolling_failure_rate", 0.25, "below"),
        ("failure_rate_below_0.20", "rolling_failure_rate", 0.20, "below"),
        ("failure_rate_below_0.15", "rolling_failure_rate", 0.15, "below"),
    ]

    for name, column, threshold, direction in checks:
        if direction == "above":
            valid = rolling[rolling[column] >= threshold]
        else:
            valid = rolling[rolling[column] <= threshold]

        area = float(valid["area"].iloc[0]) if not valid.empty else np.nan
        candidates.append(
            {
                "criterion": name,
                "area_cutoff": area,
                "metric_column": column,
                "target_value": threshold,
            }
        )

    return pd.DataFrame(candidates)


def evaluate_cutoffs(df: pd.DataFrame, cutoffs: list[float], failure_threshold: float) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        if np.isnan(cutoff):
            continue

        small = df[df["area"] <= cutoff]
        big = df[df["area"] > cutoff]

        if small.empty or big.empty:
            continue

        rows.append(
            {
                "area_cutoff": cutoff,
                "small_count": len(small),
                "big_count": len(big),
                "small_mean_dice": small["medsam_dice"].mean(),
                "big_mean_dice": big["medsam_dice"].mean(),
                "small_median_dice": small["medsam_dice"].median(),
                "big_median_dice": big["medsam_dice"].median(),
                "small_failure_rate": (small["medsam_dice"] < failure_threshold).mean(),
                "big_failure_rate": (big["medsam_dice"] < failure_threshold).mean(),
                "failure_rate_gap": (
                    (small["medsam_dice"] < failure_threshold).mean()
                    - (big["medsam_dice"] < failure_threshold).mean()
                ),
                "mean_dice_gap": big["medsam_dice"].mean() - small["medsam_dice"].mean(),
            }
        )

    return pd.DataFrame(rows)


# def plot_rolling(df: pd.DataFrame, rolling: pd.DataFrame, output_dir: Path, failure_threshold: float) -> None:
#     fig, ax = plt.subplots(figsize=(9, 6))

#     sample = df.sample(min(len(df), 25000), random_state=0)
#     ax.scatter(
#         sample["area"],
#         sample["medsam_dice"],
#         s=5,
#         alpha=0.12,
#         label="Individual fragments (sampled)",
#     )

#     ax.plot(
#         rolling["area"],
#         rolling["rolling_mean_dice"],
#         linewidth=2.2,
#         label="Rolling mean Dice",
#     )
#     ax.plot(
#         rolling["area"],
#         rolling["rolling_median_dice"],
#         linewidth=2.2,
#         label="Rolling median Dice",
#     )

#     ax.axhline(failure_threshold, linestyle="--", linewidth=1.5, label=f"Failure threshold Dice={failure_threshold}")

#     ax.set_xscale("log")
#     ax.set_xlabel("GT mask area (foreground pixels, log scale)")
#     ax.set_ylabel("MedSAM Dice score")
#     ax.set_title("MedSAM Dice as a Function of Fragment Size")
#     ax.grid(True, alpha=0.25)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(output_dir / "rolling_dice_vs_area.png", dpi=300)
#     plt.close(fig)


def plot_failure_rate(rolling: pd.DataFrame, output_dir: Path, failure_threshold: float) -> None:
    threshold = 5402

    poster_fonts = {
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "axes.titleweight": "semibold",
        "font.size": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }

    with plt.rc_context(poster_fonts):

        fig, ax = plt.subplots(figsize=(9, 6))

        x_min = float(rolling["area"].min())
        x_max = float(rolling["area"].max())

        ax.axvspan(
            x_min,
            threshold,
            color="#FF2B2B",
            alpha=0.06,
            label="Small-fragment expert",
        )

        ax.axvspan(
            threshold,
            x_max,
            color="#00AA88",
            alpha=0.05,
            label="Large-fragment expert",
        )

        ax.plot(
            rolling["area"],
            rolling["rolling_failure_rate"],
            linewidth=2.4,
            color="#1f77b4",
            label="DICE failure rate",
        )

        ax.axvline(
            threshold,
            color="#FF2B2B",
            linestyle="--",
            linewidth=2.5,
        )

        ax.annotate(
            "Gating threshold\n5,402 pixels",
            xy=(threshold, 0.65),
            xycoords=("data", "axes fraction"),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=11,
            weight="semibold",
            color="#FF2B2B",
        )

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

        ax.set_xlabel(
            "Ground-truth fragment area (foreground pixels, log scale)"
        )

        ax.set_ylabel(
            f"MedSAM failure rate (DICE < {failure_threshold})"
        )

        ax.set_title(
            "MedSAM Failure Rate",
            pad=16,
        )

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

        ax.legend(
            frameon=True,
            loc="lower left",
            facecolor="white",
            edgecolor="lightgrey",
        )

        fig.tight_layout()
        fig.savefig(
            output_dir / "rolling_failure_rate_vs_area.png",
            dpi=300,
        )

        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.features_csv)

    rolling = rolling_analysis(
        df,
        window_size=args.window_size,
        failure_threshold=args.failure_threshold,
    )

    candidates = threshold_candidates(rolling)

    extra_cutoffs = [
        df["area"].quantile(0.10),
        df["area"].quantile(0.25),
        df["area"].quantile(0.33),
        df["area"].quantile(0.50),
        df["area"].quantile(0.66),
    ]

    all_cutoffs = list(candidates["area_cutoff"].dropna()) + extra_cutoffs
    cutoff_eval = evaluate_cutoffs(df, all_cutoffs, args.failure_threshold)

    rolling.to_csv(args.output_dir / "rolling_area_dice.csv", index=False)
    candidates.to_csv(args.output_dir / "threshold_candidates.csv", index=False)
    cutoff_eval.to_csv(args.output_dir / "threshold_candidate_evaluation.csv", index=False)

    # plot_rolling(df, rolling, args.output_dir, args.failure_threshold)
    plot_failure_rate(rolling, args.output_dir, args.failure_threshold)

    print("\nArea threshold analysis")
    print(f"Samples used: {len(df)}")
    print(f"Failure threshold: Dice < {args.failure_threshold}")
    print(f"Rolling window size: {args.window_size}")

    print("\nThreshold candidates:")
    print(candidates.to_string(index=False))

    print("\nCutoff evaluation:")
    print(
        cutoff_eval.sort_values("failure_rate_gap", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())