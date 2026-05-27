from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

CLASS_COLORS = {
    "SA": "#3bbdd4",
    "LI": "#1f2ed0",
    "RI": "#b955e7",
    "unknown": "#eb1eaa",
}
GATING_POSTER_CLASS_COLORS = {
    "SA": "#BB3AC9",
    "LI": "#05789B",
    "RI": "#1ECDA7",
}
CLASS_ORDER = ("SA", "LI", "RI", "unknown")
FIGSIZE = (8.5, 5.5)
SCATTER_FIGSIZE = (7.5, 5.8)
BAR_FIGSIZE = (8.5, 5.2)
SAVE_DPI = 300


plt.rcParams.update(
    {
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "semibold",
        "font.size": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def _ordered_categories(categories: pd.Series | pd.Index | list[str]) -> list[str]:
    names = [str(category) for category in categories]
    ordered = [category for category in CLASS_ORDER if category in names]
    ordered.extend(sorted(category for category in names if category not in CLASS_ORDER))
    return ordered


def _hist_bins(values: pd.Series, bins: int, log_x: bool) -> int | np.ndarray:
    if not log_x or values.empty:
        return bins
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum <= 0 or maximum <= minimum:
        return bins
    return np.logspace(np.log10(minimum), np.log10(maximum), bins + 1)


def _finish_plot(fig: plt.Figure, path: Path, caption: str | None = None) -> None:
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=10, color="#4d4d4d")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)


def save_histogram(
    series: pd.Series,
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
    log_x: bool = False,
    caption: str | None = None,
) -> None:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if log_x:
        values = values[values > 0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(
        values,
        bins=_hist_bins(values, bins, log_x),
        color="#356d8c",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of fragments")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish_plot(fig, path, caption)


def save_histogram_per_class(
    frame: pd.DataFrame,
    column: str,
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
    log_x: bool = False,
    caption: str | None = None,
) -> None:
    values = frame[[column, "category_name"]].replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=[column])
    values["category_name"] = values["category_name"].fillna("unknown").astype(str)
    if log_x:
        values = values[values[column] > 0]
    if values.empty:
        return

    if column == "elongation" and not log_x:
        x_max = float(values[column].quantile(0.995))
        if np.isfinite(x_max):
            values = values[values[column] <= x_max]
    elif column == "compactness":
        x_max = float(values[column].quantile(0.995))
        if np.isfinite(x_max):
            values = values[values[column] <= x_max]

    if values.empty:
        return

    report_fonts = {
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "axes.titleweight": "semibold",
        "font.size": 14,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    with plt.rc_context(report_fonts):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        hist_bins = _hist_bins(values[column], bins, log_x)

        for category_name in _ordered_categories(values["category_name"].unique()):
            group = values.loc[values["category_name"] == category_name]
            if group.empty:
                continue
            counts, edges = np.histogram(group[column], bins=hist_bins)

            # Smooth counts slightly
            window = 3
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
            kernel /= kernel.sum()

            smooth_counts = np.convolve(counts, kernel, mode="same")

            # Bin centers
            centers = 0.5 * (edges[:-1] + edges[1:])

            ax.plot(
                centers,
                smooth_counts,
                label=str(category_name),
                color=GATING_POSTER_CLASS_COLORS.get(str(category_name), "#8a8a8a"),
                linewidth=3.2,
            )

        if log_x:
            ax.set_xscale("log")
        ax.set_title(title, pad=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of fracture masks")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.legend(title="Class", frameon=False, loc="upper right")
        _finish_plot(fig, path, caption)


def save_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
    log_y: bool = False,
    caption: str | None = None,
) -> None:
    values = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if log_x:
        values = values[values[x] > 0]
    if log_y:
        values = values[values[y] > 0]

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.scatter(values[x], values[y], s=12, alpha=0.28, color="#4f7aa8", edgecolors="none")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish_plot(fig, path, caption)


def save_boxplot_by_class(
    frame: pd.DataFrame,
    column: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    values = frame[[column, "category_name"]].replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=[column])
    values["category_name"] = values["category_name"].fillna("unknown").astype(str)
    if values.empty:
        return
    categories = _ordered_categories(values["category_name"].unique())
    data = [values.loc[values["category_name"] == category, column] for category in categories]

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    box = ax.boxplot(data, labels=categories, showfliers=False, patch_artist=True)
    for patch, category in zip(box["boxes"], categories):
        patch.set_facecolor(CLASS_COLORS.get(category, "#8a8a8a"))
        patch.set_alpha(0.65)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish_plot(fig, path)


def plot_class_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["category_name"].fillna("unknown").astype(str).value_counts()
    categories = _ordered_categories(counts.index)
    counts = counts.reindex(categories)
    colors = [CLASS_COLORS.get(category, "#8a8a8a") for category in categories]

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Number of Fragments per Class", pad=12)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of fragments")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish_plot(fig, path)


def plot_connected_components(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["connected_components"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(counts.index.astype(str), counts.values, color="#356d8c")
    ax.set_title("Distribution of Connected Components", pad=12)
    ax.set_xlabel("Number of disconnected mask components")
    ax.set_ylabel("Number of fragments")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish_plot(fig, path)


def plot_connected_components_per_class(frame: pd.DataFrame, path: Path) -> None:
    values = frame[["connected_components", "category_name"]].replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["connected_components"])
    values["category_name"] = values["category_name"].fillna("unknown").astype(str)
    if values.empty:
        return

    bin_labels = ["1", "2", "3-5", "6-10", "11-20", ">20"]
    bins = [0, 1, 2, 5, 10, 20, np.inf]
    values["component_group"] = pd.cut(
        values["connected_components"],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True,
    )
    values = values.dropna(subset=["component_group"])
    if values.empty:
        return

    counts = pd.crosstab(
        values["component_group"],
        values["category_name"],
    ).reindex(index=bin_labels, fill_value=0)
    categories = _ordered_categories(counts.columns)
    counts = counts.reindex(columns=categories, fill_value=0)

    report_fonts = {
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "axes.titleweight": "semibold",
        "font.size": 14,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    with plt.rc_context(report_fonts):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        x = np.arange(len(bin_labels))
        bar_width = 0.8 / max(len(categories), 1)
        offsets = (np.arange(len(categories)) - (len(categories) - 1) / 2) * bar_width

        for offset, category in zip(offsets, categories):
            ax.bar(
                x + offset,
                counts[category].values,
                width=bar_width,
                label=str(category),
                color=GATING_POSTER_CLASS_COLORS.get(str(category), "#8a8a8a"),
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_title("Connected Components by Class", pad=14)
        ax.set_xlabel("Number of disconnected mask components")
        ax.set_ylabel("Number of fracture masks")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.legend(title="Class", frameon=False, loc="upper right")
        _finish_plot(fig, path)


def plot_gating_area_threshold_poster(frame: pd.DataFrame, path: Path) -> None:
    from scipy.stats import gaussian_kde

    values = frame[["area", "category_name"]].replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["area"])
    values = values[values["area"] > 0].copy()
    values["category_name"] = values["category_name"].fillna("unknown").astype(str)

    if values.empty:
        return

    threshold = 5402

    x_min = 500
    x_max = float(values["area"].quantile(0.995))

    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = float(values["area"].max())

    if x_max <= x_min:
        x_max = x_min * 10

    # Evaluate KDE in log-space
    x_log = np.linspace(np.log10(x_min), np.log10(x_max), 1000)
    x = 10 ** x_log

    poster_fonts = {
        "axes.labelsize": 22,
        "axes.titlesize": 28,
        "axes.titleweight": "semibold",
        "font.size": 20,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }

    with plt.rc_context(poster_fonts):
        fig, ax = plt.subplots(figsize=(13, 7.5))

        target_peak = 4000

        for category_name in ("SA", "LI", "RI"):
            group = values.loc[
                values["category_name"] == category_name,
                "area",
            ]

            if len(group) < 2:
                continue

            kde = gaussian_kde(np.log10(group.values))
            density = kde(x_log)

            # Scale density to histogram-like counts
            density = density / density.max() * target_peak

            color = GATING_POSTER_CLASS_COLORS[category_name]

            ax.plot(
                x,
                density,
                linewidth=4,
                color=color,
                label=category_name,
            )

            ax.fill_between(
                x,
                density,
                alpha=0.18,
                color=color,
            )

        # Threshold line
        ax.axvline(
            threshold,
            color="#FF2B2B",
            linestyle="--",
            linewidth=4,
        )

        ax.annotate(
            "MoE gating threshold\n5,402 pixels",
            xy=(threshold, 0.95),
            xycoords=("data", "axes fraction"),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=18,
            weight="semibold",
            color="#FF2B2B",
        )

        # Background regions
        ax.axvspan(x_min, threshold, alpha=0.05, color="#FF2B2B")
        ax.axvspan(threshold, x_max, alpha=0.03, color="#00AA88")

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

        # keep histogram-style y-axis
        ax.set_ylim(0, 5000)

        ax.set_title(
            "Fragment Size Distribution per Class",
            pad=20,
        )

        ax.set_xlabel(
            "Ground-truth fragment area (foreground pixels, log scale)"
        )

        ax.set_ylabel(
            "Number of fracture masks"
        )

        ax.grid(True, alpha=0.22, linewidth=1.0)
        ax.set_axisbelow(True)

        # Move legend to top-left
        ax.legend(
            title="Fragment class",
            frameon=False,
            loc="upper left",
        )

        _finish_plot(fig, path)


def plot_medsam_trends(frame: pd.DataFrame, plots_dir: Path) -> None:
    if "medsam_dice" not in frame.columns:
        return
    scored = frame.dropna(subset=["medsam_dice"])
    if scored.empty:
        return
    save_scatter(
        scored,
        "area",
        "medsam_dice",
        plots_dir / "medsam_dice_vs_area.png",
        "MedSAM Dice Score vs Fragment Size",
        "GT mask area (foreground pixels, log scale)",
        "Dice score",
        log_x=True,
    )
    save_scatter(
        scored,
        "elongation",
        "medsam_dice",
        plots_dir / "medsam_dice_vs_elongation.png",
        "MedSAM Dice Score vs Shape Elongation",
        "GT elongation (major axis / minor axis)",
        "Dice score",
    )
    save_scatter(
        scored,
        "compactness",
        "medsam_dice",
        plots_dir / "medsam_dice_vs_compactness.png",
        "MedSAM Dice Score vs Shape Complexity",
        "GT compactness = perimeter² / area (log scale)",
        "Dice score",
        log_x=True,
    )
    save_scatter(
        scored,
        "connected_components",
        "medsam_dice",
        plots_dir / "medsam_dice_vs_connected_components.png",
        "MedSAM Dice Score vs Fragmentation",
        "Number of connected components",
        "Dice score",
    )
    save_boxplot_by_class(
        scored,
        "medsam_dice",
        plots_dir / "medsam_dice_per_class.png",
        "MedSAM Dice Score by Class",
        "Dice score",
    )


def plot_class_size_flow(frame: pd.DataFrame, path: Path) -> None:
    threshold = 5402

    values = frame[["area", "category_name"]].replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["area"])
    values = values[values["area"] > 0].copy()
    values["category_name"] = values["category_name"].fillna("unknown").astype(str)
    values = values[values["category_name"].isin(["SA", "LI", "RI"])]

    if values.empty:
        return

    values["size_group"] = np.where(
        values["area"] <= threshold,
        f"Small\n≤ {threshold:,} px",
        f"Large\n> {threshold:,} px",
    )

    total = len(values)
    class_counts = values["category_name"].value_counts().reindex(["SA", "LI", "RI"]).fillna(0)
    size_counts = (
        values.groupby(["category_name", "size_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["SA", "LI", "RI"])
    )

    class_colors = {
        "SA": "#BB3AC9",
        "LI": "#05789B",
        "RI": "#1ECDA7",
    }

    size_colors = {
        f"Small\n≤ {threshold:,} px": "#FF6B6B",
        f"Large\n> {threshold:,} px": "#4ECDC4",
    }

    def draw_flow(ax, x0, y0_top, y0_bottom, x1, y1_top, y1_bottom, color, alpha=0.35):
        verts = [
            (x0, y0_top),
            ((x0 + x1) / 2, y0_top),
            ((x0 + x1) / 2, y1_top),
            (x1, y1_top),
            (x1, y1_bottom),
            ((x0 + x1) / 2, y1_bottom),
            ((x0 + x1) / 2, y0_bottom),
            (x0, y0_bottom),
            (x0, y0_top),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        patch = PathPatch(
            Path(verts, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
        ax.add_patch(patch)

    poster_fonts = {
        "axes.titlesize": 25,
        "axes.titleweight": "semibold",
        "font.size": 17,
    }

    with plt.rc_context(poster_fonts):
        fig, ax = plt.subplots(figsize=(13, 7))

        x_total = 0.05
        x_class = 0.45
        x_size = 0.85
        bar_width = 0.035
        scale = 0.82 / total
        y_top_start = 0.92

        # Total bar
        total_height = total * scale
        total_y_top = y_top_start
        total_y_bottom = total_y_top - total_height
        ax.add_patch(Rectangle((x_total, total_y_bottom), bar_width, total_height, color="#888888"))
        ax.text(
            x_total - 0.015,
            (total_y_top + total_y_bottom) / 2,
            f"{total:,}\nFragments",
            ha="right",
            va="center",
            fontsize=18,
            weight="semibold",
        )

        # Class bars
        class_positions = {}
        y_cursor = y_top_start
        gap = 0.035

        total_segment_cursor = total_y_top

        for cls in ["SA", "LI", "RI"]:
            count = int(class_counts.loc[cls])
            height = count * scale
            y_top = y_cursor
            y_bottom = y_top - height
            class_positions[cls] = (y_top, y_bottom)

            draw_flow(
                ax,
                x_total + bar_width,
                total_segment_cursor,
                total_segment_cursor - height,
                x_class,
                y_top,
                y_bottom,
                class_colors[cls],
                alpha=0.28,
            )

            ax.add_patch(Rectangle((x_class, y_bottom), bar_width, height, color=class_colors[cls]))
            ax.text(
                x_class + bar_width + 0.012,
                (y_top + y_bottom) / 2,
                f"{cls}\n{count:,}",
                ha="left",
                va="center",
                fontsize=17,
                weight="semibold",
            )

            total_segment_cursor -= height
            y_cursor = y_bottom - gap

        # Size bars on the right
        small_label = f"Small\n≤ {threshold:,} px"
        large_label = f"Large\n> {threshold:,} px"

        final_counts = values["size_group"].value_counts().reindex([small_label, large_label]).fillna(0)
        size_positions = {}

        y_cursor = y_top_start
        for group in [small_label, large_label]:
            count = int(final_counts.loc[group])
            height = count * scale
            y_top = y_cursor
            y_bottom = y_top - height
            size_positions[group] = [y_top, y_bottom, y_top]

            ax.add_patch(Rectangle((x_size, y_bottom), bar_width, height, color=size_colors[group]))
            ax.text(
                x_size + bar_width + 0.012,
                (y_top + y_bottom) / 2,
                f"{group}\n{count:,}",
                ha="left",
                va="center",
                fontsize=17,
                weight="semibold",
            )

            y_cursor = y_bottom - 0.08

        # Class -> size flows
        for cls in ["SA", "LI", "RI"]:
            class_top, class_bottom = class_positions[cls]
            class_cursor = class_top

            for group in [small_label, large_label]:
                count = int(size_counts.loc[cls, group]) if group in size_counts.columns else 0
                if count == 0:
                    continue

                height = count * scale
                src_top = class_cursor
                src_bottom = class_cursor - height
                class_cursor -= height

                target_top = size_positions[group][2]
                target_bottom = target_top - height
                size_positions[group][2] = target_bottom

                draw_flow(
                    ax,
                    x_class + bar_width,
                    src_top,
                    src_bottom,
                    x_size,
                    target_top,
                    target_bottom,
                    class_colors[cls],
                    alpha=0.30,
                )

        ax.text(x_total + bar_width / 2, 0.98, "All fragments", ha="center", va="bottom", fontsize=18, weight="semibold")
        ax.text(x_class + bar_width / 2, 0.98, "Anatomical class", ha="center", va="bottom", fontsize=18, weight="semibold")
        ax.text(x_size + bar_width / 2, 0.98, "Gating group", ha="center", va="bottom", fontsize=18, weight="semibold")

        ax.set_title("Fragment Distribution by Class and Size-Based Gating", pad=20)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axis("off")

        _finish_plot(fig, path)


def generate_plots(frame: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_histogram(
        frame["area"],
        plots_dir / "area_distribution.png",
        "Distribution of Fragment Mask Area",
        "Mask area (foreground pixels, log scale)",
        log_x=True,
    )
    save_histogram(
        frame["aspect_ratio"],
        plots_dir / "aspect_ratio_distribution.png",
        "Distribution of Bounding Box Aspect Ratio",
        "Bounding box aspect ratio (width / height)",
    )
    save_histogram(
        frame["elongation"],
        plots_dir / "elongation_distribution.png",
        "Distribution of Shape Elongation",
        "Elongation (major axis / minor axis)",
    )
    plot_connected_components(frame, plots_dir / "connected_components_distribution.png")
    save_scatter(
        frame,
        "area",
        "elongation",
        plots_dir / "area_vs_elongation.png",
        "Mask Size vs Shape Elongation",
        "Mask area (foreground pixels, log scale)",
        "Elongation (major axis / minor axis)",
        log_x=True,
    )
    save_scatter(
        frame,
        "area",
        "compactness",
        plots_dir / "area_vs_compactness.png",
        "Mask Size vs Shape Complexity",
        "Mask area (foreground pixels, log scale)",
        "Compactness = perimeter² / area (log scale)",
        log_x=True,
        log_y=True,
    )
    plot_class_distribution(frame, plots_dir / "class_distribution.png")
    save_histogram_per_class(
        frame,
        "area",
        plots_dir / "area_distribution_per_class.png",
        "Distribution of Fragment Mask Area by Class",
        "Mask area (foreground pixels, log scale)",
        log_x=True,
    )
    plot_gating_area_threshold_poster(frame, plots_dir / "gating_area_threshold_poster.png")
    save_histogram_per_class(
        frame,
        "elongation",
        plots_dir / "elongation_distribution_per_class.png",
        "Distribution of Shape Elongation by Class",
        "Elongation (major axis / minor axis)",
    )
    save_histogram_per_class(
        frame,
        "compactness",
        plots_dir / "compactness_distribution_per_class.png",
        "Distribution of Shape Complexity by Class",
        "Compactness = perimeter² / area (log scale)",
        log_x=True,
    )
    plot_connected_components_per_class(frame, plots_dir / "connected_components_per_class.png")
    plot_medsam_trends(frame, plots_dir)
    plot_class_size_flow(frame, plots_dir / "class_size_gating_flow.png")


def normalize_xray_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= image.min() - 1e-2
    image = -np.log(image)
    lo, hi = float(image.min()), float(image.max())
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return image


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return transform.resize(
        mask,
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(mask.dtype)


def load_row_mask(row: pd.Series) -> np.ndarray | None:
    if row["mask_source"] == "medsam":
        mask_path = Path(str(row.get("mask_path", "")))
        if not mask_path.exists():
            return None
        masks = np.load(mask_path)["masks"]
        mask_index = int(row["mask_index"])
        if mask_index >= masks.shape[0]:
            return None
        return masks[mask_index].astype(np.uint8)

    label_path = Path(str(row.get("label_path", "")))
    if not label_path.exists():
        return None
    segmentation = np.array(Image.open(label_path))
    shift = 10 * (int(row["category_id"]) - 1) + int(row["fragment_id"])
    return ((segmentation >> shift) & 1).astype(np.uint8)


def save_overlay(row: pd.Series, output_path: Path) -> None:
    mask = load_row_mask(row)
    if mask is None:
        return
    mask_bool = mask.astype(bool)
    image_path = Path(str(row.get("image_path", "")))

    plt.figure(figsize=(6, 6))
    if image_path.exists():
        image = normalize_xray_image(np.array(Image.open(image_path)))
        image = resize_nearest(image, mask.shape[:2])
        plt.imshow(image, cmap="gray")
        overlay = np.zeros(mask_bool.shape + (4,), dtype=np.float32)
        overlay[mask_bool] = (0.95, 0.18, 0.12, 0.45)
        plt.imshow(overlay)
        plt.contour(mask_bool, colors=["#ffd166"], linewidths=0.8)
    else:
        plt.imshow(mask_bool, cmap="gray")

    plt.title(
        f"{row['case_id']} {row['category_name']}-{row['fragment_id']} "
        f"area={int(row['area'])} elong={row['elongation']:.2f} "
        f"cc={int(row['connected_components'])}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()


def save_example_group(rows: pd.DataFrame, examples_dir: Path, prefix: str) -> None:
    group_dir = examples_dir / prefix
    group_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        category = str(row.get("category_name", "unknown"))
        fragment_id = row.get("fragment_id", "unknown")
        filename = f"{rank:02d}_{row['sample_name']}_{category}_{fragment_id}.png"
        save_overlay(row, group_dir / filename)


def generate_examples(frame: pd.DataFrame, examples_dir: Path) -> None:
    examples_dir.mkdir(parents=True, exist_ok=True)
    valid_area = frame[frame["area"] > 0]
    save_example_group(valid_area.nsmallest(5, "area"), examples_dir, "smallest_masks")
    save_example_group(valid_area.nlargest(5, "area"), examples_dir, "largest_masks")
    save_example_group(frame.dropna(subset=["elongation"]).nlargest(5, "elongation"), examples_dir, "most_elongated")
    save_example_group(frame.nlargest(5, "connected_components"), examples_dir, "most_fragmented")
    save_example_group(frame.nlargest(5, "compactness"), examples_dir, "most_complex_shapes")
