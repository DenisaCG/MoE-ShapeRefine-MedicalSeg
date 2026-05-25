#!/usr/bin/env python3
"""Add score overlays to already-rendered comparison PNGs.

This is a post-processing helper for old comparison figures when the binary
masks are no longer available. It reads Dice scores from evaluation CSVs,
matches each image by sample name and fragment index, and draws compact labels
in the lower-right corner of the MedSAM, cnnROI, cnnNoROI, and FlowSDF panels.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PANEL_ORDER = ["X-ray", "MedSAM", "cnnNoROI", "cnnROI", "FlowSDF", "Ground truth", "Diff"]
METHOD_COLORS = {
    "MedSAM": (51, 153, 255),
    "cnnNoROI": (204, 77, 230),
    "cnnROI": (255, 128, 0),
    "FlowSDF": (0, 217, 191),
}
METHOD_TO_CSV_KEY = {
    "MedSAM": "medsam",
    "cnnNoROI": "noroi",
    "cnnROI": "roi",
    "FlowSDF": "flowsdf",
}


FILENAME_RE = re.compile(
    r"^(?P<group>.+?)__"
    r"(?P<sample>XRAY_PENGWIN_\d+_\d+)__"
    r"frag(?P<fragment_index>\d+)__"
    r"(?P<category>[^_]+)__"
    r"(?P<size_group>[^_]+)__"
    r"delta(?P<delta>[+-]\d+(?:\.\d+)?)\.png$"
)


def load_scores(csv_path: Path) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["sample_name"], int(row["fragment_index"]))
                scores[key] = float(row["dice"])
            except (KeyError, TypeError, ValueError):
                continue
    return scores


def parse_image_name(path: Path) -> tuple[str, int]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Could not parse sample/fragment from image name: {path.name}\n"
            "Expected names like: random_large__XRAY_PENGWIN_011_0118__frag000__SA__large__delta-0.053.png"
        )
    return match.group("sample"), int(match.group("fragment_index"))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    pad: int,
) -> None:
    x_right, y_bottom = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0 = x_right - text_w - 2 * pad
    y0 = y_bottom - text_h - 2 * pad
    x1 = x_right
    y1 = y_bottom

    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(4, pad), fill=(0, 0, 0, 174))
    draw.multiline_text((x0 + pad, y0 + pad), text, fill=color + (255,), font=font, spacing=3)


def contiguous_segments(active: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(active)))
    return segments


def longest_segment(segments: list[tuple[int, int]], min_len: int) -> tuple[int, int] | None:
    valid = [segment for segment in segments if segment[1] - segment[0] >= min_len]
    if not valid:
        return None
    return max(valid, key=lambda segment: segment[1] - segment[0])


def detect_panel_image_boxes(
    image: Image.Image,
    panel_count: int,
    background_tolerance: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Detect the displayed image rectangle inside each panel cell.

    The PNG includes title text, legends, and figure background around the actual
    panel image. We first split the figure into rough panel cells, then find the
    largest dense non-background rectangle inside each cell.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width = arr.shape[:2]
    background = arr[0, 0]
    foreground = np.any(np.abs(arr - background) > background_tolerance, axis=2)
    boxes: list[tuple[int, int, int, int]] = []

    for panel_idx in range(panel_count):
        cell_x0 = round(panel_idx * width / panel_count)
        cell_x1 = round((panel_idx + 1) * width / panel_count)
        cell = foreground[:, cell_x0:cell_x1]
        cell_width = cell_x1 - cell_x0

        row_density = cell.mean(axis=1)
        row_threshold = max(0.12, float(row_density.max()) * 0.45)
        row_segment = longest_segment(
            contiguous_segments(row_density > row_threshold),
            min_len=max(30, round(height * 0.20)),
        )

        if row_segment is None:
            # Conservative fallback to the rough panel cell.
            boxes.append((cell_x0, 0, cell_x1, height))
            continue

        y0, y1 = row_segment
        image_rows = cell[y0:y1, :]
        col_density = image_rows.mean(axis=0)
        col_threshold = max(0.12, float(col_density.max()) * 0.45)
        col_segment = longest_segment(
            contiguous_segments(col_density > col_threshold),
            min_len=max(30, round(cell_width * 0.20)),
        )

        if col_segment is None:
            boxes.append((cell_x0, y0, cell_x1, y1))
            continue

        x0, x1 = col_segment
        boxes.append((cell_x0 + x0, y0, cell_x0 + x1, y1))

    return boxes


def annotate_image(
    image_path: Path,
    out_path: Path,
    scores_by_method: dict[str, dict[tuple[str, int], float]],
    panel_order: list[str],
    font_size: int,
    image_inset_px: int,
) -> None:
    sample_name, fragment_index = parse_image_name(image_path)
    key = (sample_name, fragment_index)

    method_scores: dict[str, float] = {}
    missing: list[str] = []
    for method, csv_key in METHOD_TO_CSV_KEY.items():
        score = scores_by_method[csv_key].get(key)
        if score is None:
            missing.append(method)
        else:
            method_scores[method] = score

    if missing:
        raise KeyError(
            f"Missing scores for {sample_name} fragment {fragment_index}: {', '.join(missing)}"
        )

    medsam_score = method_scores["MedSAM"]

    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(font_size)
    pad = max(6, round(font_size * 0.45))
    panel_count = len(panel_order)
    panel_boxes = detect_panel_image_boxes(image, panel_count)

    for panel_idx, method in enumerate(panel_order):
        if method not in method_scores:
            continue
        score = method_scores[method]
        delta = score - medsam_score
        label = f"Dice {score:.3f}\nΔMedSAM {delta:+.3f}"
        _, _, x1, y1 = panel_boxes[panel_idx]
        draw_label(
            draw=draw,
            xy=(x1 - image_inset_px, y1 - image_inset_px),
            text=label,
            color=METHOD_COLORS[method],
            font=font,
            pad=pad,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)
    print(f"saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", default=[],
                        help="Rendered comparison PNG to annotate. Can be repeated.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Optionally annotate all PNGs in this directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for annotated PNGs. Default: <image-dir>/annotated.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite input PNGs instead of writing annotated copies.")
    parser.add_argument("--panel-order", default=",".join(DEFAULT_PANEL_ORDER),
                        help="Comma-separated panel order in the existing PNG.")
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--image-inset-px", type=int, default=10,
                        help="Inset from the detected lower-right image corner.")
    parser.add_argument("--medsam-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "evaluation_pengwin.csv")
    parser.add_argument("--noroi-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_moe.csv")
    parser.add_argument("--roi-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "roi-predictions" / "evaluation_roi.csv")
    parser.add_argument("--flowsdf-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "flowsdf-moe-predictions" / "test" / "evaluation_flowsdf.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = list(args.image)
    if args.input_dir is not None:
        images.extend(sorted(args.input_dir.glob("*.png")))
    if not images:
        raise ValueError("Provide at least one --image or --input-dir.")

    for csv_path in [args.medsam_csv, args.noroi_csv, args.roi_csv, args.flowsdf_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

    scores_by_method = {
        "medsam": load_scores(args.medsam_csv),
        "noroi": load_scores(args.noroi_csv),
        "roi": load_scores(args.roi_csv),
        "flowsdf": load_scores(args.flowsdf_csv),
    }
    panel_order = [name.strip() for name in args.panel_order.split(",") if name.strip()]

    for image_path in images:
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if args.overwrite:
            out_path = image_path
        elif args.output_dir is not None:
            out_path = args.output_dir / image_path.name
        else:
            out_path = image_path.parent / "annotated" / image_path.name
        annotate_image(
            image_path=image_path,
            out_path=out_path,
            scores_by_method=scores_by_method,
            panel_order=panel_order,
            font_size=args.font_size,
            image_inset_px=args.image_inset_px,
        )


if __name__ == "__main__":
    main()
