#!/usr/bin/env python3
"""Interactive viewer for the shared PENGWIN dataset.

Supports:
- Task 1 CT volumes in .mha format
- Task 2 X-ray .tif images with encoded segmentation overlays

Examples:
    python scripts/view_pengwin.py --task xray
    python scripts/view_pengwin.py --task ct
    python scripts/view_pengwin.py --task xray --case 001_0000
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider
from PIL import Image

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency for CT viewing
    sitk = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT / "data" / "pengwin" / "original"

XRAY_IMAGE_ROOT = DATA_ROOT / "task2_xray" / "train" / "input" / "images" / "x-ray"
XRAY_LABEL_ROOT = DATA_ROOT / "task2_xray" / "train" / "output" / "images" / "x-ray"
CT_IMAGE_ROOT = DATA_ROOT / "task1_ct" / "images"
CT_LABEL_ROOT = DATA_ROOT / "task1_ct" / "labels"

XRAY_CATEGORIES = {
    1: "SA",
    2: "LI",
    3: "RI",
}
XRAY_COLORS = {
    1: np.array([0.93, 0.34, 0.23]),
    2: np.array([0.17, 0.63, 0.17]),
    3: np.array([0.21, 0.49, 0.86]),
}


@dataclass
class CaseData:
    image: np.ndarray
    overlay: np.ndarray | None
    title: str
    legend: str
    slice_index: int = 0
    max_slice: int = 0


def list_xray_cases() -> list[str]:
    return sorted(path.stem for path in XRAY_IMAGE_ROOT.glob("*.tif"))


def list_ct_cases() -> list[str]:
    return sorted(path.stem for path in CT_IMAGE_ROOT.glob("*.mha"))


def decode_xray_segmentation(segmentation: np.ndarray) -> tuple[np.ndarray, list[str]]:
    overlay = np.zeros(segmentation.shape + (4,), dtype=np.float32)
    legend_items: list[str] = []
    for category_id, label in XRAY_CATEGORIES.items():
        category_mask = np.zeros(segmentation.shape, dtype=bool)
        for fragment_id in range(1, 11):
            shift = 10 * (category_id - 1) + fragment_id
            mask = ((segmentation >> shift) & 1).astype(bool)
            if np.any(mask):
                category_mask |= mask
                legend_items.append(f"{label}-{fragment_id}")
        if np.any(category_mask):
            overlay[category_mask, :3] = XRAY_COLORS[category_id]
            overlay[category_mask, 3] = 1.0
    return overlay, legend_items


def normalize_xray_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    image = -np.log(image + 1e-2)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    return image


def load_xray_case(case_id: str) -> CaseData:
    image_path = XRAY_IMAGE_ROOT / f"{case_id}.tif"
    label_path = XRAY_LABEL_ROOT / f"{case_id}.tif"
    image = np.array(Image.open(image_path))
    segmentation = np.array(Image.open(label_path))
    overlay, legend_items = decode_xray_segmentation(segmentation)
    return CaseData(
        image=normalize_xray_image(image),
        overlay=overlay,
        title=f"Task 2 X-ray: {case_id}",
        legend=", ".join(legend_items) if legend_items else "No labels",
    )


def load_ct_case(case_id: str, slice_index: int | None = None) -> CaseData:
    if sitk is None:
        raise RuntimeError(
            "CT viewing requires SimpleITK. Install it in your environment first, "
            "for example with: pip install SimpleITK"
        )

    image_path = CT_IMAGE_ROOT / f"{case_id}.mha"
    label_path = CT_LABEL_ROOT / f"{case_id}.mha"
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.int32)

    image = np.clip(image, -1000, 1500)
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom

    max_slice = int(image.shape[0] - 1)
    if slice_index is None:
        nonzero_slices = np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]
        slice_index = int(nonzero_slices[len(nonzero_slices) // 2]) if len(nonzero_slices) else max_slice // 2
    slice_index = max(0, min(slice_index, max_slice))

    overlay = np.zeros(labels.shape + (4,), dtype=np.float32)
    label_mask = labels > 0
    overlay[label_mask, :3] = np.array([0.95, 0.29, 0.20], dtype=np.float32)
    overlay[label_mask, 3] = 1.0

    label_values = np.unique(labels)
    label_values = label_values[label_values > 0]
    legend = f"Labels present: {len(label_values)}"
    if len(label_values):
        legend += f" | ids: {', '.join(map(str, label_values[:12]))}"
        if len(label_values) > 12:
            legend += ", ..."

    return CaseData(
        image=image,
        overlay=overlay,
        title=f"Task 1 CT: {case_id}",
        legend=legend,
        slice_index=slice_index,
        max_slice=max_slice,
    )


class PengwinViewer:
    def __init__(self, task: str, case_id: str | None = None, overlay_alpha: float = 0.35):
        self.task = task
        self.case_ids = list_xray_cases() if task == "xray" else list_ct_cases()
        if not self.case_ids:
            raise RuntimeError(f"No cases found for task '{task}'.")

        self.case_index = self.case_ids.index(case_id) if case_id in self.case_ids else 0
        self.overlay_visible = True
        self.overlay_alpha = overlay_alpha
        self.slice_index = 0

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.06, 0.17, 0.72, 0.76])
        self.ax.set_axis_off()

        self.case_text = self.fig.text(0.06, 0.95, "", fontsize=14, fontweight="bold")
        self.legend_text = self.fig.text(0.06, 0.92, "", fontsize=10)
        self.help_text = self.fig.text(
            0.06,
            0.03,
            "Keys: left/right previous/next case | up/down previous/next slice | o toggle overlay",
            fontsize=9,
        )

        self.prev_ax = self.fig.add_axes([0.82, 0.84, 0.12, 0.05])
        self.next_ax = self.fig.add_axes([0.82, 0.77, 0.12, 0.05])
        self.prev_btn = Button(self.prev_ax, "Prev Case")
        self.next_btn = Button(self.next_ax, "Next Case")
        self.prev_btn.on_clicked(self.on_prev_case)
        self.next_btn.on_clicked(self.on_next_case)

        self.toggle_ax = self.fig.add_axes([0.82, 0.64, 0.12, 0.10])
        self.toggle = CheckButtons(self.toggle_ax, ["Overlay"], [self.overlay_visible])
        self.toggle.on_clicked(self.on_toggle_overlay)

        self.alpha_ax = self.fig.add_axes([0.82, 0.56, 0.12, 0.03])
        self.alpha_slider = Slider(self.alpha_ax, "Alpha", 0.0, 1.0, valinit=self.overlay_alpha)
        self.alpha_slider.on_changed(self.on_alpha_change)

        self.slice_slider = None
        if self.task == "ct":
            self.slice_ax = self.fig.add_axes([0.06, 0.09, 0.72, 0.03])
            self.slice_slider = Slider(self.slice_ax, "Slice", 0, 1, valinit=0, valstep=1)
            self.slice_slider.on_changed(self.on_slice_change)

        self.image_artist = None
        self.overlay_artist = None
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.load_current_case()

    def load_current_case(self) -> None:
        case_id = self.case_ids[self.case_index]
        if self.task == "xray":
            self.case = load_xray_case(case_id)
        else:
            self.case = load_ct_case(case_id, self.slice_index)
            self.slice_index = self.case.slice_index
            assert self.slice_slider is not None
            self.slice_slider.valmax = self.case.max_slice
            self.slice_slider.ax.set_xlim(self.slice_slider.valmin, self.slice_slider.valmax)
            self.slice_slider.set_val(self.slice_index)
        self.render()

    def current_image(self) -> np.ndarray:
        if self.task == "xray":
            return self.case.image
        return self.case.image[self.slice_index]

    def current_overlay(self) -> np.ndarray | None:
        if self.case.overlay is None:
            return None
        if self.task == "xray":
            return self.case.overlay
        return self.case.overlay[self.slice_index]

    def render(self) -> None:
        image = self.current_image()
        overlay = self.current_overlay()

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(image, cmap="gray")
            if overlay is not None:
                self.overlay_artist = self.ax.imshow(overlay, interpolation="nearest")
        else:
            self.image_artist.set_data(image)
            if overlay is not None:
                if self.overlay_artist is None:
                    self.overlay_artist = self.ax.imshow(overlay, interpolation="nearest")
                else:
                    self.overlay_artist.set_data(overlay)

        if self.overlay_artist is not None:
            self.overlay_artist.set_visible(self.overlay_visible and overlay is not None)
            self.overlay_artist.set_alpha(self.overlay_alpha)

        title = self.case.title
        if self.task == "ct":
            title += f" | slice {self.slice_index + 1}/{self.case.max_slice + 1}"
        self.case_text.set_text(title)
        self.legend_text.set_text(self.case.legend)
        self.ax.set_title(self.case_ids[self.case_index], fontsize=11)
        self.fig.canvas.draw_idle()

    def step_case(self, delta: int) -> None:
        self.case_index = (self.case_index + delta) % len(self.case_ids)
        self.load_current_case()

    def step_slice(self, delta: int) -> None:
        if self.task != "ct":
            return
        self.slice_index = max(0, min(self.slice_index + delta, self.case.max_slice))
        assert self.slice_slider is not None
        self.slice_slider.set_val(self.slice_index)

    def on_prev_case(self, _event) -> None:
        self.step_case(-1)

    def on_next_case(self, _event) -> None:
        self.step_case(1)

    def on_toggle_overlay(self, _label: str) -> None:
        self.overlay_visible = not self.overlay_visible
        self.render()

    def on_alpha_change(self, value: float) -> None:
        self.overlay_alpha = value
        self.render()

    def on_slice_change(self, value: float) -> None:
        if self.task != "ct":
            return
        self.slice_index = int(value)
        self.render()

    def on_key_press(self, event) -> None:
        if event.key == "left":
            self.step_case(-1)
        elif event.key == "right":
            self.step_case(1)
        elif event.key == "up":
            self.step_slice(1)
        elif event.key == "down":
            self.step_slice(-1)
        elif event.key == "o":
            self.overlay_visible = not self.overlay_visible
            if self.toggle.get_status()[0] != self.overlay_visible:
                self.toggle.set_active(0)
            else:
                self.render()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("xray", "ct"), default="xray")
    parser.add_argument("--case", type=str, default=None, help="Case id without extension.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Initial overlay opacity.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer = PengwinViewer(task=args.task, case_id=args.case, overlay_alpha=args.alpha)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
