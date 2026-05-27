from __future__ import annotations

import numpy as np
from skimage import measure


def compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    y_idx, x_idx = np.where(mask > 0)
    if y_idx.size == 0:
        return 0, 0, 0, 0, 0, 0
    x_min = int(x_idx.min())
    y_min = int(y_idx.min())
    x_max = int(x_idx.max()) + 1
    y_max = int(y_idx.max()) + 1
    return x_min, y_min, x_max, y_max, x_max - x_min, y_max - y_min


def largest_component_regionprops(labeled: np.ndarray) -> list:
    labels = labeled[labeled > 0]
    if labels.size == 0:
        return []
    counts = np.bincount(labels)
    counts[0] = 0
    largest_label = int(counts.argmax())
    largest_mask = (labeled == largest_label).astype(np.uint8)
    return measure.regionprops(largest_mask)


def extract_mask_features(mask: np.ndarray, image_shape: tuple[int, int]) -> dict:
    area = int(mask.sum())
    x_min, y_min, x_max, y_max, width, height = compute_bbox(mask)
    aspect_ratio = float(width / height) if height > 0 else np.nan
    relative_area = float(area / (image_shape[0] * image_shape[1]))

    binary = mask.astype(bool)
    perimeter = float(measure.perimeter(binary, neighborhood=8)) if area > 0 else 0.0
    compactness = float((perimeter**2) / area) if area > 0 else np.nan

    labeled = measure.label(binary, connectivity=2)
    num_components = int(labeled.max())

    props = largest_component_regionprops(labeled)
    if props:
        prop = props[0]
        major_axis = float(prop.major_axis_length)
        minor_axis = float(prop.minor_axis_length)
        elongation = float(major_axis / minor_axis) if minor_axis > 0 else np.nan
        eccentricity = float(prop.eccentricity)
    else:
        major_axis = minor_axis = elongation = eccentricity = np.nan

    return {
        "area": area,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "relative_area": relative_area,
        "perimeter": perimeter,
        "compactness": compactness,
        "major_axis_length": major_axis,
        "minor_axis_length": minor_axis,
        "elongation": elongation,
        "eccentricity": eccentricity,
        "connected_components": num_components,
    }


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)
