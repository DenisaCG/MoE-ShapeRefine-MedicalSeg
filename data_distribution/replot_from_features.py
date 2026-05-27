#!/usr/bin/env python3
"""Regenerate report plots and example overlays from an existing features.csv, so we don't have to rerun the entire data exploration."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "moe_shape_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd

try:
    from .plotting import generate_examples, generate_plots
except ImportError:
    from plotting import generate_examples, generate_plots


def script_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    cwd_path = (Path.cwd() / path).expanduser().resolve()
    if cwd_path.exists():
        return cwd_path
    return (script_root().parent / path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plots and example overlays from an existing features.csv."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to the existing features.csv file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory containing plots/ and examples/ folders.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plots-only",
        action="store_true",
        help="Only regenerate plots/ and skip examples/.",
    )
    mode.add_argument(
        "--examples-only",
        action="store_true",
        help="Only regenerate examples/ and skip plots/.",
    )
    return parser.parse_args()


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    features_csv = resolve_path(args.features_csv)
    output_root = resolve_path(args.output_root)

    if not features_csv.exists():
        raise FileNotFoundError(f"features.csv not found: {features_csv}")

    frame = pd.read_csv(features_csv)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.examples_only:
        plots_dir = output_root / "plots"
        recreate_dir(plots_dir)
        generate_plots(frame, plots_dir)
        print(f"Saved plots: {plots_dir}")

    if not args.plots_only:
        examples_dir = output_root / "examples"
        recreate_dir(examples_dir)
        generate_examples(frame, examples_dir)
        print(f"Saved examples: {examples_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
