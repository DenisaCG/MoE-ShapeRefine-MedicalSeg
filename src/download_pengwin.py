#!/usr/bin/env python3
"""Download the official PENGWIN training data into the shared workspace data folder.

This script stores the canonical dataset copy at:
    <workspace>/data/pengwin

MedSAM can keep using its existing relative paths via a symlink:
    <workspace>/MedSAM/data -> ../data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT / "data" / "pengwin"

DOWNLOADS = {
    "ct": [
        (
            "PENGWIN_CT_train_images_part1.zip",
            "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part1.zip?download=1",
        ),
        (
            "PENGWIN_CT_train_images_part2.zip",
            "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part2.zip?download=1",
        ),
        (
            "PENGWIN_CT_train_labels.zip",
            "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_labels.zip?download=1",
        ),
    ],
    "xray": [
        (
            "train.zip",
            "https://zenodo.org/records/10913196/files/train.zip?download=1",
        ),
        (
            "pengwin_utils.py",
            "https://zenodo.org/records/10913196/files/pengwin_utils.py?download=1",
        ),
        (
            "requirements.txt",
            "https://zenodo.org/records/10913196/files/requirements.txt?download=1",
        ),
    ],
}

RAW_ROOTS = {
    "ct": DATA_ROOT / "raw" / "task1_ct",
    "xray": DATA_ROOT / "raw" / "task2_xray",
}


def get_extract_root(task: str, filename: str) -> Path:
    if task == "ct":
        if "labels" in filename:
            return DATA_ROOT / "original" / "task1_ct" / "labels"
        return DATA_ROOT / "original" / "task1_ct" / "images"
    return DATA_ROOT / "original" / "task2_xray"


def is_valid_download(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".zip":
        return zipfile.is_zipfile(path)
    return path.stat().st_size > 0


def download_file(url: str, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if is_valid_download(destination):
            print(f"skip download: {destination}")
            return
        print(f"re-downloading invalid file: {destination}")

    temp_destination = destination.with_name(destination.name + ".part")
    if temp_destination.exists():
        temp_destination.unlink()

    print(f"downloading: {destination.name}")
    with urllib.request.urlopen(url) as response, temp_destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    temp_destination.replace(destination)


def extract_zip(archive_path: Path, destination: Path, force: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f".{archive_path.stem}.extracted"
    if marker.exists() and not force:
        print(f"skip extract: {archive_path.name}")
        return

    print(f"extracting: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    marker.write_text(f"{archive_path.name}\n", encoding="utf-8")


def create_directories() -> None:
    (DATA_ROOT / "raw" / "task1_ct").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "raw" / "task2_xray").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "original" / "task1_ct" / "images").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "original" / "task1_ct" / "labels").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "original" / "task2_xray").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "derived" / "medsam").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "derived" / "moe").mkdir(parents=True, exist_ok=True)


def create_medsam_symlink() -> None:
    medsam_root = WORKSPACE_ROOT / "MedSAM"
    data_link = medsam_root / "data"
    target = Path("..") / "data"

    if data_link.is_symlink():
        if data_link.readlink() == target:
            print(f"symlink exists: {data_link} -> {target}")
            return
        data_link.unlink()
    elif data_link.exists():
        raise RuntimeError(
            f"cannot create symlink because {data_link} already exists and is not a symlink"
        )

    data_link.symlink_to(target, target_is_directory=True)
    print(f"created symlink: {data_link} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "ct", "xray"),
        default="all",
        help="Which official training data split to download.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only create directories and symlink setup.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download archives but do not unpack them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files and re-extract archives even if markers exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    create_directories()
    create_medsam_symlink()

    tasks = ("ct", "xray") if args.task == "all" else (args.task,)
    for task in tasks:
        raw_root = RAW_ROOTS[task]
        for filename, url in DOWNLOADS[task]:
            destination = raw_root / filename
            if not args.skip_download:
                download_file(url, destination, force=args.force)
            if (
                destination.suffix == ".zip"
                and not args.skip_extract
                and destination.exists()
            ):
                extract_root = get_extract_root(task, filename)
                extract_zip(destination, extract_root, force=args.force)

    print(f"shared data root: {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
