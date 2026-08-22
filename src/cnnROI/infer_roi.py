#!/usr/bin/env python3
"""MoE inference with RoI crops (cnnROI).

For each image in the gated CSV:
    1. Load shared embedding (256, 64, 64) once.
    2. Load all MedSAM binary masks (N, 1024, 1024) once.
    3. For each fragment:
       a. Extract bbox from MedSAM binary mask at 1024 scale.
       b. Crop embedding (projected to 64×64 scale) → resize to ROI_SIZE.
       c. Crop binary mask to bbox → resize to ROI_SIZE → compute SDF on crop.
       d. Run CNNExpert → (1, ROI_SIZE, ROI_SIZE).
       e. Upsample crop to bbox dimensions → paste into blank 1024×1024 canvas.
    4. Stack refined masks → (N, 1024, 1024) and save as .npz.

Fallback for empty MedSAM masks: global resize to ROI_SIZE, upsample output to
1024×1024 (identical to cnnNoROI behaviour).

Output format matches MedSAM .npz convention (key "masks") so
evaluate_medsam_pengwin.py can read it unchanged.

Usage:
    python infer_roi.py [--split test] [OPTIONS]

    python infer_roi.py --split val --limit 10   # smoke test
    python infer_roi.py --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader_utils import load_prediction_masks, resolve_existing_path   # noqa: E402
from sdf_utils import sdf_channel_from_mask                                  # noqa: E402
from cnnNoROI.cnnMoE import CNNExpert                                        # noqa: E402
from cnnROI.roi_utils import (                                               # noqa: E402
    crop_to_roi, extract_bbox, paste_into_canvas, scale_bbox,
)
# --------------------------------------------------------------------------

FEAT_SIZE   = 64     # MedSAM embedding spatial resolution (fixed)
OUTPUT_SIZE = 1024   # final output mask resolution
ROI_SIZE    = 64     # default RoI crop size — same as cnnNoROI internal grid
ROI_PAD     = 64     # default bbox padding at 1024 scale


def load_expert(checkpoint_path: Path, device: torch.device) -> CNNExpert:
    model = CNNExpert(c_in=258).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def refine_fragment(
    embedding:   torch.Tensor,   # (256, 64, 64) on device, float32
    binary_mask: np.ndarray,     # (1024, 1024) uint8/float
    model:       CNNExpert,
    device:      torch.device,
    roi_size:    int = ROI_SIZE,
    roi_pad:     int = ROI_PAD,
) -> np.ndarray:
    """Refine one fragment mask with RoI crop. Returns (1024, 1024) uint8."""
    bbox = extract_bbox(binary_mask, pad=roi_pad)

    mask_t = (
        torch.from_numpy(binary_mask.astype(np.float32))
        .unsqueeze(0).unsqueeze(0).to(device)
    )                                                              # (1, 1, 1024, 1024)

    if bbox is None:
        # Empty MedSAM prediction — fall back to global resize (cnnNoROI behaviour)
        mask_small = F.interpolate(mask_t, size=(roi_size, roi_size), mode="nearest")
        emb_crop   = F.interpolate(
            embedding.unsqueeze(0), size=(roi_size, roi_size),
            mode="bilinear", align_corners=False,
        )
        sdf_np = sdf_channel_from_mask(mask_small[0, 0].cpu().numpy())
        sdf    = torch.from_numpy(sdf_np).unsqueeze(0).to(device)
        x      = torch.cat([emb_crop, mask_small, sdf], dim=1)
        pred   = model(x)
        pred_up = F.interpolate(
            pred, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False,
        )
        return (pred_up[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    # --- crop embedding at 64×64 scale ---
    emb_crop = crop_to_roi(
        embedding,
        scale_bbox(bbox, 1024, FEAT_SIZE),
        roi_size, mode="bilinear",
    ).unsqueeze(0)                                                 # (1, 256, roi_size, roi_size)

    # --- crop binary mask at 1024 scale → resize to roi_size ---
    mask_crop = crop_to_roi(
        mask_t[0], bbox, roi_size, mode="nearest",
    ).unsqueeze(0)                                                 # (1, 1, roi_size, roi_size)

    # --- SDF on the cropped mask (higher effective resolution than full-grid SDF) ---
    sdf_np = sdf_channel_from_mask(mask_crop[0, 0].cpu().numpy()) # (1, roi_size, roi_size)
    sdf    = torch.from_numpy(sdf_np).unsqueeze(0).to(device)     # (1, 1, roi_size, roi_size)

    x    = torch.cat([emb_crop, mask_crop, sdf], dim=1)           # (1, 258, roi_size, roi_size)
    pred = model(x)                                                # (1, 1,   roi_size, roi_size)

    # --- paste prediction crop back into 1024×1024 canvas ---
    canvas = paste_into_canvas(pred[0], bbox, OUTPUT_SIZE)         # (1, 1024, 1024)
    return (canvas[0].cpu().numpy() > 0.5).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split",          default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "cnnROI")
    parser.add_argument("--output-dir",     type=Path,
                        default=PROJECT_ROOT / "data" / "roi-predictions" / "binary_masks")
    parser.add_argument("--roi-size",       type=int, default=ROI_SIZE,
                        help="Must match the value used during training.")
    parser.add_argument("--roi-pad",        type=int, default=ROI_PAD,
                        help="Must match the value used during training.")
    parser.add_argument(
        "--gating-csv-dir", type=Path, default=None,
        help=(
            "Directory containing gated_{split}_records.csv. Defaults to "
            "src/gating_mechanism/ (the canonical 5402-threshold area routing). "
            "Must match the gating CSV used for the checkpoints in "
            "--checkpoint-dir (i.e. the same routing ablation arm)."
        ),
    )
    parser.add_argument("--limit",          type=int, default=None,
                        help="Process at most N images (smoke test).")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  roi_size={args.roi_size}  roi_pad={args.roi_pad}")

    experts: dict[str, CNNExpert] = {}
    for expert_id in ("expert_small", "expert_large"):
        ckpt_path = args.checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run train_roi.py first."
            )
        experts[expert_id] = load_expert(ckpt_path, device)
        print(f"Loaded {expert_id} from {ckpt_path}")

    gating_csv_dir = args.gating_csv_dir or GATING_DIR
    csv_path = gating_csv_dir / f"gated_{args.split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Gated CSV not found: {csv_path}")
    print(f"gating csv dir: {gating_csv_dir}")

    df = pd.read_csv(csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_names = df["sample_name"].unique()
    if args.limit is not None:
        sample_names = sample_names[: args.limit]

    print(f"Running inference on {len(sample_names)} images …")

    for sample_name in tqdm(sample_names, desc="Inference"):
        rows = (
            df[df["sample_name"] == sample_name]
            .sort_values("medsam_instance_id")
            .reset_index(drop=True)
        )

        embedding_path = resolve_existing_path(rows.iloc[0]["embedding_path"])
        embedding = (
            torch.from_numpy(np.load(embedding_path))
            .float().to(device)
        )                                                          # (256, 64, 64)

        masks_path = resolve_existing_path(rows.iloc[0]["binary_masks_path"])
        all_masks  = load_prediction_masks(masks_path)             # (N, 1024, 1024)

        refined: list[np.ndarray] = []
        for _, row in rows.iterrows():
            instance_idx = int(row["medsam_instance_id"]) - 1
            binary_mask  = all_masks[instance_idx]                 # (1024, 1024)
            model        = experts[row["expert"]]

            refined_mask = refine_fragment(
                embedding, binary_mask, model, device,
                roi_size=args.roi_size, roi_pad=args.roi_pad,
            )
            refined.append(refined_mask)

        out_masks = np.stack(refined)                              # (N, 1024, 1024)
        out_path  = args.output_dir / f"{sample_name}.npz"
        np.savez_compressed(out_path, masks=out_masks)

    print(f"\nDone. Refined masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
