#!/usr/bin/env python3
"""MoE inference script — Alt 3 (no RoI crops).

Identical to infer_moe.py, with one addition: --gating-csv-dir, so this can
be pointed at gating output for a different upstream model's predictions
(e.g. data/sam-predictions/gating/ instead of the canonical MedSAM-based
src/gating_mechanism/). infer_moe.py itself is left untouched.

For each image in the gated CSV:
    1. Load shared embedding (256, 64, 64) once.
    2. Load all binary masks (N, 1024, 1024) once.
    3. For each fragment: route to expert_small or expert_large via the CSV,
       resize mask to 64×64, compute SDF, run CNN, upsample to 1024×1024.
    4. Stack refined masks → (N, 1024, 1024) and save as .npz.

Output format matches the MedSAM .npz convention (key "masks") so the
existing evaluate_medsam_pengwin.py evaluator can read it unchanged.

Usage:
    python infer_moe_sam.py --gating-csv-dir data/sam-predictions/gating [OPTIONS]
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
# --------------------------------------------------------------------------

FEAT_SIZE   = 64     # embedding spatial resolution
OUTPUT_SIZE = 1024   # output mask resolution (matches MedSAM npz format)


def load_expert(checkpoint_path: Path, device: torch.device) -> CNNExpert:
    model = CNNExpert(c_in=258).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def refine_fragment(
    embedding: torch.Tensor,   # (256, 64, 64), on device, float32
    binary_mask: np.ndarray,   # (1024, 1024) uint8/float
    model: CNNExpert,
    device: torch.device,
) -> np.ndarray:
    """Refine one fragment mask. Returns (1024, 1024) uint8 binary mask."""
    mask_t = (
        torch.from_numpy(binary_mask.astype(np.float32))
        .unsqueeze(0).unsqueeze(0)
        .to(device)
    )                                                              # (1, 1, 1024, 1024)

    mask_small = F.interpolate(
        mask_t, size=(FEAT_SIZE, FEAT_SIZE), mode="nearest"
    )                                                              # (1, 1, 64, 64)

    sdf_np = sdf_channel_from_mask(mask_small[0, 0].cpu().numpy())  # (1, 64, 64)
    sdf    = torch.from_numpy(sdf_np).unsqueeze(0).to(device)        # (1, 1, 64, 64)

    x    = torch.cat([embedding.unsqueeze(0), mask_small, sdf], dim=1)  # (1, 258, 64, 64)
    pred = model(x)                                                       # (1, 1, 64, 64)

    pred_up = F.interpolate(
        pred, size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bilinear", align_corners=False,
    )                                                              # (1, 1, 1024, 1024)

    return (pred_up[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split",          default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "cnnNoROI-sam")
    parser.add_argument("--output-dir",     type=Path,
                        default=PROJECT_ROOT / "data" / "sam-moe-predictions" / "binary_masks")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Process at most N images (smoke test).")
    parser.add_argument(
        "--gating-csv-dir", type=Path, default=None,
        help=(
            "Directory containing gated_{split}_records.csv. Defaults to "
            "src/gating_mechanism/ (the canonical MedSAM-based routing). "
            "Point this at data/sam-predictions/gating/ to run inference on "
            "plain SAM's gated predictions instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load both expert checkpoints
    experts: dict[str, CNNExpert] = {}
    for expert_id in ("expert_small", "expert_large"):
        ckpt_path = args.checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run train_moe_sam.py first."
            )
        experts[expert_id] = load_expert(ckpt_path, device)
        print(f"Loaded {expert_id} from {ckpt_path}")

    # Load gated CSV for this split
    csv_dir  = args.gating_csv_dir or GATING_DIR
    csv_path = csv_dir / f"gated_{args.split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Gated CSV not found: {csv_path}")
    print(f"gating csv dir: {csv_dir}")

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

        # Load shared assets once per image
        embedding_path = resolve_existing_path(rows.iloc[0]["embedding_path"])
        embedding = (
            torch.from_numpy(np.load(embedding_path))
            .float()
            .to(device)
        )                                                         # (256, 64, 64)

        masks_path = resolve_existing_path(rows.iloc[0]["binary_masks_path"])
        all_masks  = load_prediction_masks(masks_path)            # (N, 1024, 1024)

        refined: list[np.ndarray] = []
        for _, row in rows.iterrows():
            instance_idx = int(row["medsam_instance_id"]) - 1    # 0-based
            binary_mask  = all_masks[instance_idx]                # (1024, 1024)
            model        = experts[row["expert"]]

            refined_mask = refine_fragment(embedding, binary_mask, model, device)
            refined.append(refined_mask)

        out_masks = np.stack(refined)                             # (N, 1024, 1024)
        out_path  = args.output_dir / f"{sample_name}.npz"
        np.savez_compressed(out_path, masks=out_masks)

    print(f"\nDone. Refined masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
