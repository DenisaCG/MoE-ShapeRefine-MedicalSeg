#!/usr/bin/env python3
"""Train MoE CNN experts with RoI crops (cnnROI).

Pipeline per batch:
    1. For each sample: extract bbox from binary_mask at 1024×1024 scale.
    2. Crop embedding to bbox (projected to 64×64) → resize to ROI_SIZE.
    3. Crop binary_mask to bbox → resize to ROI_SIZE → compute SDF on crop.
    4. Concatenate: embedding_crop (256) + mask_crop (1) + SDF (1) → (258, ROI_SIZE, ROI_SIZE).
    5. Forward through CNNExpert → (1, ROI_SIZE, ROI_SIZE).
    6. Crop GT mask to same bbox (projected to 448×448) → resize to ROI_SIZE.
    7. Boundary-Dice-BCE loss at ROI_SIZE — no upsample step.

Fallback for empty MedSAM masks: global resize to ROI_SIZE (identical to cnnNoROI).

Model, loss, and dataloader are shared with cnnNoROI — only the per-batch
preprocessing differs.

Usage:
    python train_roi.py [--expert expert_small|expert_large|both] [OPTIONS]

    python train_roi.py --expert both --epochs 40 --patience 10 --batch-size 64 --large-subsample 17000
    python train_roi.py --no-wandb --expert expert_small   # disable W&B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdf_utils import sdf_channel_from_mask           # noqa: E402
from cnnNoROI.cnnMoE import CNNExpert                 # noqa: E402
from cnnNoROI.losses import boundary_dice_bce_loss    # noqa: E402
from cnnROI.roi_utils import crop_to_roi, extract_bbox, scale_bbox  # noqa: E402
from dataset import build_expert_dataloaders           # noqa: E402
# --------------------------------------------------------------------------

FEAT_SIZE = 64    # MedSAM embedding spatial resolution (fixed)
GT_SIZE   = 448   # GT mask spatial resolution from dataloader (fixed)
ROI_SIZE  = 64    # default RoI crop output size — same as cnnNoROI internal grid
ROI_PAD   = 64    # default padding around fragment bbox at 1024 scale


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def wandb_init(args: argparse.Namespace, expert_id: str) -> bool:
    """Initialise a W&B run. Returns True if W&B is active."""
    if args.no_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{expert_id}__roi__{args.wandb_project}",
            config={
                "expert":          expert_id,
                "epochs":          args.epochs,
                "patience":        args.patience,
                "lr":              args.lr,
                "batch_size":      args.batch_size,
                "w_boundary":      args.w_boundary,
                "boundary_radius": args.boundary_radius,
                "dice_weight":     args.dice_weight,
                "large_subsample": args.large_subsample,
                "roi_size":        args.roi_size,
                "roi_pad":         args.roi_pad,
                "feat_size":       FEAT_SIZE,
            },
        )
        return True
    except Exception as exc:
        print(f"WARNING: W&B init failed ({exc}). Continuing without W&B.")
        return False


def wandb_log(metrics: dict, step: int, active: bool) -> None:
    if not active:
        return
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


def wandb_finish(active: bool) -> None:
    if not active:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    expert_id: str,
    checkpoint_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, label="train loss")
        ax.plot(epochs, val_losses,   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{expert_id} (RoI) — training curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = checkpoint_dir / f"{expert_id}_loss_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss curve saved: {out}")
    except Exception as exc:
        print(f"  WARNING: could not save loss curve ({exc})")


# ---------------------------------------------------------------------------
# RoI batch preparation
# ---------------------------------------------------------------------------

def prepare_roi_batch(
    embedding:   torch.Tensor,   # (B, 256, 64, 64)
    binary_mask: torch.Tensor,   # (B, 1, 1024, 1024)
    gt_mask:     torch.Tensor,   # (B, 1, 448, 448)
    roi_size:    int,
    roi_pad:     int,
    device:      torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RoI-cropped batch tensors.

    For each sample, extracts a bbox from the MedSAM binary mask, crops
    the embedding and GT to that bbox, and resizes all crops to roi_size.
    Falls back to global resize for empty masks.

    Returns:
        emb_b:  (B, 256, roi_size, roi_size)
        mask_b: (B,   1, roi_size, roi_size)
        sdf_b:  (B,   1, roi_size, roi_size)
        gt_b:   (B,   1, roi_size, roi_size)
    """
    B = embedding.size(0)
    emb_crops: list[torch.Tensor]  = []
    mask_crops: list[torch.Tensor] = []
    sdf_crops: list[torch.Tensor]  = []
    gt_crops: list[torch.Tensor]   = []

    for i in range(B):
        mask_np = binary_mask[i, 0].cpu().numpy()       # (1024, 1024)
        bbox    = extract_bbox(mask_np, pad=roi_pad)

        if bbox is None:
            # Empty MedSAM mask — fall back to global resize (same as cnnNoROI)
            emb_c  = F.interpolate(
                embedding[i:i+1], size=(roi_size, roi_size),
                mode="bilinear", align_corners=False,
            )[0]                                         # (256, roi_size, roi_size)
            mask_c = F.interpolate(
                binary_mask[i:i+1], size=(roi_size, roi_size), mode="nearest",
            )[0]                                         # (1, roi_size, roi_size)
            gt_c   = F.interpolate(
                gt_mask[i:i+1], size=(roi_size, roi_size),
                mode="bilinear", align_corners=False,
            )[0]                                         # (1, roi_size, roi_size)
        else:
            emb_c  = crop_to_roi(
                embedding[i],
                scale_bbox(bbox, 1024, FEAT_SIZE),
                roi_size, mode="bilinear",
            )                                            # (256, roi_size, roi_size)
            mask_c = crop_to_roi(
                binary_mask[i], bbox, roi_size, mode="nearest",
            )                                            # (1, roi_size, roi_size)
            gt_c   = crop_to_roi(
                gt_mask[i],
                scale_bbox(bbox, 1024, GT_SIZE),
                roi_size, mode="bilinear",
            )                                            # (1, roi_size, roi_size)

        sdf_np = sdf_channel_from_mask(mask_c[0].cpu().numpy())  # (1, roi_size, roi_size)
        sdf_c  = torch.from_numpy(sdf_np).to(device)

        emb_crops.append(emb_c)
        mask_crops.append(mask_c)
        sdf_crops.append(sdf_c)
        gt_crops.append(gt_c)

    return (
        torch.stack(emb_crops),    # (B, 256, roi_size, roi_size)
        torch.stack(mask_crops),   # (B,   1, roi_size, roi_size)
        torch.stack(sdf_crops),    # (B,   1, roi_size, roi_size)
        torch.stack(gt_crops),     # (B,   1, roi_size, roi_size)
    )


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def run_one_epoch(
    model:           CNNExpert,
    loader,
    device:          torch.device,
    optimizer:       AdamW | None,
    roi_size:        int,
    roi_pad:         int,
    w_boundary:      float,
    boundary_radius: int,
    dice_weight:     float,
    desc:            str,
) -> float:
    """One training or validation epoch. Returns mean loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            embedding   = batch["embedding"].to(device)      # (B, 256, 64, 64)
            binary_mask = batch["binary_mask"].to(device)    # (B, 1, 1024, 1024)
            gt_mask     = batch["gt_mask"].to(device)        # (B, 1, 448, 448)

            emb_b, mask_b, sdf_b, gt_b = prepare_roi_batch(
                embedding, binary_mask, gt_mask, roi_size, roi_pad, device,
            )

            x    = torch.cat([emb_b, mask_b, sdf_b], dim=1)  # (B, 258, roi_size, roi_size)
            pred = model(x)                                    # (B, 1,   roi_size, roi_size)

            loss = boundary_dice_bce_loss(
                pred, gt_b, w_boundary, boundary_radius, dice_weight,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_expert(
    expert_id:    str,
    train_loader,
    val_loader,
    device:       torch.device,
    args:         argparse.Namespace,
) -> None:
    model     = CNNExpert(c_in=258).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wb_active      = wandb_init(args, expert_id)
    best_val_loss  = float("inf")
    epochs_no_improve = 0
    last_epoch     = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        tag = f"[{expert_id}] {epoch}/{args.epochs}"

        train_loss = run_one_epoch(
            model, train_loader, device, optimizer,
            args.roi_size, args.roi_pad,
            args.w_boundary, args.boundary_radius, args.dice_weight,
            desc=f"{tag} train",
        )
        val_loss = run_one_epoch(
            model, val_loader, device, None,
            args.roi_size, args.roi_pad,
            args.w_boundary, args.boundary_radius, args.dice_weight,
            desc=f"{tag} val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"{tag}  train={train_loss:.4f}  val={val_loss:.4f}")

        wandb_log(
            {f"{expert_id}/train_loss": train_loss, f"{expert_id}/val_loss": val_loss},
            step=epoch,
            active=wb_active,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = args.checkpoint_dir / f"{expert_id}_best.pth"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
                ckpt,
            )
            print(f"  -> best checkpoint: {ckpt}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"[{expert_id}] early stopping at epoch {epoch} "
                    f"(no val improvement for {args.patience} epochs, "
                    f"best_val_loss={best_val_loss:.4f})"
                )
                break

    final = args.checkpoint_dir / f"{expert_id}_final.pth"
    torch.save({"epoch": last_epoch, "model_state": model.state_dict()}, final)
    print(f"[{expert_id}] done. Final: {final}")

    save_loss_curve(train_losses, val_losses, expert_id, args.checkpoint_dir)
    wandb_finish(wb_active)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--expert", choices=["expert_small", "expert_large", "both"], default="both")
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--patience",        type=int,   default=10,
                        help="Early-stopping patience: stop after this many epochs "
                             "without val-loss improvement (tracked independently per expert).")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--num-workers",     type=int,   default=4)
    parser.add_argument("--roi-size",        type=int,   default=ROI_SIZE,
                        help="Spatial size of the RoI crop fed to the CNN.")
    parser.add_argument("--roi-pad",         type=int,   default=ROI_PAD,
                        help="Padding around fragment bbox at 1024 scale.")
    parser.add_argument("--w-boundary",      type=float, default=3.0)
    parser.add_argument("--boundary-radius", type=int,   default=3)
    parser.add_argument("--dice-weight",     type=float, default=0.5)
    parser.add_argument("--large-subsample", type=int,   default=None)
    parser.add_argument("--checkpoint-dir",  type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "cnnROI")
    parser.add_argument(
        "--gating-csv-dir", type=Path, default=None,
        help=(
            "Directory containing gated_{split}_records.csv. Defaults to "
            "src/gating_mechanism/ (the canonical 5402-threshold area routing). "
            "Point this at an alternate gating output directory (e.g. a "
            "different --threshold or --routing random run of "
            "gating_mechanism.py) to train this expert on a different "
            "routing ablation arm."
        ),
    )
    parser.add_argument("--wandb-project",   type=str,   default="moe-shaprefine")
    parser.add_argument("--no-wandb",        action="store_true",
                        help="Disable W&B logging. Loss curve PNG is always saved locally.")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  roi_size={args.roi_size}  roi_pad={args.roi_pad}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    experts = ["expert_small", "expert_large"] if args.expert == "both" else [args.expert]

    print("Building dataloaders …")
    if args.gating_csv_dir is not None:
        print(f"  gating csv dir: {args.gating_csv_dir}")
    train_loaders = build_expert_dataloaders(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        large_subsample=args.large_subsample,
        csv_dir=args.gating_csv_dir,
    )
    val_loaders = build_expert_dataloaders(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csv_dir=args.gating_csv_dir,
    )

    for expert_id in experts:
        print(f"\n{'='*60}\nTraining {expert_id} (RoI crops)\n{'='*60}")
        train_expert(
            expert_id=expert_id,
            train_loader=train_loaders[expert_id],
            val_loader=val_loaders[expert_id],
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
