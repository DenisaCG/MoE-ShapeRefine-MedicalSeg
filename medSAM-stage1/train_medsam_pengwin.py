#!/usr/bin/env python3
"""Fine-tune MedSAM on PENGWIN X-ray data using the project's train/val split.

Uses gated_train_records.csv and gated_val_records.csv as split indices
(same split used by the MoE models). The gating expert column is ignored —
all training fragments are used regardless of size.

For each fragment:
  - Load raw .tif image, apply neglog normalisation, resize to 1024x1024
  - Decode GT fragment mask from bit-packed label
  - Derive bounding box from GT mask with random perturbation (bbox_shift)
  - Fine-tune MedSAM image encoder + mask decoder (prompt encoder frozen)

Checkpoint saved to --checkpoint-dir, format: {'model', 'optimizer', 'epoch'}
compatible with run_medsam_with_pengwin_boxes.py --finetuned-ckpt.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDSAM_ROOT = PROJECT_ROOT.parent / "MedSAM"
CSV_DIR = PROJECT_ROOT / "src" / "gating_mechanism"
GT_DIR = PROJECT_ROOT.parent / "data" / "pengwin" / "original" / "task2_xray" / "train" / "output" / "images" / "x-ray"
IMAGE_DIR = PROJECT_ROOT.parent / "data" / "pengwin" / "original" / "task2_xray" / "train" / "input" / "images" / "x-ray"
DEFAULT_CHECKPOINT = MEDSAM_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from paths import SCRATCH_CHECKPOINTS

DEFAULT_CKPT_DIR = SCRATCH_CHECKPOINTS / "medsam-finetuned"

if str(MEDSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything import sam_model_registry  # noqa: E402
from src.dataloader_utils import (  # noqa: E402
    decode_pengwin_fragment_from_record,
    load_pengwin_label,
    resolve_existing_path,
)


# ---------------------------------------------------------------------------
# Image preprocessing  (matches run_medsam_with_pengwin_boxes.py)
# ---------------------------------------------------------------------------

def load_and_preprocess_image(path: Path, image_size: int = 1024) -> np.ndarray:
    """Load a raw DRR TIF → neglog → quantile window → uint8 → resize → float32 [0,1]."""
    image = np.array(Image.open(resolve_existing_path(path))).astype(np.float32)
    image += image.min() + 1e-3
    image = -np.log(image)
    q_lo = np.quantile(image, 0.01)
    q_hi = np.quantile(image, 0.95)
    if q_lo == q_hi:
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo + 1e-7)
    else:
        image = (image - q_hi) / (q_lo - q_hi + 1e-7)
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((image_size, image_size), Image.BILINEAR))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image.astype(np.float32) / 255.0  # (1024, 1024, 3) in [0, 1]


def bbox_from_mask(mask: np.ndarray, bbox_shift: int = 20) -> np.ndarray:
    """GT mask → perturbed xyxy bounding box on the 448-pixel label grid."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, mask.shape[1], mask.shape[0]], dtype=np.float32)
    H, W = mask.shape
    x_min = max(0, xs.min() - random.randint(0, bbox_shift))
    x_max = min(W, xs.max() + random.randint(0, bbox_shift))
    y_min = max(0, ys.min() - random.randint(0, bbox_shift))
    y_max = min(H, ys.max() + random.randint(0, bbox_shift))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def scale_bbox_to_1024(bbox: np.ndarray, orig_h: int, orig_w: int, target: int = 1024) -> np.ndarray:
    """Scale a bbox from original label resolution to target (1024) grid."""
    scale_x = target / orig_w
    scale_y = target / orig_h
    return np.array([
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PengwinFragmentDataset(Dataset):
    """One row per fragment from a gated_*_records.csv.

    Returns (image_tensor, gt_mask_tensor, bbox_tensor) where:
      image_tensor : float32 (3, 1024, 1024) in [0, 1]
      gt_mask_tensor: long   (1, 1024, 1024) binary
      bbox_tensor  : float32 (4,) xyxy on 1024 grid
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        label_dir: Path,
        bbox_shift: int = 20,
        image_size: int = 1024,
    ):
        self.records = records
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.bbox_shift = bbox_shift
        self.image_size = image_size


    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int):
        rec = self.records[idx]
        sample_name = rec["sample_name"]
        case_id = sample_name.replace("XRAY_PENGWIN_", "")  # e.g. 001_0000

        # --- image ---
        img_path = self.image_dir / f"{case_id}.tif"
        image = load_and_preprocess_image(img_path, image_size=self.image_size)

        # --- GT label ---
        lbl_path = self.label_dir / f"{case_id}.tif"
        label = load_pengwin_label(lbl_path)  # (H, W) uint32

        gt_mask = decode_pengwin_fragment_from_record(label, rec)  # (H, W) uint8

        # --- bbox (perturbed, on label grid, then scaled to 1024) ---
        orig_h, orig_w = gt_mask.shape
        bbox_orig = bbox_from_mask(gt_mask, bbox_shift=self.bbox_shift)
        bbox_1024 = scale_bbox_to_1024(bbox_orig, orig_h, orig_w, target=self.image_size)

        # resize GT mask to 1024x1024
        gt_1024 = np.array(
            Image.fromarray(gt_mask).resize((self.image_size, self.image_size), Image.NEAREST)
        ).astype(np.uint8)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()     # (3, 1024, 1024)
        gt_t = torch.from_numpy(gt_1024).unsqueeze(0).long()            # (1, 1024, 1024)
        bbox_t = torch.from_numpy(bbox_1024).float()                    # (4,)

        return image_t, gt_t, bbox_t

    def __getitem__(self, idx: int):
        # Corpus has known-corrupt raw .tif files (see 042_0211.tif, 082_0302.tif
        # incidents) — a multi-day training run must not die on one bad sample.
        for attempt in range(len(self.records)):
            try:
                return self._load((idx + attempt) % len(self.records))
            except Exception as e:
                bad_rec = self.records[(idx + attempt) % len(self.records)]
                print(f"  [skip] corrupt sample {bad_rec.get('sample_name')}: {e}")
        raise RuntimeError("No loadable sample found in dataset.")


# ---------------------------------------------------------------------------
# Loss  (replaces monai.losses.DiceLoss to avoid extra dependency)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Sigmoid Dice loss matching monai DiceLoss(sigmoid=True, squared_pred=True)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        inter = (probs * targets).sum(dim=dims)
        denom = (probs ** 2 + targets ** 2).sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
        return dice.mean()


# ---------------------------------------------------------------------------
# Model  (identical to train_one_gpu.py)
# ---------------------------------------------------------------------------

class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, boxes: np.ndarray) -> torch.Tensor:
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=image.device)
            if box_torch.ndim == 2:
                box_torch = box_torch[:, None, :]
            sparse_emb, dense_emb = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        return F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Base MedSAM SAM vit_b checkpoint.")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR,
                   help="Directory to save checkpoints.")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping: stop if val loss does not improve for this many epochs.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--bbox-shift", type=int, default=20)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a previous checkpoint.")
    p.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    p.add_argument("--label-dir", type=Path, default=GT_DIR)
    p.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    p.add_argument("--no-amp", action="store_true", default=False,
                   help="Disable automatic mixed precision (AMP).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    torch.manual_seed(2023)
    random.seed(2023)
    np.random.seed(2023)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_df = pd.read_csv(args.csv_dir / "gated_train_records.csv")
    val_df   = pd.read_csv(args.csv_dir / "gated_val_records.csv")
    train_records = train_df.to_dict("records")
    val_records   = val_df.to_dict("records")
    print(f"Train fragments: {len(train_records)}  |  Val fragments: {len(val_records)}")

    train_loader = DataLoader(
        PengwinFragmentDataset(train_records, args.image_dir, args.label_dir,
                               bbox_shift=args.bbox_shift, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PengwinFragmentDataset(val_records, args.image_dir, args.label_dir,
                               bbox_shift=0,  # no perturbation at val time
                               image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- model ---
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    model = MedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder).to(device)

    params = list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = DiceLoss()
    ce_loss  = nn.BCEWithLogitsLoss(reduction="mean")
    use_amp  = not args.no_amp
    scaler   = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")
        print(f"  best_val_loss={best_val_loss:.4f}  epochs_without_improvement={epochs_without_improvement}")

    print(f"Trainable parameters: {sum(p.numel() for p in params if p.requires_grad):,}")

    for epoch in range(start_epoch, args.epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for image, gt, bbox in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            image, gt = image.to(device), gt.to(device)
            boxes_np = bbox.numpy()
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(image, boxes_np)
                loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, gt, bbox in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                image, gt = image.to(device), gt.to(device)
                boxes_np = bbox.numpy()
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(image, boxes_np)
                    loss = seg_loss(pred, gt.float()) + ce_loss(pred, gt.float())
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val_loss": best_val_loss,
                 "epochs_without_improvement": epochs_without_improvement},
                args.checkpoint_dir / "medsam_pengwin_best.pth",
            )
            print(f"  → new best val loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  → no improvement ({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "best_val_loss": best_val_loss,
             "epochs_without_improvement": epochs_without_improvement},
            args.checkpoint_dir / "medsam_pengwin_latest.pth",
        )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
