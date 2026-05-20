#!/usr/bin/env python3
"""Run FlowSDF MoE expert inference on gated PENGWIN fragments.

For each image in a gated split CSV:
    1. Load the shared MedSAM embedding and all MedSAM binary masks once.
    2. For each routed fragment, build the same conditioning tensor used during
       FlowSDF MoE training: resized MedSAM embedding + resized coarse mask.
    3. Select expert_small or expert_large from the gated CSV.
    4. Sample a refined SDF with the trained FlowSDF expert via Euler ODE steps.
    5. Threshold the SDF to a binary mask and upsample back to 1024x1024.
    6. Save stacked masks as .npz with key "masks", matching MedSAM conventions.

Usage:
    python infer_flowsdf_moe.py --split val --limit 10
    python infer_flowsdf_moe.py --split test --checkpoint-dir ../../checkpoints/flowsdf
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # MoE-ShapeRefine-MedicalSeg/src
REPO_ROOT = PROJECT_ROOT.parent
GATING_DIR = PROJECT_ROOT / "gating_mechanism"

for _p in (str(PROJECT_ROOT), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader_utils import load_prediction_masks, resolve_existing_path  # noqa: E402
from models import unet_segdiff  # noqa: E402


OUTPUT_SIZE = 1024


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def build_model(img_cond_channels: int, device: torch.device) -> torch.nn.Module:
    """Build the same FlowSDF UNet used by train_flowsdf_moe.py."""
    model = unet_segdiff.UNetModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=12,
        img_cond_channels=img_cond_channels,
    ).to(device)
    return model


def load_checkpoint(path: Path, device: torch.device, state_key: str) -> tuple[dict, dict]:
    """Load checkpoint and return (checkpoint, state_dict)."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if state_key == "auto":
        for key in ("ema_state", "model_state", "state_dict"):
            if key in ckpt:
                return ckpt, ckpt[key]
        raise KeyError(f"{path} has none of ema_state, model_state, or state_dict")

    if state_key not in ckpt:
        raise KeyError(f"{path} does not contain requested state key {state_key!r}")
    return ckpt, ckpt[state_key]


def load_expert(
    checkpoint_path: Path,
    device: torch.device,
    state_key: str,
    fallback_img_cond_channels: int,
) -> torch.nn.Module:
    ckpt, state_dict = load_checkpoint(checkpoint_path, device, state_key)
    img_cond_channels = int(ckpt.get("img_cond_channels", fallback_img_cond_channels))

    model = build_model(img_cond_channels=img_cond_channels, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"Loaded {checkpoint_path.name}: "
        f"state_key={state_key}, img_cond_channels={img_cond_channels}, "
        f"params={count_parameters(model)}"
    )
    return model


def prepare_conditioning(
    embedding: torch.Tensor,
    binary_mask: np.ndarray,
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return FlowSDF conditioning tensor, shape (1, 257, img_size, img_size)."""
    embedding = embedding.to(device=device, dtype=torch.float32)
    if embedding.ndim != 3:
        raise ValueError(f"expected embedding shape (256, H, W), got {tuple(embedding.shape)}")

    embedding_resized = F.interpolate(
        embedding.unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    mask = torch.from_numpy(binary_mask.astype(np.float32)).to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask, size=(img_size, img_size), mode="nearest")
    mask_resized = (mask_resized > 0.5).float()

    return torch.cat([embedding_resized, mask_resized], dim=1)


@torch.no_grad()
def sample_sdf(
    model: torch.nn.Module,
    img_cond: torch.Tensor,
    sigma_min: float,
    ode_steps: int,
    n_eval: int,
) -> torch.Tensor:
    """Sample and optionally average one or more FlowSDF trajectories."""
    samples = []
    device = img_cond.device
    _, _, height, width = img_cond.shape

    for _ in range(n_eval):
        m0 = torch.randn((1, 1, height, width), device=device)

        def func_conditional(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            t_curr = torch.ones(m.shape[0], device=m.device) * t
            m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
            return model(m_ipt, t_curr.reshape(m.shape[0], -1), img_cond=img_cond)

        times = torch.linspace(0, 1, ode_steps, device=device)
        m = m0
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            m = m + dt * func_conditional(times[i], m)
        samples.append(m)

    if len(samples) == 1:
        return samples[0]
    return torch.stack(samples, dim=0).mean(dim=0)


@torch.no_grad()
def refine_fragment(
    embedding: torch.Tensor,
    binary_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
    sigma_min: float,
    ode_steps: int,
    n_eval: int,
    sdf_binary_threshold: float,
) -> np.ndarray:
    """Refine one fragment and return a 1024x1024 uint8 binary mask."""
    img_cond = prepare_conditioning(embedding, binary_mask, img_size, device)
    expected_channels = model.rrdb.conv_first.in_channels
    if img_cond.shape[1] != expected_channels:
        raise ValueError(
            f"conditioning has {img_cond.shape[1]} channels, but model expects {expected_channels}"
        )

    sdf = sample_sdf(
        model=model,
        img_cond=img_cond,
        sigma_min=sigma_min,
        ode_steps=ode_steps,
        n_eval=n_eval,
    )
    mask_small = (sdf <= sdf_binary_threshold).float()

    mask_up = F.interpolate(
        mask_small,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="nearest",
    )
    return mask_up[0, 0].cpu().numpy().astype(np.uint8)


def split_output_dir(output_root: Path, split: str) -> Path:
    return output_root / split / "binary_masks"


def read_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_sample_names(records: list[dict[str, str]]) -> list[str]:
    seen = set()
    names = []
    for record in records:
        sample_name = record["sample_name"]
        if sample_name not in seen:
            seen.add(sample_name)
            names.append(sample_name)
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "flowsdf",
        help="Directory containing expert_small_best.pth and expert_large_best.pth.",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default="best",
        choices=["best", "final"],
        help="Which expert checkpoint suffix to load.",
    )
    parser.add_argument(
        "--state-key",
        default="ema_state",
        choices=["auto", "ema_state", "model_state", "state_dict"],
        help="Checkpoint state dict to load. Training saves ema_state and model_state.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "flowsdf-moe-predictions",
        help="Root where <split>/binary_masks/*.npz and metadata are saved.",
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--img-cond-channels", type=int, default=257)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument(
        "--n-eval",
        type=int,
        default=1,
        help="Number of stochastic samples to average per fragment.",
    )
    parser.add_argument(
        "--sdf-binary-threshold",
        type=float,
        default=0.03,
        help="Foreground threshold for sampled SDF. Original sampler uses <= 3 * 1e-2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images for a smoke test.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output .npz already exists.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    experts: dict[str, torch.nn.Module] = {}
    for expert_id in ("expert_small", "expert_large"):
        ckpt_path = args.checkpoint_dir / f"{expert_id}_{args.checkpoint_suffix}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        experts[expert_id] = load_expert(
            checkpoint_path=ckpt_path,
            device=device,
            state_key=args.state_key,
            fallback_img_cond_channels=args.img_cond_channels,
        )

    csv_path = GATING_DIR / f"gated_{args.split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing gated CSV: {csv_path}")

    records = read_records(csv_path)
    sample_names = unique_sample_names(records)
    if args.limit is not None:
        sample_names = sample_names[: args.limit]

    output_dir = split_output_dir(args.output_root, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    print(f"Running FlowSDF MoE inference on {len(sample_names)} {args.split} images")

    for sample_name in tqdm(sample_names, desc="FlowSDF MoE inference"):
        out_path = output_dir / f"{sample_name}.npz"
        if args.skip_existing and out_path.exists():
            continue

        rows = [record for record in records if record["sample_name"] == sample_name]
        rows = sorted(rows, key=lambda record: int(record["medsam_instance_id"]))

        embedding_path = resolve_existing_path(rows[0]["embedding_path"])
        masks_path = resolve_existing_path(rows[0]["binary_masks_path"])

        embedding = torch.from_numpy(np.load(embedding_path)).float()
        medsam_masks = load_prediction_masks(masks_path)
        refined_masks = np.zeros_like(medsam_masks, dtype=np.uint8)

        for row in rows:
            instance_idx = int(row["medsam_instance_id"]) - 1
            if instance_idx < 0 or instance_idx >= len(medsam_masks):
                raise IndexError(
                    f"{sample_name} medsam_instance_id={row['medsam_instance_id']} "
                    f"is outside loaded masks shape {medsam_masks.shape}"
                )

            expert_id = str(row["expert"])
            if expert_id not in experts:
                raise KeyError(f"unknown expert {expert_id!r} in {csv_path}")

            refined_masks[instance_idx] = refine_fragment(
                embedding=embedding,
                binary_mask=medsam_masks[instance_idx],
                model=experts[expert_id],
                device=device,
                img_size=args.img_size,
                sigma_min=args.sigma_min,
                ode_steps=args.ode_steps,
                n_eval=args.n_eval,
                sdf_binary_threshold=args.sdf_binary_threshold,
            )

        np.savez_compressed(out_path, masks=refined_masks)
        metadata_rows.append(
            {
                "sample_name": sample_name,
                "binary_masks_path": str(out_path.resolve()),
                "source_binary_masks_path": str(masks_path.resolve()),
                "embedding_path": str(embedding_path.resolve()),
                "num_masks": int(refined_masks.shape[0]),
            }
        )

    metadata_path = args.output_root / args.split / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Done. Masks: {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
