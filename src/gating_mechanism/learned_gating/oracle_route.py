#!/usr/bin/env python3
"""Oracle expert-routing labels for the learned-gating classifier.

For every fragment in gated_{split}_records.csv, runs BOTH experts of the
selected backend (expert_small, expert_large) -- regardless of which one it
was actually routed to -- scores each against GT with Dice, and records
which one is actually better for that fragment. Reuses already-trained
checkpoints; does not retrain anything.

Three backends are supported, matching the expert families already in this
repo's checkpoints/ (each with its own architecture, conditioning tensor,
and refinement procedure -- see cnnROI/infer_roi.py, cnnNoROI/infer_moe.py,
and FlowSDF/infer_flowsdf_moe.py). checkpoints/cnnNoROI-sam is a separate
SAM-baseline pipeline (not MedSAM-based) and is out of scope here.
  --backend cnnroi    (default) checkpoints/cnnROI    -- CNN + RoI crop, single forward pass
  --backend cnnnoroi                checkpoints/cnnNoROI -- CNN, no RoI crop, single forward pass
  --backend flowsdf                 checkpoints/flowsdf  -- FlowSDF, Euler-ODE SDF sampling

This is the ground-truth "correct expert" label used to:
  - train the learned-gating classifier (shape descriptors -> oracle_expert)
  - score any routing rule's accuracy against (fixed-threshold, random, learned)

Output: oracle_labels_{backend}_{split}.csv, one row per fragment, with
dice_expert_small, dice_expert_large, oracle_expert (argmax), dice_margin,
and fixed_expert (whatever the input gating CSV's "expert" column already
says, for a free routing-accuracy comparison).

Usage:
    python oracle_route.py --backend cnnroi  --split all
    python oracle_route.py --backend flowsdf --split all
    python oracle_route.py --split test --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader_utils import (                                               # noqa: E402
    load_prediction_masks, resolve_existing_path, load_box_metadata_map,
    get_label_path, load_pengwin_label, decode_pengwin_fragment_from_array,
    resize_binary_nearest,
)
# --------------------------------------------------------------------------

DEFAULT_BOX_METADATA = PROJECT_ROOT / "data" / "bounding-boxes-xrays" / "metadata.jsonl"
EXPERTS = ("expert_small", "expert_large")

RefineFn = Callable[[torch.Tensor, np.ndarray, torch.nn.Module, torch.device], np.ndarray]


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool, gt_bool = pred.astype(bool), gt.astype(bool)
    denom = int(pred_bool.sum() + gt_bool.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    return 2.0 * intersection / denom


# --- backend dispatch -------------------------------------------------------
# Each backend has its own architecture, checkpoint layout, and refine_fragment
# signature (see cnnROI/infer_roi.py vs FlowSDF/infer_flowsdf_moe.py). Only the
# selected backend's inference module is imported, and only once its sys.path
# prerequisite (if any) is in place, to avoid the two families' internal
# generic module names (e.g. FlowSDF's "models" package) colliding.

def load_experts_cnnroi(args: argparse.Namespace, device: torch.device) -> dict:
    from cnnROI.infer_roi import load_expert
    experts = {}
    for expert_id in EXPERTS:
        ckpt_path = args.checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train_roi.py first.")
        experts[expert_id] = load_expert(ckpt_path, device)
        print(f"Loaded {expert_id} from {ckpt_path}")
    return experts


def make_refine_fn_cnnroi(args: argparse.Namespace) -> RefineFn:
    from cnnROI.infer_roi import refine_fragment

    def refine(embedding, binary_mask, model, device):
        return refine_fragment(
            embedding, binary_mask, model, device,
            roi_size=args.roi_size, roi_pad=args.roi_pad,
        )

    return refine


def load_experts_cnnnoroi(args: argparse.Namespace, device: torch.device) -> dict:
    from cnnNoROI.infer_moe import load_expert
    experts = {}
    for expert_id in EXPERTS:
        ckpt_path = args.checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train_moe.py first.")
        experts[expert_id] = load_expert(ckpt_path, device)
        print(f"Loaded {expert_id} from {ckpt_path}")
    return experts


def make_refine_fn_cnnnoroi(args: argparse.Namespace) -> RefineFn:
    from cnnNoROI.infer_moe import refine_fragment

    def refine(embedding, binary_mask, model, device):
        return refine_fragment(embedding, binary_mask, model, device)

    return refine


def load_experts_flowsdf(args: argparse.Namespace, device: torch.device) -> dict:
    flowsdf_dir = SRC_DIR / "FlowSDF"
    if str(flowsdf_dir) not in sys.path:
        # infer_flowsdf_moe.py does `from models import unet_segdiff`, which
        # only resolves if FlowSDF/ itself (not just src/) is on sys.path --
        # normally true only when it's run standalone as __main__.
        sys.path.insert(0, str(flowsdf_dir))
    from FlowSDF.infer_flowsdf_moe import load_expert

    experts = {}
    for expert_id in EXPERTS:
        ckpt_path = args.checkpoint_dir / f"{expert_id}_{args.checkpoint_suffix}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train_flowsdf_moe.py first.")
        experts[expert_id] = load_expert(
            checkpoint_path=ckpt_path, device=device,
            state_key=args.state_key,
            fallback_img_cond_channels=args.img_cond_channels,
        )
    return experts


def make_refine_fn_flowsdf(args: argparse.Namespace) -> RefineFn:
    from FlowSDF.infer_flowsdf_moe import refine_fragment

    def refine(embedding, binary_mask, model, device):
        return refine_fragment(
            embedding=embedding, binary_mask=binary_mask, model=model, device=device,
            img_size=args.img_size, sigma_min=args.sigma_min, ode_steps=args.ode_steps,
            n_eval=args.n_eval, sdf_binary_threshold=args.sdf_binary_threshold,
        )

    return refine


BACKENDS = {
    "cnnroi": {
        "default_checkpoint_dir": PROJECT_ROOT / "checkpoints" / "cnnROI",
        "load_experts": load_experts_cnnroi,
        "make_refine_fn": make_refine_fn_cnnroi,
    },
    "cnnnoroi": {
        "default_checkpoint_dir": PROJECT_ROOT / "checkpoints" / "cnnNoROI",
        "load_experts": load_experts_cnnnoroi,
        "make_refine_fn": make_refine_fn_cnnnoroi,
    },
    "flowsdf": {
        "default_checkpoint_dir": PROJECT_ROOT / "checkpoints" / "flowsdf",
        "load_experts": load_experts_flowsdf,
        "make_refine_fn": make_refine_fn_flowsdf,
    },
}
# --------------------------------------------------------------------------


def route_one_split(
    split: str,
    experts: dict,
    refine_fn: RefineFn,
    device: torch.device,
    gating_csv_dir: Path,
    box_metadata: dict,
    limit: int | None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    csv_path = gating_csv_dir / f"gated_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Gated CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sample_names = df["sample_name"].unique()
    if limit is not None:
        sample_names = sample_names[:limit]

    rows: list[dict] = []
    done_samples: set[str] = set()

    if checkpoint_path is not None and resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                done_samples.add(record["sample_name"])
                rows.extend(record["rows"])
        print(
            f"[{split}] resuming from checkpoint: {len(done_samples)} samples already "
            f"processed, {len(rows)} rows loaded from {checkpoint_path}"
        )

    checkpoint_handle = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")

    def write_checkpoint(sample_name: str, sample_rows: list[dict]) -> None:
        if checkpoint_handle is None:
            return
        checkpoint_handle.write(json.dumps({"sample_name": sample_name, "rows": sample_rows}) + "\n")
        checkpoint_handle.flush()

    skipped_no_label = 0
    skipped_error = 0

    try:
        for sample_name in tqdm(sample_names, desc=f"[{split}] oracle routing"):
            if sample_name in done_samples:
                continue

            sample_rows = (
                df[df["sample_name"] == sample_name]
                .sort_values("medsam_instance_id")
                .reset_index(drop=True)
            )

            try:
                label_path = get_label_path({"sample_name": sample_name}, box_metadata)
                if label_path is None or not Path(label_path).exists():
                    skipped_no_label += len(sample_rows)
                    write_checkpoint(sample_name, [])
                    continue
                segmentation = load_pengwin_label(label_path)

                embedding_path = resolve_existing_path(sample_rows.iloc[0]["embedding_path"])
                embedding = torch.from_numpy(np.load(embedding_path)).float().to(device)

                masks_path = resolve_existing_path(sample_rows.iloc[0]["binary_masks_path"])
                all_masks  = load_prediction_masks(masks_path)

                sample_out_rows: list[dict] = []
                for _, row in sample_rows.iterrows():
                    instance_idx = int(row["medsam_instance_id"]) - 1
                    binary_mask  = all_masks[instance_idx]

                    gt = decode_pengwin_fragment_from_array(
                        segmentation, int(row["category_id"]), int(row["fragment_id"]),
                    )

                    dice_by_expert = {}
                    for expert_id in EXPERTS:
                        refined = refine_fn(embedding, binary_mask, experts[expert_id], device)
                        gt_matched = resize_binary_nearest(gt, refined.shape)
                        dice_by_expert[expert_id] = dice_score(refined, gt_matched)

                    oracle_expert = max(dice_by_expert, key=dice_by_expert.get)
                    sample_out_rows.append({
                        "split":              split,
                        "case_id":            row["case_id"],
                        "sample_name":        sample_name,
                        "medsam_instance_id": int(row["medsam_instance_id"]),
                        "category_id":        int(row["category_id"]),
                        "category_name":      row["category_name"],
                        "fragment_id":        int(row["fragment_id"]),
                        "dice_expert_small":  dice_by_expert["expert_small"],
                        "dice_expert_large":  dice_by_expert["expert_large"],
                        "oracle_expert":      oracle_expert,
                        "dice_margin":        abs(dice_by_expert["expert_small"] - dice_by_expert["expert_large"]),
                        "fixed_expert":       row["expert"],
                    })
            except Exception:
                skipped_error += len(sample_rows)
                print(f"[{split}] ERROR on sample {sample_name!r} -- skipping ({len(sample_rows)} fragments):")
                traceback.print_exc()
                write_checkpoint(sample_name, [])
                continue

            rows.extend(sample_out_rows)
            write_checkpoint(sample_name, sample_out_rows)
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()

    if skipped_no_label:
        print(f"[{split}] skipped {skipped_no_label} fragments (missing GT label file)")
    if skipped_error:
        print(f"[{split}] skipped {skipped_error} fragments (per-sample error, see traceback above)")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=list(BACKENDS), default="cnnroi",
                        help="Which expert family to oracle-route with.")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Defaults to checkpoints/cnnROI or checkpoints/flowsdf depending on --backend.",
    )
    parser.add_argument(
        "--gating-csv-dir", type=Path, default=None,
        help="Directory with gated_{split}_records.csv. Defaults to src/gating_mechanism/.",
    )
    parser.add_argument("--box-metadata", type=Path, default=DEFAULT_BOX_METADATA)
    parser.add_argument("--out-dir", type=Path, default=GATING_DIR / "learned_gating")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images per split (smoke test).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the per-split checkpoint file in out-dir, if present.",
    )

    # cnnroi-specific
    parser.add_argument("--roi-size", type=int, default=64,
                        help="[cnnroi] Must match the value used to train the checkpoints.")
    parser.add_argument("--roi-pad", type=int, default=64,
                        help="[cnnroi] Must match the value used to train the checkpoints.")

    # flowsdf-specific -- mirrors infer_flowsdf_moe.py's defaults exactly
    parser.add_argument("--checkpoint-suffix", default="best", choices=["best", "final"],
                        help="[flowsdf] Which expert checkpoint suffix to load.")
    parser.add_argument("--state-key", default="ema_state",
                        choices=["auto", "ema_state", "model_state", "state_dict"],
                        help="[flowsdf] Checkpoint state dict to load.")
    parser.add_argument("--img-size", type=int, default=128, help="[flowsdf]")
    parser.add_argument("--img-cond-channels", type=int, default=257, help="[flowsdf]")
    parser.add_argument("--sigma-min", type=float, default=1e-5, help="[flowsdf]")
    parser.add_argument("--ode-steps", type=int, default=4, help="[flowsdf]")
    parser.add_argument("--n-eval", type=int, default=1,
                        help="[flowsdf] Number of stochastic samples to average per fragment.")
    parser.add_argument("--sdf-binary-threshold", type=float, default=0.03, help="[flowsdf]")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = BACKENDS[args.backend]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = backend["default_checkpoint_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"backend: {args.backend}  device: {device}  checkpoint_dir: {args.checkpoint_dir}")

    experts   = backend["load_experts"](args, device)
    refine_fn = backend["make_refine_fn"](args)

    gating_csv_dir = args.gating_csv_dir or GATING_DIR
    print(f"gating csv dir: {gating_csv_dir}")

    box_metadata = load_box_metadata_map(args.box_metadata)
    print(f"box metadata: {len(box_metadata)} samples from {args.box_metadata}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        checkpoint_path = args.out_dir / f"oracle_labels_{args.backend}_{split}_checkpoint.jsonl"
        df = route_one_split(
            split, experts, refine_fn, device, gating_csv_dir, box_metadata, args.limit,
            checkpoint_path=checkpoint_path, resume=args.resume,
        )
        if df.empty:
            print(f"[{split}] no fragments processed -- skipping write")
            continue

        out_path = args.out_dir / f"oracle_labels_{args.backend}_{split}.csv"
        df.to_csv(out_path, index=False)

        n_small = int((df["oracle_expert"] == "expert_small").sum())
        n_large = int((df["oracle_expert"] == "expert_large").sum())
        fixed_matches = int((df["oracle_expert"] == df["fixed_expert"]).sum())
        print(
            f"[{split}] {len(df)} fragments -> {out_path}\n"
            f"  oracle:  expert_small={n_small} ({100*n_small/len(df):.1f}%)  "
            f"expert_large={n_large} ({100*n_large/len(df):.1f}%)\n"
            f"  input gating CSV's routing already matches oracle on "
            f"{fixed_matches}/{len(df)} ({100*fixed_matches/len(df):.1f}%)"
        )


if __name__ == "__main__":
    main()
