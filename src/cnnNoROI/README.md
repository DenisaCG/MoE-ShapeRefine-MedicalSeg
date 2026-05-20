# cnnNoROI — Stage 2 MoE Refinement (No RoI Crops)

This package implements **Stage 2** of the PENGWIN Task 2 segmentation pipeline:
a Mixture of Experts (MoE) CNN that takes MedSAM's coarse predictions and refines them.

The "NoROI" label distinguishes this from a potential Phase 2 approach using region-of-interest crops.
Here the whole 64×64 feature map is processed without any spatial cropping.

---

## Context: Two-Stage Pipeline

```
Raw X-ray (1024×1024)
        │
        ▼
  ┌─────────────┐
  │   MedSAM    │  Stage 1 (pre-computed, not trained here)
  │  (ViT-B)   │
  └──────┬──────┘
         │ embedding (256, 64, 64) + binary_mask (N, 1024, 1024)
         ▼
  ┌─────────────────────────────────┐
  │   Gating Mechanism              │  routes each fragment by bounding-box area
  │   (src/gating_mechanism/)       │  threshold ~5,402 px on 1024-grid
  └──────┬──────────────────────────┘
         │ gated_{split}_records.csv
         ▼
  ┌────────────────────┐   ┌────────────────────┐
  │   expert_small     │   │   expert_large      │  Stage 2 — this package
  │   CNNExpert        │   │   CNNExpert         │
  └────────────────────┘   └────────────────────┘
         │                         │
         └─────────────────────────┘
                       │
              refined masks (N, 1024, 1024) as .npz
```

---

## Directory Structure

```
src/cnnNoROI/
├── cnnMoE.py               # Model definition
├── losses.py               # Training loss functions
├── train_moe.py            # Training script
├── infer_moe.py            # Inference script
├── __init__.py             # Makes this a Python package
└── jobs-slurmOutputs/      # Slurm job scripts and their .out logs
    ├── 1_train_moe_cnn.job
    ├── 2_infer_moe.job
    ├── 3_evaluate_medsam_baseline.job
    ├── 4_evaluate_moe.job
    ├── 5_visualize_moe.job
    └── slurm_*.out          # Outputs from completed jobs
```

---

## Scripts

### `cnnMoE.py` — Model Definition

Defines `CNNExpert`, a lightweight 3-layer CNN spatial decoder (~371k parameters, ~1.5 MB).

**Intuition:** MedSAM has already extracted rich image features into its 256-channel embedding.
The CNN's job is not feature extraction but spatial *decoding* — combining those features with the
coarse mask shape (binary mask + SDF) to produce a sharper boundary prediction.

**Input:** `(B, 258, 64, 64)` — 256 embedding channels + 1 binary mask channel + 1 SDF channel  
**Output:** `(B, 1, 64, 64)` — refined mask probabilities in [0, 1] (sigmoid output)

Two separate instances of `CNNExpert` are trained: one for small fragments, one for large.

**External dependencies:** none beyond PyTorch.

---

### `losses.py` — Training Losses

Provides the boundary-focused combined loss used during training.

**Components:**

- **`dice_loss(pred, gt)`** — Soft Dice loss. Measures global foreground overlap.
  Robust to class imbalance (important for small fragments).

- **`_boundary_weight_map(gt, w_boundary, radius)`** — Builds a per-pixel weight map
  that upweights pixels near GT mask edges. The boundary is approximated as
  `dilated(gt) - eroded(gt)` using max-pool operations, which runs entirely on GPU
  without scipy.

- **`boundary_weighted_bce(pred, gt)`** — Standard BCE with the boundary weight map
  applied per pixel. Forces the model to focus on getting edges right.

- **`boundary_dice_bce_loss(pred, gt)`** — Final combined loss:
  `0.5 x Dice + 0.5 x boundary-weighted BCE` (weights configurable via `--dice-weight`).

**External dependencies:** PyTorch only.

---

### `train_moe.py` — Training Script

Trains one or both CNN experts end-to-end.

**Per-batch pipeline:**
```
binary_mask (B, 1, 1024, 1024)
    │  interpolate (nearest) to 64x64
    ▼
mask_small (B, 1, 64, 64)
    │  compute SDF on CPU via sdf_utils.sdf_channel_from_mask()
    ▼
sdf (B, 1, 64, 64)
    │  cat with embedding
    ▼
x (B, 258, 64, 64) ──► CNNExpert ──► pred (B, 1, 64, 64)
    │  upsample (bilinear) to 448x448
    ▼
pred_up (B, 1, 448, 448) ──► boundary_dice_bce_loss ──► gt_mask (B, 1, 448, 448)
```

**Checkpoints:** saves `{expert_id}_best.pth` (lowest validation loss) and `{expert_id}_final.pth`
to `checkpoints/cnnNoROI/`.

**WandB integration:** each expert logs to the `moe-shaprefine` project under a separate run.
WandB calls are isolated in `wandb_init/wandb_log/wandb_finish` helpers — if WandB fails or
`--no-wandb` is passed, training continues unaffected. A local loss curve PNG is always saved.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--expert` | `both` | `expert_small`, `expert_large`, or `both` |
| `--epochs` | `20` | Training epochs per expert |
| `--batch-size` | `8` | Batch size |
| `--large-subsample` | `None` | Cap the large-fragment training set (balances small/large) |
| `--no-wandb` | off | Disable WandB; loss PNG still saved |
| `--wandb-project` | `moe-shaprefine` | WandB project name |

**External dependencies:**

| Module | Location |
|---|---|
| `build_expert_dataloaders()` | `src/gating_mechanism/dataset.py` |
| `sdf_channel_from_mask()` | `src/sdf_utils.py` |
| `CNNExpert` | `src/cnnNoROI/cnnMoE.py` |
| `boundary_dice_bce_loss` | `src/cnnNoROI/losses.py` |

The dataloader reads pre-computed gated CSVs from `src/gating_mechanism/gated_{split}_records.csv`,
which list per-fragment embedding paths, MedSAM mask paths, and expert assignments.

---

### `infer_moe.py` — Inference Script

Runs trained experts on a dataset split and produces refined mask files.

**Per-image pipeline:**
1. Load the shared ViT-B embedding `(256, 64, 64)` once.
2. Load all MedSAM binary masks `(N, 1024, 1024)` once.
3. For each fragment: look up its expert assignment from the gated CSV,
   resize mask → SDF → cat with embedding → forward → upsample to 1024×1024 → threshold at 0.5.
4. Stack refined masks `(N, 1024, 1024)` and save as `{sample_name}.npz` (key `"masks"`).

The output format is identical to MedSAM's `.npz` files, so the existing evaluator
(`evaluate_medsam_pengwin.py`) can read MoE predictions without modification.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--split` | `test` | Dataset split to run inference on |
| `--checkpoint-dir` | `checkpoints/cnnNoROI` | Directory with `*_best.pth` files |
| `--output-dir` | `data/moe-predictions/binary_masks` | Where to write `.npz` files |
| `--limit` | `None` | Process at most N images (smoke test) |

**External dependencies:**

| Module | Location |
|---|---|
| `load_prediction_masks()` | `src/dataloader_utils.py` |
| `resolve_existing_path()` | `src/dataloader_utils.py` |
| `sdf_channel_from_mask()` | `src/sdf_utils.py` |
| `CNNExpert` | `src/cnnNoROI/cnnMoE.py` |
| gated CSV | `src/gating_mechanism/gated_{split}_records.csv` |

---

## `jobs-slurmOutputs/` — Slurm Jobs

All jobs are submitted from the project root:
```bash
cd ~/projects/MoE-ShapeRefine-MedicalSeg
sbatch src/cnnNoROI/jobs-slurmOutputs/<job>.job
```

The jobs are numbered in execution order. Each depends on the previous completing successfully.

### `1_train_moe_cnn.job`
**Partition:** `gpu_a100` | **Time:** 4h | **GPUs:** 1

Trains both experts sequentially in one job (`--expert both`).
Uses `--large-subsample 17000` to balance the large-fragment training set against the smaller
small-fragment set, preserving SA/LI/RI class proportions within each expert.
Logs training and validation loss per epoch to WandB.

### `2_infer_moe.job`
**Partition:** `gpu_a100` | **Time:** 2h | **GPUs:** 1

Runs `infer_moe.py` on the test split.
Requires `1_train_moe_cnn.job` to have completed (needs `*_best.pth` checkpoints).
Output: `data/moe-predictions/binary_masks/{sample_name}.npz`

### `3_evaluate_medsam_baseline.job`
**Partition:** `rome` (CPU) | **Time:** 4h | **CPUs:** 16

Evaluates all 50k MedSAM Stage 1 predictions against PENGWIN ground truth (Dice, IoU, HD95, ASSD).
Only needs to be run once. Output: `data/medsam-predictions/evaluation_pengwin.csv`.
This CSV is required by `4_evaluate_moe.job` for the MoE vs MedSAM delta comparison.
Can run in parallel with jobs 1 and 2.

### `4_evaluate_moe.job`
**Partition:** `rome` (CPU) | **Time:** 2h | **CPUs:** 8

Evaluates MoE predictions and computes a per-fragment delta CSV (MoE minus MedSAM).
Requires jobs 2 and 3 to have completed.

Outputs:
- `data/moe-predictions/evaluation_moe.csv` — per-fragment metrics
- `data/moe-predictions/evaluation_delta.csv` — per-fragment MoE minus MedSAM delta

Also prints a formatted per-class / per-size summary table via `evaluation/summarize_eval_csv.py`.

### `5_visualize_moe.job`
**Partition:** `rome` (CPU) | **Time:** 30min | **CPUs:** 2

Generates 20 side-by-side PNG figures (5 best, 5 worst, 5 random small, 5 random large)
via `evaluation/visualize_moe_pengwin.py`. Requires job 4 (needs `evaluation_delta.csv`).

Each figure shows 5 panels cropped to the fragment bounding box:
```
X-ray  |  MedSAM overlay  |  MoE overlay  |  Ground truth  |  Diff
```
The diff panel colour-codes where MoE fixed vs broke the prediction relative to MedSAM.

Output: `data/moe-predictions/visualizations/*.png`

---

## External Dependencies Summary

| File | Location | Purpose |
|---|---|---|
| `src/dataloader_utils.py` | project `src/` | Path resolution, label decoding, mask loading |
| `src/sdf_utils.py` | project `src/` | Signed Distance Field computation |
| `src/gating_mechanism/dataset.py` | project `src/` | `build_expert_dataloaders()` and gated CSVs |
| `src/gating_mechanism/gated_*.csv` | project `src/` | Pre-computed fragment routing assignments |
| `evaluation/evaluate_medsam_pengwin.py` | project `evaluation/` | Core evaluator (called by evaluate_moe wrapper) |
| `evaluation/evaluate_moe_pengwin.py` | project `evaluation/` | MoE evaluation wrapper with delta computation |
| `evaluation/summarize_eval_csv.py` | project `evaluation/` | Formats per-class/per-size metric tables |
| `evaluation/visualize_moe_pengwin.py` | project `evaluation/` | Side-by-side visualisation figures |
| `checkpoints/cnnNoROI/` | project root | Trained model weights |
| `data/moe-predictions/` | project `data/` | Inference outputs, CSVs, visualizations |
| `data/medsam-predictions/` | project `data/` | MedSAM Stage 1 masks and metadata |

---

## Running the Full Pipeline

```bash
cd ~/projects/MoE-ShapeRefine-MedicalSeg

# Run training and MedSAM baseline evaluation in parallel
JID1=$(sbatch --parsable src/cnnNoROI/jobs-slurmOutputs/1_train_moe_cnn.job)
JID3=$(sbatch --parsable src/cnnNoROI/jobs-slurmOutputs/3_evaluate_medsam_baseline.job)

# Inference — depends on training
JID2=$(sbatch --parsable --dependency=afterok:$JID1 src/cnnNoROI/jobs-slurmOutputs/2_infer_moe.job)

# Evaluation — depends on inference AND MedSAM baseline
JID4=$(sbatch --parsable --dependency=afterok:$JID2:$JID3 src/cnnNoROI/jobs-slurmOutputs/4_evaluate_moe.job)

# Visualisation — depends on evaluation
sbatch --dependency=afterok:$JID4 src/cnnNoROI/jobs-slurmOutputs/5_visualize_moe.job
```
