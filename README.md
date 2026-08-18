# MoE-ShapeRefine-MedicalSeg

Mixture of Experts for Fine-Grained Shape Refinement in Medical Segmentation

A two-stage pipeline for X-ray fragment segmentation on the [PENGWIN](https://pengwin.grand-challenge.org/) dataset. Stage 1 uses [MedSAM](https://github.com/bowang-lab/MedSAM) (ViT-B) to produce coarse fragment masks and image embeddings. Stage 2 routes each fragment to a specialized expert model via a size-based gating mechanism, which refines the coarse mask using the MedSAM embedding as conditioning.

Three Stage 2 expert families are implemented:

| Expert | Input | Architecture | Notes |
|---|---|---|---|
| `cnnNoROI` | Full 64×64 feature map | Lightweight 3-layer CNN (~371k params) | Baseline refinement |
| `cnnROI` | RoI-cropped feature map | Same CNN, fragment-cropped | Spatial focus on fragment |
| `FlowSDF` | Full 64×64 feature map | UNet + RRDB flow matching | SDF-based generation |

---

## Repository Structure

```
MoE-ShapeRefine-MedicalSeg/
├── checkpoints/                        # Trained model weights
│   ├── cnnNoROI/                       # CNN MoE experts (no RoI crops)
│   ├── cnnROI/                         # CNN MoE experts (with RoI crops)
│   └── flowsdf/                        # FlowSDF MoE experts
│
├── data/                               # Dataset, predictions, and evaluation outputs
│   ├── bounding-boxes-xrays/           # X-ray images + bounding boxes for MedSAM
│   ├── medsam-predictions/             # Stage 1 outputs: binary_masks/ + embeddings/
│   ├── moe-predictions/                # CNN MoE (no RoI) refined masks + evaluation
│   ├── roi-predictions/                # CNN MoE (RoI) refined masks
│   └── flowsdf-moe-predictions/        # FlowSDF expert refined masks
│
├── evaluation/                         # Evaluation and visualization scripts
│   ├── evaluate_medsam_pengwin.py      # Core per-fragment evaluator (Dice, IoU, HD95, ASSD)
│   ├── evaluate_moe_pengwin.py         # MoE evaluation wrapper + delta vs. MedSAM
│   ├── compare_roi_noroi_pengwin.py    # Side-by-side cnnROI vs cnnNoROI comparison
│   ├── visualize_moe_pengwin.py        # Side-by-side PNG figures (best/worst/random)
│   └── summarize_eval_csv.py           # Per-class and per-size metric tables
│
├── src/                                # Source code
│   ├── cnnNoROI/                       # Stage 2: CNN MoE without RoI crops
│   │   ├── cnnMoE.py                   # CNNExpert model definition
│   │   ├── losses.py                   # Boundary-focused composite loss
│   │   ├── train_moe.py                # Training script
│   │   ├── infer_moe.py                # Inference script
│   │   └── README.md                   # Detailed module documentation
│   │
│   ├── cnnROI/                         # Stage 2: CNN MoE with RoI crops
│   │   ├── roi_utils.py                # RoI extraction helpers
│   │   ├── train_roi.py                # Training script
│   │   └── infer_roi.py                # Inference script
│   │
│   ├── FlowSDF/                        # Stage 2: Flow matching SDF expert
│   │   ├── train_flowsdf_moe.py        # Training script
│   │   ├── infer_flowsdf_moe.py        # Inference script
│   │   ├── trainer.py                  # Training loop utilities
│   │   ├── sampler.py                  # ODE-based SDF sampling
│   │   ├── datasets/                   # Data loader adapters
│   │   ├── models/                     # UNet + RRDB architecture
│   │   │   ├── unet_segdiff.py         # UNet with timestep embeddings and attention
│   │   │   └── RRDB.py                 # Residual in Residual Dense Block
│   │   ├── cfg/                        # YAML configs (monuseg.yaml, glas.yaml)
│   │   └── ExpertSetup.md              # FlowSDF training setup documentation
│   │
│   ├── gating_mechanism/               # Fragment routing by size
│   │   ├── gating_mechanism.py         # Runs once — gates all fragments, saves CSVs
│   │   ├── dataset.py                  # FragmentDataset + build_expert_dataloaders()
│   │   └── README.md                   # Gating and dataset documentation
│   │
│   ├── sdf_utils.py                    # Signed Distance Field computation utilities
│   ├── dataloader_utils.py             # Path resolution, label decoding, mask loading
│   ├── pengwin_utils.py                # PENGWIN-specific helpers (augmentation, categories)
│   ├── download_pengwin.py             # Dataset download script
│   ├── prepare_pengwin_xray_boxes_for_medsam_inference.py
│   └── run_medsam_with_pengwin_boxes.py
│
├── data_distribution/                  # Dataset analysis scripts
│   ├── analyze_dataset.py              # Extract shape features + MedSAM Dice per fragment
│   ├── analyze_area_threshold.py       # Evaluate gating threshold candidates from features.csv
│   ├── replot_from_features.py         # Regenerate plots from an existing features.csv
│   ├── plotting.py                     # Plot and example-image helpers
│   └── shape_features.py              # Shape feature extraction (area, elongation, compactness…)
│
├── data_distribution_results/          # Pre-computed analysis outputs (features.csv excluded — regenerate locally)
│   ├── data_analysis_gt_with_medsam/   # Train/val split: GT shape features + MedSAM Dice
│   │   ├── plots/                      # Distribution and correlation plots
│   │   ├── poster/plots/               # Poster-formatted versions of the same plots
│   │   ├── report/plots/               # Report-formatted versions
│   │   ├── examples/                   # Example fragment overlays (largest, smallest, most elongated…)
│   │   ├── area_threshold_analysis/    # Rolling Dice vs. area, threshold candidate CSVs
│   │   ├── medsam_failure_summary_*.csv
│   │   └── threshold_suggestions.csv
│   ├── data_analysis_medsam/           # Train/val split: MedSAM prediction shape features (no GT Dice)
│
├── medSAM-stage1/                      # Drop-in files for the MedSAM fork (Stage 1)
│   ├── scripts/
│   │   ├── prepare_pengwin_xray_for_medsam.py  # PENGWIN → MedSAM training format
│   │   ├── view_pengwin.py             # Interactive dataset viewer
│   │   └── pengwin_prep_smoke.job      # Slurm smoke-test job
│   ├── train_one_gpu.py                # Patched: accepts 3D instance masks
│   ├── train_multi_gpus.py             # Patched: accepts 3D instance masks
│   ├── trial_run.py                    # Single-image inference test
│   ├── medsam_env.job                  # Slurm environment setup job
│   └── README.md                       # Setup instructions for the fork
│
├── snellius-scripts/                   # Slurm HPC job submission scripts
├── fix_checkpoint.py                   # MedSAM checkpoint format fixer
└── environment.yml                     # Conda environment (Python 3.10, CUDA 11.8)
```

---

## MedSAM Fork (Stage 1 Baseline)

Stage 1 predictions are produced using a fork of the official [MedSAM repository](https://github.com/bowang-lab/MedSAM). All files needed to reproduce Stage 1 are included in [`medSAM-stage1/`](medSAM-stage1/). Clone the upstream repo and copy that directory into it:

```bash
git clone https://github.com/bowang-lab/MedSAM.git
cp -r medSAM-stage1/* MedSAM/
```

The table below describes each file and where it fits.

| File | Purpose |
|---|---|
| `scripts/prepare_pengwin_xray_for_medsam.py` | Converts PENGWIN X-ray data into MedSAM's 2D training format |
| `scripts/view_pengwin.py` | Interactive viewer for PENGWIN Task 1 (CT) and Task 2 (X-ray) |
| `scripts/pengwin_prep_smoke.job` | Slurm job to smoke-test the preprocessing pipeline |
| `trial_run.py` | Standalone inference script for testing MedSAM on a single image |
| `medsam_env.job` | Slurm job to provision the `medsam` conda environment on HPC |
| `train_one_gpu.py` *(modified)* | Extended `NpyDataset` to accept 3D instance-mask stacks |
| `train_multi_gpus.py` *(modified)* | Same change as above for multi-GPU training |

### Data preprocessing — `scripts/prepare_pengwin_xray_for_medsam.py`

Converts PENGWIN Task 2 X-ray images and their 30-bit encoded segmentation labels into MedSAM's NumPy training format.

```bash
# Smoke test (5 samples)
sbatch scripts/pengwin_prep_smoke.job

# Full run
python scripts/prepare_pengwin_xray_for_medsam.py \
  --output-root /path/to/derived/medsam/xray
```

Input: PENGWIN `original/` directory with `.tif` image/label pairs.

Output layout:

```text
<output_root>/
  imgs/           # float32 (1024, 1024, 3), neg-log normalized, [0, 1]
  gts/            # uint8 (N, 1024, 1024), one channel per fragment instance
  metadata.jsonl  # per-sample records with fragment details and paths
```

Key options:
- `--limit 5` — process only 5 cases (smoke test)
- `--keep-empty` — retain images with no positive fragments (skipped by default)

### Dataset viewer — `scripts/view_pengwin.py`

Interactive matplotlib viewer for exploring the PENGWIN dataset before training.

```bash
# Browse X-ray cases interactively
python scripts/view_pengwin.py --task xray

# Jump to a specific case
python scripts/view_pengwin.py --task xray --case 001_0000

# CT with custom overlay opacity
python scripts/view_pengwin.py --task ct --alpha 0.5
```

Keyboard shortcuts: `←` / `→` to navigate cases, `↑` / `↓` for CT slices, `o` to toggle overlay.

### Training on PENGWIN — modified `NpyDataset`

The upstream `train_one_gpu.py` and `train_multi_gpus.py` expect 2D single-label ground truth masks. The fork patches `NpyDataset.__getitem__` to also handle the 3D instance-mask stacks produced by `prepare_pengwin_xray_for_medsam.py`:

- If `gt.ndim == 3`: randomly samples one instance channel per training step.
- If `gt.ndim == 2`: falls back to the original label-sampling logic.

This change is the only modification needed to fine-tune MedSAM on PENGWIN fragment data.

```bash
# Single GPU fine-tuning
python train_one_gpu.py \
  --tr_npy_path /path/to/derived/medsam/xray \
  --medsam_checkpoint work_dir/MedSAM/medsam_vit_b.pth \
  --max_epoch 10 \
  --batch_size 4 \
  --task_name PENGWIN_xray

# Multi-GPU fine-tuning
bash train_multi_gpus.sh
```

### Environment setup — `medsam_env.job`

```bash
sbatch medsam_env.job
```

Creates or updates the `medsam` conda environment on Snellius (Python 3.10, PyTorch 2.0, CUDA 11.8).

### Single-image inference test — `trial_run.py`

```bash
srun python trial_run.py \
  -i assets/img_demo.png \
  -o results \
  --box 95,200,190,200 \
  --checkpoint work_dir/MedSAM/medsam_vit_b.pth
```

Saves the predicted mask as `results/mask.npy` and an overlay as `results/overlay.png`.

---

## Environment Setup

### MedSAM / CNN MoE environment

```bash
# Create environment — only needed once
conda env create -f environment.yml

# Activate
conda activate MoE
```

### FlowSDF environment

A separate `flowSDF` conda environment is required for the FlowSDF expert:

```bash
sbatch snellius-scripts/job/create_flowSDF_MoE_env.job

# After completion:
conda activate flowSDF
```

### Fix MedSAM checkpoint format

```bash
# Only needed once, before running Stage 1 inference
srun python fix_checkpoint.py
```

---

## Full Pipeline Overview

```
Raw X-ray (1024×1024)
        │
        ▼
 ┌─────────────┐
 │   MedSAM    │  Stage 1 — pre-computed once
 │  (ViT-B)   │
 └──────┬──────┘
        │ embedding (256, 64, 64)  +  binary_mask (N, 1024, 1024)
        ▼
 ┌─────────────────────────┐
 │   Gating Mechanism      │  routes each fragment by bounding-box area
 └──────┬──────────────────┘
        │ gated_{split}_records.csv
        ▼
 ┌──────────────────┐   ┌──────────────────┐
 │  expert_small    │   │  expert_large    │  Stage 2 — one of three approaches
 └──────────────────┘   └──────────────────┘
        │                       │
        └───────────────────────┘
                     │
          refined masks (N, 1024, 1024) as .npz
```

---

## Step 1 — Download PENGWIN Data

```bash
python src/download_pengwin.py --task xray
```

Creates the canonical dataset layout under:

```text
data/pengwin/
  raw/
  original/
```

---

## Step 2 — Prepare Bounding Boxes for MedSAM

Prepares normalized X-ray images and per-fragment bounding boxes as MedSAM box prompts.

```bash
python src/prepare_pengwin_xray_boxes_for_medsam_inference.py
```

Output written to `data/bounding-boxes-xrays/` by default:

```text
data/bounding-boxes-xrays/
  imgs/             (empty — populated by MedSAM inference)
  boxes/
    XRAY_PENGWIN_001_0000.npy   # float32 (N, 4), xyxy on 1024×1024 grid
    ...
  metadata.jsonl
```

Useful options:
- `--bbox-pad 5` — add padding around each box
- `--limit 10` — smoke test on 10 samples
- `--keep-empty` — retain images with no positive fragments
- `--resume` — skip already-processed samples
- `--overwrite` — force regeneration

---

## Step 3 — MedSAM Inference (Stage 1)

For each image: runs the ViT-B encoder once and the mask decoder once per fragment (batched).

```bash
# Submit on Snellius
sbatch run_inference_extract_features.job

# Or run directly
python src/run_medsam_with_pengwin_boxes.py
```

Reads from `data/bounding-boxes-xrays/`, writes to `data/medsam-predictions/`:

```text
data/medsam-predictions/
  binary_masks/
    XRAY_PENGWIN_001_0000.npz   # uint8 (N, 1024, 1024), key "masks"
    ...
  embeddings/
    XRAY_PENGWIN_001_0000.npy   # float16 (256, 64, 64)
    ...
  metadata.jsonl
```

Load outputs:

```python
import numpy as np

masks     = np.load("data/medsam-predictions/binary_masks/XRAY_PENGWIN_001_0000.npz")["masks"]
# shape: (N, 1024, 1024), uint8

embedding = np.load("data/medsam-predictions/embeddings/XRAY_PENGWIN_001_0000.npy").astype(np.float32)
# shape: (256, 64, 64)
```

Useful options:
- `--resume` — continue an interrupted run
- `--overwrite` — reprocess everything
- `--limit 10` — smoke test
- `--threshold 0.5` — binary mask cutoff
- `--case-id 001_0000` or `--sample-name XRAY_PENGWIN_001_0000` — single sample

---

## Step 4 — Gating Mechanism (run once)

Routes each fragment to `expert_small` or `expert_large` based on foreground pixel count on the 1024×1024 grid.

**Threshold:** `area ≤ 5,402` pixels (≈0.52% of 1024×1024) → `expert_small`, else → `expert_large`.

```bash
# Full run
python src/gating_mechanism/gating_mechanism.py

# Smoke test (5 cases per split)
python src/gating_mechanism/gating_mechanism.py --smoke
```

Output CSVs (one row per fragment):

```text
src/gating_mechanism/
  gated_train_records.csv
  gated_val_records.csv
  gated_test_records.csv
```

| Column | Description |
|---|---|
| `sample_name` | e.g. `XRAY_PENGWIN_001_0000` |
| `medsam_instance_id` | Index into the `.npz` masks array (1-based) |
| `category_name` | SA / LI / RI |
| `area` | Foreground pixel count on 1024×1024 |
| `expert` | `expert_small` or `expert_large` |
| `embedding_path` | Path to `.npy` image embedding |
| `binary_masks_path` | Path to `.npz` MedSAM masks |

---

## Dataset Analysis

The `data_distribution/` scripts characterise the PENGWIN fragment distribution and motivated the gating threshold. Pre-computed outputs (plots, summary CSVs, example images) are in `data_distribution_results/`. The large per-fragment `features.csv` files are excluded from the repo (gitignored) and must be regenerated locally.

### Scripts

| Script | Purpose |
|---|---|
| `analyze_dataset.py` | Extract per-fragment shape features and MedSAM Dice; writes `features.csv` + plots |
| `analyze_area_threshold.py` | Sweep area thresholds on an existing `features.csv`; outputs rolling Dice/failure-rate curves and threshold candidate CSVs |
| `replot_from_features.py` | Regenerate all plots and example images from an existing `features.csv` without reloading masks |
| `plotting.py` | Shared plotting helpers (distribution plots, scatter plots, example overlays) |
| `shape_features.py` | Feature extraction functions: area, aspect ratio, elongation, compactness, connected components |

### Running the analysis

```bash
# Full analysis — reads GT masks and MedSAM predictions, writes features.csv + all plots
python data_distribution/analyze_dataset.py \
  --output-dir data_distribution_results/data_analysis_gt_with_medsam \
  --split train

# Test split
python data_distribution/analyze_dataset.py \
  --output-dir data_distribution_results/data_analysis_gt_with_medsam_test \
  --split test

# Regenerate plots only (fast — skips mask loading)
python data_distribution/replot_from_features.py \
  --features-csv data_distribution_results/data_analysis_gt_with_medsam/features.csv \
  --output-dir data_distribution_results/data_analysis_gt_with_medsam

# Threshold analysis (informs gating mechanism cutoff)
python data_distribution/analyze_area_threshold.py \
  --features-csv data_distribution_results/data_analysis_gt_with_medsam/features.csv \
  --output-dir data_distribution_results/data_analysis_gt_with_medsam/area_threshold_analysis
```

### Pre-computed results (`data_distribution_results/`)

Each subdirectory covers one split and mask source:

| Directory | Split | Mask source |
|---|---|---|
| `data_analysis_gt_with_medsam/` | train/val | GT shape features + MedSAM Dice per fragment |
| `data_analysis_gt_with_medsam_test/` | test | GT shape features + MedSAM Dice per fragment |
| `data_analysis_medsam/` | train/val | MedSAM prediction shape features only |
| `data_analysis_medsam_test/` | test | MedSAM prediction shape features only |

Each directory contains:
- `plots/` — area distribution, elongation, compactness, MedSAM Dice vs. shape features (per class)
- `poster/plots/` and `report/plots/` — same plots restyled for poster/report use
- `examples/` — annotated fragment overlay PNGs: largest, smallest, most elongated, most complex, most fragmented
- `area_threshold_analysis/` — rolling Dice and failure-rate curves, threshold candidate tables *(train split only)*
- `medsam_failure_summary_*.csv` — failure rates grouped by shape group and anatomy class *(GT splits only)*
- `threshold_suggestions.csv` — ranked area threshold candidates with Dice/failure-rate trade-offs

> `features.csv` (150 MB per split) is gitignored. Regenerate with `analyze_dataset.py`.

---

## Step 5 — Stage 2: Train & Infer

Choose one of three Stage 2 expert approaches below.

---

### Option A — CNN MoE (No RoI)

A lightweight 3-layer CNN decoder (~371k parameters) that processes the full 64×64 feature map.

**Input:** `(B, 258, 64, 64)` — 256 MedSAM embedding channels + 1 binary mask + 1 SDF  
**Output:** `(B, 1, 64, 64)` → upsampled to `(1, 1024, 1024)`

**Loss:** `0.5 × Dice + 0.5 × boundary-weighted BCE`

#### Train

```bash
python src/cnnNoROI/train_moe.py \
  --expert both \
  --epochs 20 \
  --batch-size 64 \
  --large-subsample 17000
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--expert` | `both` | `expert_small`, `expert_large`, or `both` |
| `--epochs` | `20` | Training epochs per expert |
| `--batch-size` | `8` | Batch size |
| `--large-subsample` | `None` | Cap expert_large training set (balances small/large) |
| `--no-wandb` | off | Disable WandB; loss PNG still saved locally |

Checkpoints saved to `checkpoints/cnnNoROI/{expert_id}_best.pth` and `{expert_id}_final.pth`.

#### Infer

```bash
python src/cnnNoROI/infer_moe.py --split test
```

Output: `data/moe-predictions/binary_masks/{sample_name}.npz` (key `"masks"`)

| Argument | Default | Description |
|---|---|---|
| `--split` | `test` | Dataset split |
| `--checkpoint-dir` | `checkpoints/cnnNoROI` | Directory with `*_best.pth` |
| `--output-dir` | `data/moe-predictions/binary_masks` | Output directory |
| `--limit` | `None` | Process at most N images |

#### Slurm jobs (in `src/cnnNoROI/jobs-slurmOutputs/`)

```bash
cd ~/projects/MoE-ShapeRefine-MedicalSeg

# Training and baseline evaluation run in parallel
JID1=$(sbatch --parsable src/cnnNoROI/jobs-slurmOutputs/1_train_moe_cnn.job)
JID3=$(sbatch --parsable src/cnnNoROI/jobs-slurmOutputs/3_evaluate_medsam_baseline.job)

# Inference — depends on training
JID2=$(sbatch --parsable --dependency=afterok:$JID1 src/cnnNoROI/jobs-slurmOutputs/2_infer_moe.job)

# Evaluation — depends on inference AND baseline
JID4=$(sbatch --parsable --dependency=afterok:$JID2:$JID3 src/cnnNoROI/jobs-slurmOutputs/4_evaluate_moe.job)

# Visualisation — depends on evaluation
sbatch --dependency=afterok:$JID4 src/cnnNoROI/jobs-slurmOutputs/5_visualize_moe.job
```

| Job | Partition | Time | Resources |
|---|---|---|---|
| `1_train_moe_cnn.job` | `gpu_a100` | 4h | 1 GPU |
| `2_infer_moe.job` | `gpu_a100` | 2h | 1 GPU |
| `3_evaluate_medsam_baseline.job` | `rome` (CPU) | 4h | 16 CPUs |
| `4_evaluate_moe.job` | `rome` (CPU) | 2h | 8 CPUs |
| `5_visualize_moe.job` | `rome` (CPU) | 30min | 2 CPUs |

---

### Option B — CNN MoE (With RoI)

Same architecture as Option A, but the embedding is cropped to the fragment's bounding box before processing, providing spatial focus.

#### Train

```bash
python src/cnnROI/train_roi.py \
  --expert both \
  --epochs 20 \
  --batch-size 64
```

#### Infer

```bash
python src/cnnROI/infer_roi.py --split test
```

Output: `data/roi-predictions/binary_masks/{sample_name}.npz`

---

### Option C — FlowSDF MoE

Flow matching model with a UNet + RRDB backbone that generates refined SDFs conditioned on MedSAM features.

**Architecture:**
- Input conditioning: `(257, H, W)` — 256 MedSAM embedding + 1 coarse mask
- Model channels: 128, multipliers `(1, 1, 2, 2, 4, 4)`, attention at resolutions 16 and 8
- 12 RRDB blocks per level
- EMA model averaging for stable inference

**Data pipeline:**
```
MedSAM outputs → resize to img_size (128×128) → embed + coarse mask as conditioning
gt_mask → FlowSDF SDF → flow matching target
```

#### Train

```bash
# Snellius
sbatch src/FlowSDF/train_flowsdf_moe.job   # 8h on A100

# Local
python src/FlowSDF/train_flowsdf_moe.py \
  --expert both \
  --epochs 100 \
  --batch-size 8 \
  --img-size 128
```

Checkpoints saved to `checkpoints/flowsdf/{expert_id}_best.pth` (contains both `model_state` and `ema_state`).

#### Infer

```bash
# Smoke test on val
python src/FlowSDF/infer_flowsdf_moe.py \
  --split val \
  --checkpoint-dir checkpoints/flowsdf \
  --limit 10

# Full test split
python src/FlowSDF/infer_flowsdf_moe.py \
  --split test \
  --checkpoint-dir checkpoints/flowsdf \
  --output-root data/flowsdf-moe-predictions
```

Output: `data/flowsdf-moe-predictions/<split>/binary_masks/{sample_name}.npz`

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--split` | `test` | Dataset split |
| `--img-size` | `128` | Feature map resolution |
| `--ode-steps` | `100` | Euler integration steps |
| `--sdf-binary-threshold` | `0.03` | SDF → binary mask threshold |
| `--use-model-state` | off | Use `model_state` instead of `ema_state` |

---

## Step 6 — Evaluate

### Evaluate MedSAM baseline (Stage 1)

```bash
# Fast (Dice + IoU only)
python evaluation/evaluate_medsam_pengwin.py \
  --skip-boundary \
  --num-workers 4 \
  --output-csv data/medsam-predictions/eval_overlap_full.csv

# Full with boundary metrics (HD95, ASSD) on a subset
python evaluation/evaluate_medsam_pengwin.py \
  --limit 1000 \
  --num-workers 4 \
  --output-csv data/medsam-predictions/eval_boundary_1000.csv
```

Evaluation is done per fragment. The script prints subgroup summaries by anatomy class (`SA`, `LI`, `RI`) and size group (`small`, `large`).

### Evaluate MoE predictions (Stage 2)

```bash
# Fast mode
python evaluation/evaluate_moe_pengwin.py --skip-boundary

# Full with boundary metrics
python evaluation/evaluate_moe_pengwin.py
```

Outputs:
- `data/moe-predictions/evaluation_moe.csv` — per-fragment metrics
- `data/moe-predictions/evaluation_delta.csv` — MoE minus MedSAM delta per fragment

### Compare cnnROI vs cnnNoROI

```bash
python evaluation/compare_roi_noroi_pengwin.py
```

### Summarize metrics as tables

```bash
python evaluation/summarize_eval_csv.py data/moe-predictions/evaluation_moe.csv
```

---

## Step 7 — Visualize

Generates 20 side-by-side PNG figures: 5 best, 5 worst, 5 random small, 5 random large fragments.

```bash
python evaluation/visualize_moe_pengwin.py
```

Each figure shows (cropped to fragment bounding box):

```
X-ray  |  MedSAM overlay  |  MoE overlay  |  Ground truth  |  Diff
```

The diff panel colour-codes where MoE fixed vs broke the prediction relative to MedSAM.

Output: `data/moe-predictions/visualizations/*.png`

---

## Using MedSAM Outputs as MoE Inputs

Each MoE training sample corresponds to one fragment:

| Input | Source | Shape |
|---|---|---|
| Initial mask | `binary_masks/*.npz`, slice `[i]` | `(1, 1024, 1024)` uint8 |
| Image embedding | `embeddings/*.npy` | `(256, 64, 64)` float32 |
| Bounding box | `metadata.jsonl → fragments[i]["bbox_xyxy_1024"]` | `(4,)` float32 |
| Category | `metadata.jsonl → fragments[i]["category_name"]` | SA / LI / RI |

For per-fragment features via RoI-align:

```python
import torchvision.ops as ops

# embedding: (1, 256, 64, 64) tensor, boxes in 1024-grid coords → scale to 64-grid
scale = 64 / 1024
box_64 = [x * scale for x in bbox_xyxy_1024]
roi = ops.roi_align(embedding.unsqueeze(0), [torch.tensor([box_64])], output_size=(7, 7))
# roi: (1, 256, 7, 7)
```

Building a DataLoader directly from gated CSVs:

```python
from src.gating_mechanism.dataset import build_expert_dataloaders

loaders = build_expert_dataloaders(
    split="train",
    batch_size=8,
    num_workers=4,
    large_subsample=17000,   # balance ~194k large vs ~17k small fragments
)

for batch in loaders["expert_small"]:
    embedding   = batch["embedding"]    # (B, 256, 64, 64)
    binary_mask = batch["binary_mask"]  # (B, 1, 1024, 1024)
    gt_mask     = batch["gt_mask"]      # (B, 1, 448, 448)
```

---

## Dependencies

Core environment (`environment.yml`, conda `MoE`):

| Package | Purpose |
|---|---|
| Python 3.10, PyTorch 2.0, CUDA 11.8 | Framework |
| numpy, scipy | Numerical computing, SDF computation |
| pandas | CSV / gating data |
| opencv-python, scikit-image, pillow | Image processing, PENGWIN I/O |
| matplotlib | Visualization |
| transformers 4.35–4.45, datasets, accelerate | HuggingFace stack |
| segment-anything | MedSAM model |
| wandb | Experiment tracking (optional) |

FlowSDF environment (`flowSDF` conda, `src/FlowSDF/requirements.txt`):

| Package | Purpose |
|---|---|
| torchdiffeq | ODE integration (original sampler) |
| tensorboard | Training monitoring |
| pyyaml | Config parsing |
| h5py | HDF5 data loading |
| torchmetrics==0.11.1 | Evaluation metrics |
