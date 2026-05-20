# FlowSDF MoE Training Setup — Summary

Describe complete training pipeline for FlowSDF experts on MoE-gated X-ray fragment data.


### Training Scripts (in `src/FlowSDF/`)

**`train_flowsdf_moe.py`** — Main training script
- Wraps gated dataloader data into FlowSDF format
- Trains `expert_small` and `expert_large` separately
- Features:
  - Automatic data resizing (MedSAM embedding + binary masks + gt_mask)
  - GT mask to FlowSDF SDF conversion
  - MedSAM embedding + coarse mask conditioning
  - Flow matching loss (MSE velocity prediction)
  - EMA model averaging for stability
  - Best checkpoint tracking + loss curve visualization
- Supports subsampling `expert_large` to balance dataset sizes

**`infer_flowsdf_moe.py`** — FlowSDF MoE inference script
- Loads trained `expert_small` and `expert_large` checkpoints
- Rebuilds the same FlowSDF UNet/RRDB expert architecture used in training
- Recreates the training-time conditioner: `(256 MedSAM embedding + 1 coarse mask, H, W)`
- Runs FlowSDF Euler sampling per routed fragment
- Converts sampled SDFs to binary masks and upsamples to 1024×1024
- Saves `.npz` files with key `masks`, matching the MedSAM/CNN MoE evaluator convention

**`train_flowsdf_moe.job`** — Snellius job script
- Partition: `gpu_a100` (full A100 GPU)
- Time: 8 hours
- Default settings: 100 epochs, batch_size=8, num_workers=8, img_size=128×128
- Output: `slurm_train_flowsdf_*.out` logs

**`TRAINING.md`** — Complete usage documentation
- Prerequisites (gating mechanism setup)
- Local training vs. Snellius submission
- All command-line arguments
- Output checkpoint format
- Next steps for inference/evaluation

### Environment Setup (in `snellius-scripts/job/`)

**`create_flowSDF_MoE_env.job`** — New environment creation script
- Creates `flowSDF` conda environment (Python 3.11)
- Installs FlowSDF base requirements
- Adds MoE-specific dependencies: scipy, scikit-image, pillow, opencv-python
- Verifies all imports and GPU availability

**Updated `create_flowSDF_env.job`**
- Added MoE dependencies to existing setup

---

## Quick Start

### 1. Create environment on Snellius (one-time)

```bash
sbatch snellius-scripts/job/create_flowSDF_MoE_env.job

# Wait for completion, then:
source activate flowSDF
```

### 2. Prepare gated data (one-time)

```bash
python gating_mechanism/gating_mechanism.py
```

### 3. Train on Snellius

```bash
sbatch train_flowsdf_moe.job
```

Or run locally:
```bash
python train_flowsdf_moe.py --expert both --epochs 50 --batch-size 8
```

### 4. Run FlowSDF MoE inference

Smoke test on the validation split:
```bash
python infer_flowsdf_moe.py \
  --split val \
  --checkpoint-dir ../../checkpoints/flowsdf \
  --limit 10
```

Full test split:
```bash
python infer_flowsdf_moe.py \
  --split test \
  --checkpoint-dir ../../checkpoints/flowsdf \
  --output-root ../../data/flowsdf-moe-predictions
```

---

## Data Pipeline

```
MedSAM outputs (gating_mechanism)
├── binary_mask (1, 1024, 1024)  ← coarse MedSAM prediction
├── embedding (256, 64, 64)       ← ViT-B image encoder output
└── gt_mask (1, 448, 448)         ← ground truth from PENGWIN

         ↓ MoEFlowSDFDataset

Resized to img_size (default 128×128)
├── image (257, 128, 128)  ← conditioning: embedding + coarse mask
└── mask (128, 128)        ← FlowSDF SDF target from gt_mask

         ↓ FlowSDF training

Flow matching loss on predicted velocity vs target velocity
```

---

## FlowSDF MoE Inference Pipeline

`infer_flowsdf_moe.py` is the MoE equivalent of FlowSDF's original `sampler.py`. The stock sampler loads a trained FlowSDF checkpoint, takes `batch["image"]` as the conditioning tensor, starts from Gaussian noise, integrates the learned velocity field with Euler ODE steps, then thresholds the final SDF. The MoE script keeps that same sampling logic and changes only the data adapter, expert routing, checkpoint format, and output packaging.

### Step-by-step inference

1. **Load routed fragment records**
   - Reads `gating_mechanism/gated_<split>_records.csv`.
   - Groups records by `sample_name`.
   - Uses each row's `expert` column to choose `expert_small` or `expert_large`.

2. **Load trained experts**
   - Loads `expert_small_best.pth` and `expert_large_best.pth` by default.
   - Uses `ema_state` by default because training maintains EMA weights for stable inference.
   - Rebuilds the same architecture as training:
     - `n_cin = 1`
     - `n_fm = 128`
     - `channel_mult = (1, 1, 2, 2, 4, 4)`
     - `attention_resolutions = (16, 8)`
     - `rrdb_blocks = 12`
     - `img_cond_channels = 257`

3. **Load image-level MedSAM assets once**
   - Loads the shared MedSAM embedding from `embedding_path`, shape `(256, 64, 64)`.
   - Loads all MedSAM coarse masks from `binary_masks_path`, shape `(N, 1024, 1024)`.

4. **Build the FlowSDF conditioner per fragment**
   - Resizes the MedSAM embedding to `(256, img_size, img_size)` with bilinear interpolation.
   - Resizes the fragment's MedSAM coarse mask to `(1, img_size, img_size)` with nearest-neighbor interpolation.
   - Thresholds the coarse mask at `0.5`.
   - Concatenates them into `(1, 257, img_size, img_size)`.

5. **Sample a refined SDF**
   - Starts from Gaussian noise `m0`, same as `sampler.py`.
   - Uses the same conditional vector field:
     ```python
     m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
     v = model(m_ipt, t[:, None], img_cond=conditioner)
     ```
   - Integrates from `t=0` to `t=1` with Euler steps.
   - The script uses an explicit PyTorch Euler loop instead of importing `torchdiffeq.odeint`; this is equivalent to the original sampler's `method="euler"` path and avoids a SciPy dependency at CLI import time.

6. **Convert SDF to binary mask**
   - The original FlowSDF sampler thresholds SDF foreground as:
     ```python
     sampled_sdf <= 3 * thresh
     ```
   - With the original `thresh = 1e-2`, this is `0.03`.
   - `infer_flowsdf_moe.py` uses `--sdf-binary-threshold 0.03` by default.

7. **Return to MedSAM output format**
   - Upsamples each binary fragment mask to `(1024, 1024)`.
   - Stores refined masks in the original MedSAM instance order.
   - Writes:
     ```text
     data/flowsdf-moe-predictions/<split>/binary_masks/<sample_name>.npz
     ```
     with key `masks`.
   - Writes a companion metadata file:
     ```text
     data/flowsdf-moe-predictions/<split>/metadata.jsonl
     ```

### Important alignment with original FlowSDF

- Same UNet/RRDB backbone, adapted to `257` conditioning channels.
- Same SDF-valued generated variable.
- Same Gaussian initial state.
- Same conditional vector-field formula as `sampler.py`.
- Same Euler integration interval, `t=0 → 1`.
- Same foreground threshold convention for SDF outputs.

### Intentional MoE differences

- The conditioning tensor is MedSAM embedding + MedSAM coarse mask, not a raw image.
- There are two separately trained experts instead of one model.
- Expert selection comes from the gating CSV.
- Checkpoints use `ema_state` / `model_state`, not the stock FlowSDF `state_dict` field.
- Output is packaged as MedSAM-style `.npz` files so existing PENGWIN evaluation code can consume it.

---

## Key Design Choices

1. **Separate expert training**: Each expert (small/large) trained independently with their own routing data
2. **FlowSDF SDF targets**: GT fragment masks are converted to signed distance fields with the FlowSDF notebook recipe
3. **MedSAM feature conditioning**: RRDB receives `257` channels: 256 MedSAM embedding channels plus 1 coarse mask channel
4. **Flexible sizing**: `--img-size` parameter allows tuning resolution vs. speed tradeoff
5. **Checkpoint format**: Saves both model and EMA states for inference flexibility
6. **Subsampling**: `--large-subsample` balances dataset imbalance (~17k small, ~194k large)

---

## Dependencies Installed

| Package | Purpose |
|---------|---------|
| torch, torchvision | Deep learning framework |
| numpy, scipy | Numerical computing |
| pandas | Data handling (gating CSVs) |
| matplotlib | Loss curve plotting |
| opencv-python | Image processing |
| scikit-image | Image processing utilities |
| pillow | Image I/O (PENGWIN masks) |
| tqdm | Progress bars |
| pyyaml | Config parsing |
| tensorboard | Training monitoring |
| torchdiffeq | Original FlowSDF sampler dependency; MoE inference uses an explicit Euler loop |

---

## Next Steps

1. **Run training**: `sbatch train_flowsdf_moe.job` (8 hrs on A100)
2. **Run inference**: `python infer_flowsdf_moe.py --split val --checkpoint-dir ../../checkpoints/flowsdf`
3. **Evaluate**: Compare refined masks against ground truth
4. **Integrate**: Use outputs in the full MoE evaluation pipeline
5. **Tune**: Adjust `--img-size`, `--epochs`, `--large-subsample`, `--ode-steps`, and `--sdf-binary-threshold`

See `TRAINING.md` for complete documentation.
