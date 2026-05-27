# FlowSDF MoE Training

Offers documentation on the training of the FlowSDF experts separately on MoE-gated fragment data (expert_small and expert_large).

## Overview

**Pipeline per batch:**
1. Load embedding (256, 64, 64), binary_mask (1, 1024, 1024), gt_mask (1, 448, 448) from gated dataloaders
2. Resize the MedSAM embedding, binary mask, and GT fragment mask to training size (default 128×128)
3. Convert the GT fragment mask to a FlowSDF signed distance field (SDF)
4. Concatenate MedSAM embedding + binary mask as the conditioner: `(256 + 1, H, W)`
5. Train FlowSDF to predict the flow-matching velocity from noisy SDF samples to the GT SDF target
6. Loss: MSE flow matching loss

**Data routing:** Fragments are automatically routed by area (gating mechanism):
- `expert_small`: fragments with area ≤ 5402 pixels
- `expert_large`: fragments with area > 5402 pixels

---

## FlowSDF MoE Pipeline And Architecture

This training script follows the same FlowSDF workflow as `main.py`, `trainer.py`, and `sampler.py`, with the dataset adapter changed for PENGWIN/MedSAM MoE fragments.

### Data preprocessing

The original FlowSDF README expects images and precomputed SDF masks listed in `train.csv` and `test.csv`. For the MoE pipeline, `MoEFlowSDFDataset` performs the equivalent SDF preprocessing lazily per fragment:

1. `FragmentDataset` loads:
   - `embedding`: precomputed MedSAM ViT-B image embedding, `(256, 64, 64)`
   - `binary_mask`: MedSAM coarse mask for the routed instance, `(1, 1024, 1024)`
   - `gt_mask`: decoded PENGWIN fragment GT, `(1, 448, 448)`
2. `embedding` is resized bilinearly to `(256, img_size, img_size)`.
3. `binary_mask` is resized with nearest-neighbor interpolation to `(1, img_size, img_size)` and thresholded at `0.5`.
4. `gt_mask` is resized bilinearly to `(1, img_size, img_size)`, thresholded at `0.5`, then converted to FlowSDF format:
   - boundary: `abs(binary_erosion(mask) - mask)`
   - distance: `distance_transform_edt(boundary == 0)`
   - sign: negative inside foreground, positive outside
   - clipping: `[-sdf_threshold, +sdf_threshold]`, default `15`
   - normalization: divide by `sdf_threshold`, producing values in `[-1, 1]`
5. The batch item returned to FlowSDF is:
   - `image`: conditioning tensor `(257, img_size, img_size)` = MedSAM embedding + coarse mask
   - `mask`: target SDF tensor `(img_size, img_size)`

This matches the SDF recipe in `precompute_sdf.ipynb`; the difference is that the MoE version computes it on the fly from gated PENGWIN fragment masks instead of loading precomputed `.npy` SDF files.

### Flow matching workflow

For each batch, the script mirrors `trainer.TrainFlow.do()`:

1. Let `m` be the GT SDF target with shape `(B, 1, H, W)`.
2. Sample timestep `t ~ Uniform(0, 1)` for each item.
3. Sample Gaussian noise `eta` with the same shape as `m`.
4. Build the noisy SDF state:
   ```python
   sigma_t = 1 - (1 - sigma_min) * t
   mt = t * m + sigma_t * eta
   ```
5. Compute the target velocity:
   ```python
   u = (m - (1 - sigma_min) * mt) / (1 - (1 - sigma_min) * t)
   ```
6. Predict velocity with the conditional UNet:
   ```python
   v = model(mt, t[:, None], img_cond=image)
   ```
7. Optimize MSE:
   ```python
   loss = ((v - u) ** 2).mean()
   ```

Sampling in `sampler.py` uses the same vector field in reverse form through an Euler ODE solve. It starts from `m0 = randn_like(m_gt)` and repeatedly calls the model with:

```python
m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
net(m_ipt, t[:, None], img_cond=x)
```

### Model architecture

Each expert is a separate `UNetModel` instance trained on only that expert's routed fragments. The active configuration is:

| Component | Value |
|-----------|-------|
| Flow target channels | `n_cin = 1` |
| UNet base channels | `n_fm = 128` |
| Output channels | `1` velocity channel |
| Residual blocks per level | `3` |
| Channel multipliers | `(1, 1, 2, 2, 4, 4)` |
| Attention resolutions | `(16, 8)` |
| Dimensions | 2D |
| Class conditioning | disabled |
| Timestep embedding | sinusoidal embedding projected to `4 * n_fm = 512` |
| RRDB conditioner blocks | `12` |
| RRDB input channels | `257` = 256 MedSAM embedding channels + 1 coarse mask channel |
| RRDB output channels | `128`, added to the first UNet feature map |

The conditioning path is important: `UNetModel.forward()` first encodes `img_cond` with `RRDBNet`, then adds the resulting `(B, 128, H, W)` feature map into the first UNet block output. The noisy SDF `mt` remains the UNet input; the MedSAM embedding and coarse mask only condition the velocity prediction.

Compared with the stock FlowSDF scripts:

- `main.py` builds the same UNet/RRDB architecture and calls `trainer.TrainFlow.do()`.
- `sampler.py` loads the same architecture and integrates the learned velocity field with `torchdiffeq.odeint`.
- The MoE script uses MedSAM embeddings plus the coarse mask as conditioning instead of a raw image.
- The MoE script trains `expert_small` and `expert_large` separately and stores both raw and EMA state dicts in checkpoints.

---

## Prerequisites

### 1. Prepare Gated Data (one-time setup)

The gating mechanism must be run first to create the routing CSVs:

```bash
cd /gpfs/home5/scur0509/projects/MoE-ShapeRefine-MedicalSeg/src

# Full run (all fragments)
python gating_mechanism/gating_mechanism.py

# Or smoke test (5 cases per split)
python gating_mechanism/gating_mechanism.py --smoke
```

This creates:
- `gating_mechanism/gated_train_records.csv`
- `gating_mechanism/gated_val_records.csv`
- `gating_mechanism/gated_test_records.csv`

### 2. Ensure MedSAM predictions exist

Required data must be in place:
- `data/medsam-predictions/embeddings/*.npy` — ViT-B image embeddings
- `data/medsam-predictions/binary_masks/*.npz` — MedSAM coarse predictions
- Ground truth masks from PENGWIN dataset

---

## Training

### Option: Run locally

```bash
cd /gpfs/home5/scur0509/projects/MoE-ShapeRefine-MedicalSeg/src/FlowSDF

# Train both experts
python train_flowsdf_moe.py --expert both --epochs 100 --batch-size 8

# Train only expert_small
python train_flowsdf_moe.py --expert expert_small --epochs 50 --batch-size 16

# Train with subsampled expert_large (balance dataset sizes)
python train_flowsdf_moe.py \
    --expert both \
    --epochs 100 \
    --batch-size 8 \
    --large-subsample 17000
```

---

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--expert` | `both` | Which expert(s) to train: `expert_small`, `expert_large`, or `both` |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `8` | Batch size per expert |
| `--num-workers` | `4` | DataLoader workers (ignored in FlowSDF, set by trainer) |
| `--lr` | `1e-4` | Learning rate (Adam optimizer) |
| `--ema-decay` | `0.999` | EMA decay for model checkpoint averaging |
| `--sigma-min` | `1e-5` | Minimum sigma for diffusion process |
| `--img-size` | `128` | Training resolution for embedding, coarse mask, and target SDF |
| `--sdf-threshold` | `15.0` | Distance clipping threshold before SDF normalization |
| `--clip-grad` | `1.0` | Infinity-norm gradient clipping value; use a negative value to disable |
| `--img-cond-channels` | `257` | RRDB conditioner input channels: 256 embedding + 1 coarse mask |
| `--large-subsample` | `None` | Subsample expert_large to N fragments (preserves SA/LI/RI class balance) |
| `--checkpoint-dir` | `../../checkpoints/flowsdf` from `src/FlowSDF` | Output directory for model checkpoints |

---

## Output

Training saves:
```
checkpoints/flowsdf/
├── expert_small_best.pth       # Best validation loss checkpoint
├── expert_small_final.pth      # Final epoch checkpoint
├── expert_small_loss_curve.png # Training/val loss curve
├── expert_large_best.pth
├── expert_large_final.pth
└── expert_large_loss_curve.png
```

Each checkpoint contains:
```python
{
    "epoch": int,
    "model_state": dict,       # model.state_dict()
    "ema_state": dict,         # ema_model.state_dict()
    "val_loss": float,         # validation loss (for best only)
    "img_cond_channels": int,  # default 257
    "sdf_threshold": float,    # default 15.0
}
```

---

After training, use the checkpoints to:

1. **Refine MedSAM predictions** during inference
2. **Evaluate** against ground truth masks
3. **Combine** expert outputs in the full MoE pipeline

## Notes

- **Data format:** Training data is loaded lazily from MoE gated CSV records. MedSAM embeddings and masks are read from disk per fragment, then adapted into FlowSDF tensors.
- **Memory:** Peak GPU usage ~5-8 GB (fits within A100 10GB MIG slice).
- **Subsampling:** `expert_large` is ~10× larger than `expert_small`. Use `--large-subsample` to balance during training if desired.
- **Checkpoint usage:** Load with `torch.load(ckpt, map_location=device)["model_state"]` then call `model.load_state_dict(...)`.
