# Gating Mechanism & Dataset

## Overview

Two-step pipeline that routes MedSAM fragment predictions to expert models and builds PyTorch DataLoaders for training.

---

## Files

| File | Purpose |
|------|---------|
| `gating_mechanism.py` | Run once — gates all fragments and saves CSVs |
| `dataset.py` | Import every training run — builds DataLoaders from CSVs |

---

## Step 1 - Run Gating (once)

Reads all MedSAM binary masks, counts foreground pixels per fragment (`binary_mask.sum()`), and routes each fragment to `expert_small` or `expert_large` based on area. Saves one CSV per split.

```bash
# smoke test (5 cases per split)
python gating_mechanism.py --smoke

# full run (all cases)
python gating_mechanism.py
```

Or submit to Snellius:
```bash
sbatch run_gating_smoke.job
```

**Output:**
```
gating_mechanism/
├── gated_train_records.csv
├── gated_val_records.csv
└── gated_test_records.csv
```

Each CSV has one row per fragment:

| Column | Description |
|--------|-------------|
| `case_id` | Image identifier e.g. `001_0000` |
| `sample_name` | Full name e.g. `XRAY_PENGWIN_001_0000` |
| `medsam_instance_id` | Index into the `.npz` masks array (1-based) |
| `category_id` | 1=SA, 2=LI, 3=RI |
| `category_name` | SA / LI / RI |
| `fragment_id` | Fragment number within that category |
| `area` | Foreground pixel count on 1024×1024 grid |
| `expert` | `expert_small` or `expert_large` |
| `embedding_path` | Path to `.npy` image embedding |
| `binary_masks_path` | Path to `.npz` MedSAM masks |

**Gating threshold:** `area <= 10485` pixels (1% of 1024×1024) → `expert_small`, else → `expert_large`.

---

## Step 2 - Build DataLoaders (every training run)

Reads the CSVs and returns one DataLoader per expert. No gating is re-run — just a fast CSV read.

```python
from dataset import build_expert_dataloaders

# standard — use all fragments
loaders = build_expert_dataloaders(split="train", batch_size=8, num_workers=4)

# access each expert's loader
for batch in loaders["expert_small"]:
    ...
for batch in loaders["expert_large"]:
    ...
```

---

## Subsampling expert_large

`expert_large` has ~194k fragments vs ~17k for `expert_small`. Use `large_subsample` to balance them while preserving SA/LI/RI class proportions:

```python
import pandas as pd
from dataset import build_expert_dataloaders

# match expert_large size to expert_small
df = pd.read_csv("gated_train_records.csv")
n_small = len(df[df.expert == "expert_small"])

train_loaders = build_expert_dataloaders(split="train", batch_size=8, num_workers=4, large_subsample=n_small)

# val and test can use different subsample sizes independently
val_loaders  = build_expert_dataloaders(split="val",  batch_size=8, num_workers=4, large_subsample=2000)
test_loaders = build_expert_dataloaders(split="test", batch_size=8, num_workers=4, large_subsample=1000)
```

- Each split is independent - train, val, test can all have different `large_subsample` values
- If `large_subsample` exceeds the available fragments in that split, all fragments are used
- `large_subsample=None` (default) uses all fragments with no subsampling
- Only `expert_large` is subsampled - `expert_small` always uses all its fragments

---

## What Each Batch Contains

| Key | Shape | Description |
|-----|-------|-------------|
| `embedding` | `(B, 256, 64, 64)` | SAM ViT-B image encoder output, shared across all fragments of the same image |
| `binary_mask` | `(B, 1, 1024, 1024)` | MedSAM coarse prediction for this fragment |
| `gt_mask` | `(B, 1, 448, 448)` | Ground truth binary mask decoded from PENGWIN bit-packed `.tif` |

Data is loaded lazily, nothing is read from disk until a batch is requested.

---


## Fragment-to-Mask Alignment

Each row in the CSV corresponds to one fragment. The `medsam_instance_id` maps directly to the mask array:

```
masks[medsam_instance_id - 1]  →  binary mask for this fragment
```

Alignment is verified, `masks[i]` correctly corresponds to `fragments[i]` in the metadata (IoU 0.5–0.8 across test cases, lower than 1.0 because MedSAM predictions are intentionally coarse).
