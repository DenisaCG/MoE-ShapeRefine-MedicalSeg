# Gating Mechanism & Dataset

## Overview

Two-step pipeline that routes MedSAM fragment predictions to expert models and builds PyTorch DataLoaders for training.

**Gating "area" = foreground pixel count.** For each fragment, `gating_mechanism.py` loads the MedSAM binary mask and computes `area = binary_mask.sum()` — the number of foreground (`1`) pixels on the 1024×1024 grid. This is **not** the bounding-box area (width × height); it's the actual pixel count of the predicted mask. Each fragment is then routed to `expert_small` or `expert_large` based on that pixel count.

The routing mechanism supports two modes (`--routing area` / `--routing random`) as part of a gating ablation study — see [Gating Ablation Study](#gating-ablation-study) below.

---

## Files

| File | Purpose |
|------|---------|
| `gating_mechanism.py` | Run once per routing configuration — gates all fragments and saves CSVs |
| `dataset.py` | Import every training run — builds DataLoaders from CSVs |

---

## Step 1 - Run Gating

Reads all MedSAM binary masks, counts foreground pixels per fragment (`binary_mask.sum()`), and routes each fragment to `expert_small` or `expert_large`. Saves one CSV per split.

```bash
# smoke test (5 cases per split)
python gating_mechanism.py --smoke

# full run (all cases), canonical routing — identical to running with no args at all
python gating_mechanism.py
```

Or submit to Snellius:
```bash
sbatch run_gating_smoke.job
```

**Output (default invocation, no args):**
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
| `area` | Foreground pixel count on 1024×1024 grid (`binary_mask.sum()`) |
| `expert` | `expert_small` or `expert_large` |
| `embedding_path` | Path to `.npy` image embedding |
| `binary_masks_path` | Path to `.npz` MedSAM masks |

**Canonical gating threshold:** `area <= 5,402` foreground pixels → `expert_small`, else → `expert_large`. (This is `DEFAULT_THRESHOLD` in `gating_mechanism.py`; ≈0.52% of the 1024×1024 grid.)


When a non-canonical `--routing`/`--threshold` is used, two additional provenance columns are appended so ablation CSVs are self-describing without changing the canonical schema:

| Column | Description |
|--------|-------------|
| `routing_mode` | `"area"` or `"random"` |
| `routing_seed` | Seed used, if `routing_mode == "random"`; otherwise empty |
| `routing_threshold` | Threshold used, if `routing_mode == "area"`; otherwise empty |

These columns are **only** added when `--threshold`/`--routing` deviate from the canonical default (`area`, `5402`) — the default invocation's CSV schema and values are unchanged from before.

### CLI options

| Argument | Default | Description |
|---|---|---|
| `--smoke` | off | Run on 5 cases per split only (fast sanity check) |
| `--threshold` | `5402` | Foreground-pixel-area threshold for `--routing area`. Ignored when `--routing random` is used. |
| `--routing` | `area` | `area`: route by foreground pixel area vs `--threshold`. `random`: randomly assign fragments to `expert_small`/`expert_large`, preserving the same `expert_small` ratio the canonical `area`-`5402` routing would produce for each split (computed independently per split — train, val, and test each get their own random assignment, not a single split-agnostic draw). |
| `--seed` | `42` | Random seed for `--routing random` (ignored for `--routing area`). Uses `numpy.random.default_rng(seed)` for reproducibility. |
| `--out-dir` | this file's directory | Where to write `gated_{split}_records.csv`. Defaults to the canonical location. Pass a different directory for ablation runs so the canonical CSVs used by existing checkpoints/training runs are never overwritten. |

Running `python gating_mechanism.py` with **no arguments** reproduces the original, pre-ablation CSVs exactly — same columns, same values, same destination.

### Routing / ratio summary output

After generating each split's CSV, the script prints a concise per-split summary (`expert_small`/`expert_large` count and percentage), and a combined table across train/val/test at the end:

```
[train] total=211926  expert_small=17335 (8.18%)  expert_large=194591 (91.82%)
[val] total=26544  expert_small=2258 (8.51%)  expert_large=24286 (91.49%)
[test] total=26457  expert_small=2136 (8.07%)  expert_large=24321 (91.93%)

Routing split summary (all splits):
split      total   small_n   small_%   large_n   large_%
---------------------------------------------------------
train     211926     17335     8.18%    194591    91.82%
val        26544      2258     8.51%     24286    91.49%
test       26457      2136     8.07%     24321    91.93%
```

This is especially useful for `--routing random`: compare its `small_%`/`large_%` per split against the canonical `area`-`5402` run's table to confirm the random assignment preserved the same ratio.

---

## Gating Ablation Study

Three routing strategies are compared, all evaluated on the exact same train/val/test case splits (`data/pengwin/splits/*.csv`, unaffected by routing):

| Strategy | Routing | Threshold / Seed |
|---|---|---|
| Current routing (canonical) | `area` | threshold = 5,402 |
| Alternative threshold | `area` | threshold = 14,188 |
| Random routing baseline | `random` | seed = 42, ratio-matched to the 5,402 threshold |

Each strategy writes to its own `--out-dir` so the canonical CSVs are never touched, and downstream training/inference scripts (`src/cnnROI/train_roi.py`, `src/cnnROI/infer_roi.py`) read the desired strategy's CSVs via `--gating-csv-dir`.

**Generate the three gating CSV variants:**

```bash
# Strategy 1 — canonical (5402), written to a parallel dir for symmetry with the others
python gating_mechanism.py \
    --threshold 5402 \
    --out-dir gating_area_5402

# Strategy 2 — alternate threshold
python gating_mechanism.py \
    --threshold 14188 \
    --out-dir gating_area_14188

# Strategy 3 — random routing, ratio-matched to the 5402 threshold, seeded
python gating_mechanism.py \
    --routing random \
    --seed 42 \
    --out-dir gating_random_seed42
```

Each command prints the per-split and combined `expert_small`/`expert_large` ratio tables described above — use these to sanity-check that:
- Strategy 2 (`area_14188`) shifts more fragments into `expert_small` than the canonical 5,402 threshold (since the threshold is higher).
- Strategy 3 (`random_seed42`) reproduces the same per-split ratios as Strategy 1 (canonical 5,402), just with a randomized `expert_small`/`expert_large` assignment instead of an area-based one.

See `src/cnnROI/train_roi.py --gating-csv-dir` and `src/cnnROI/infer_roi.py --gating-csv-dir` for pointing training/inference at a specific strategy's CSVs.

---

## Step 2 - Build DataLoaders (every training run)

Reads the CSVs and returns one DataLoader per expert. No gating is re-run — just a fast CSV read.

```python
from dataset import build_expert_dataloaders

# standard — use all fragments, canonical gating CSVs
loaders = build_expert_dataloaders(split="train", batch_size=8, num_workers=4)

# point at an alternate gating run (e.g. one of the ablation strategies above)
loaders = build_expert_dataloaders(
    split="train", batch_size=8, num_workers=4,
    csv_dir="gating_area_14188",
)

# access each expert's loader
for batch in loaders["expert_small"]:
    ...
for batch in loaders["expert_large"]:
    ...
```

`csv_dir` defaults to `None`, which resolves to `CSV_DIR` (this file's directory, i.e. the canonical gating output) — omitting it preserves the original behavior exactly.

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
