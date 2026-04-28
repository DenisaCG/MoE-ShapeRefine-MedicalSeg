# MoE-ShapeRefine-MedicalSeg
Mixture of Experts for Fine-Grained Shape Refinement in Medical Segmentation

### MedSAM Environment
```bash
# Create environment - only needed the first time.
conda env create -f environment.yml

# Activate environment
conda activate MoE
```

#### Fixing format of model weights
```bash
# Fix checkpoint structure - only needed for the first time.
srun python fix_checkpoint.py
```

### PENGWIN Data
The Pengwin scripts in `src/` use the shared workspace data directory:

```text
/home/scur0509/projects/data/pengwin
```

To download and extract the official Pengwin data:

```bash
python src/download_pengwin.py --task xray
```

This creates the canonical dataset layout under:

```text
data/pengwin/
  raw/
  original/
```

### PENGWIN Boxes For First-Stage MedSAM
For the first-stage segmentation pass we do not prepare SFT masks. Instead, we
prepare normalized X-ray images together with per-fragment bounding boxes that
can be used directly as MedSAM box prompts during inference.

Run:

```bash
python src/prepare_pengwin_xray_boxes_for_medsam_inference.py
```

By default this writes to:

```text
data/bounding-boxes-xrays/
```

Output layout:

```text
<output_root>/
  imgs/           (empty — populated by downstream MedSAM inference)
  boxes/
  metadata.jsonl
```

Each sample contains:
- `boxes/*.npy`: `float32` array of shape `(N, 4)` in `xyxy` format on the resized `1024x1024` grid
- `metadata.jsonl`: fragment metadata, `box_path` pointer, and bounding boxes in both original-pixel and `1024`-grid coordinates; image paths point back to the original `.tif` files

This is intended for the MoE pipeline where MedSAM produces the first-stage
segmentation prediction from an image and a bounding box, and that prediction is
then passed to the second-stage refinement model.

Useful options:
- `--bbox-pad 5` to add a small padding margin around each box
- `--limit 10` for a smoke test
- `--keep-empty` to retain images with no positive fragments
- `--resume` to continue a previous interrupted run and skip samples that already have outputs and metadata
- `--overwrite` to force regeneration of already processed samples


### Running MedSAM Inference and Feature Extraction

This is the first stage of the MoE pipeline. For each image the script:
1. Loads and preprocesses the X-ray on the fly (neglog normalisation → uint8 → 1024×1024)
2. Runs the MedSAM ViT-B image encoder once → saves the image embedding
3. Runs the mask decoder once per fragment (batched) → saves binary masks

Submit on Snellius:

```bash
sbatch run_inference_extract_features.job
```

Or run directly:

```bash
python src/run_medsam_with_pengwin_boxes.py
```

By default reads from `data/bounding-boxes-xrays/` and writes to `data/medsam-predictions/`.

Output layout:

```text
data/medsam-predictions/
  binary_masks/
    XRAY_PENGWIN_001_0000.npz   # uint8 (N, 1024, 1024), key "masks"
    ...
  embeddings/
    XRAY_PENGWIN_001_0000.npy   # float16 (256, 64, 64)
    ...
  metadata.jsonl                # paths + fragment metadata for every prediction
```

Load outputs:

```python
import numpy as np

masks     = np.load("data/medsam-predictions/binary_masks/XRAY_PENGWIN_001_0000.npz")["masks"]
# shape: (N, 1024, 1024), uint8 — one mask per fragment, same order as metadata["fragments"]

embedding = np.load("data/medsam-predictions/embeddings/XRAY_PENGWIN_001_0000.npy").astype(np.float32)
# shape: (256, 64, 64) — ViT-B image embedding, shared across all N fragments of this image
```

Useful options:
- `--resume` to continue an interrupted run (safe to resubmit)
- `--overwrite` to reprocess everything from scratch
- `--limit 10` for a smoke test
- `--threshold 0.5` to change the binary mask cutoff
- `--case-id 001_0000` or `--sample-name XRAY_PENGWIN_001_0000` to run one sample

### Using MedSAM Outputs as MoE Inputs

The outputs above are designed to feed directly into the second-stage MoE
refinement model. Each training sample for the MoE corresponds to one
fragment and consists of:

| Input | Source | Shape |
|---|---|---|
| Initial mask | `binary_masks/*.npz`, slice `[i]` | `(1, 1024, 1024)` uint8 |
| Image embedding | `embeddings/*.npy` | `(256, 64, 64)` float32 |
| Bounding box | `metadata.jsonl → fragments[i]["bbox_xyxy_1024"]` | `(4,)` float32 |
| Category | `metadata.jsonl → fragments[i]["category_name"]` | SA / LI / RI |

The embedding is image-level (shared across all fragments of an image). To
obtain per-fragment features, use RoI-align to crop the embedding at the
fragment's bounding box:

```python
import torchvision.ops as ops

# embedding: (1, 256, 64, 64) tensor, boxes in 1024-grid coords → scale to 64-grid
scale = 64 / 1024
box_64 = [x * scale for x in bbox_xyxy_1024]  # [x1, y1, x2, y2] on the 64×64 grid
roi = ops.roi_align(embedding.unsqueeze(0), [torch.tensor([box_64])], output_size=(7, 7))
# roi: (1, 256, 7, 7) — fragment-specific feature crop
```

All outputs are matched by `sample_name` (e.g. `XRAY_PENGWIN_001_0000`) and
fragment index `i`, making it straightforward to build a PyTorch `Dataset`
that loads (mask, embedding, box, category) tuples.

### Testing MedSAM on one input
```bash
srun python trial_run.py \
  -i assets/img_demo.png \
  -o results \
  --box 95,200,190,200 \
  --checkpoint work_dir/MedSAME/medsam_vit_b.pth
```
