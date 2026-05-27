# medSAM-stage1

Files to add to a fork of [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) in order to reproduce the Stage 1 coarse predictions used by this pipeline.

## Setup

1. Fork and clone the upstream MedSAM repository:

```bash
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM
```

2. Copy everything here into the root of that clone:

```bash
cp -r medSAM-stage1/* /path/to/MedSAM/
```

This adds the PENGWIN-specific scripts and replaces the two training scripts with patched versions that accept 3D instance-mask ground truth.

3. Download the MedSAM ViT-B checkpoint into `work_dir/MedSAM/medsam_vit_b.pth` (see the upstream README for the download link).

4. Provision the conda environment on Snellius:

```bash
sbatch medsam_env.job
conda activate medsam
```

---

## Files

### `medsam_env.job`

Slurm job that creates or updates the `medsam` conda environment (Python 3.10, PyTorch 2.0, CUDA 11.8) on Snellius. Run once before training or inference.

---

### `scripts/prepare_pengwin_xray_for_medsam.py`

Converts PENGWIN Task 2 X-ray images and their 30-bit encoded segmentation labels into MedSAM's NumPy training format.

```bash
# Full run
python scripts/prepare_pengwin_xray_for_medsam.py \
  --output-root /path/to/derived/medsam/xray

# Smoke test (5 samples)
python scripts/prepare_pengwin_xray_for_medsam.py \
  --output-root /tmp/smoke \
  --limit 5
```

Output layout:

```text
<output_root>/
  imgs/           # float32 (1024, 1024, 3), neg-log normalized, range [0, 1]
  gts/            # uint8 (N, 1024, 1024), one channel per fragment instance
  metadata.jsonl  # per-sample records with fragment details and paths
```

Options:
- `--limit N` — process only N cases
- `--keep-empty` — include images with no labelled fragments (skipped by default)

---

### `scripts/pengwin_prep_smoke.job`

Slurm job that runs `prepare_pengwin_xray_for_medsam.py` with `--limit 5` as a quick validation before a full run.

```bash
sbatch scripts/pengwin_prep_smoke.job
```

---

### `scripts/view_pengwin.py`

Interactive matplotlib viewer for exploring the PENGWIN dataset.

```bash
# Browse X-ray cases
python scripts/view_pengwin.py --task xray

# Jump to a specific case
python scripts/view_pengwin.py --task xray --case 001_0000

# CT with custom overlay opacity
python scripts/view_pengwin.py --task ct --alpha 0.5
```

Keyboard: `←` / `→` navigate cases, `↑` / `↓` navigate CT slices, `o` toggles overlay.

---

### `train_one_gpu.py` and `train_multi_gpus.py`

These replace the upstream training scripts. The only change is in `NpyDataset.__getitem__`: the original code expects 2D single-label ground truth masks; these versions also accept the 3D instance-mask stacks produced by `prepare_pengwin_xray_for_medsam.py`.

- If `gt.ndim == 3`: one instance channel is randomly sampled per training step.
- If `gt.ndim == 2`: original upstream label-sampling logic is used unchanged.

```bash
# Single GPU
python train_one_gpu.py \
  --tr_npy_path /path/to/derived/medsam/xray \
  --medsam_checkpoint work_dir/MedSAM/medsam_vit_b.pth \
  --max_epoch 10 \
  --batch_size 4 \
  --task_name PENGWIN_xray

# Multi-GPU
bash train_multi_gpus.sh
```

---

### `trial_run.py`

Standalone inference test on a single image and bounding box.

```bash
srun python trial_run.py \
  -i assets/img_demo.png \
  -o results \
  --box 95,200,190,200 \
  --checkpoint work_dir/MedSAM/medsam_vit_b.pth
```

Saves `results/mask.npy` and `results/overlay.png`.
