# MoE Refinement Pipeline — Work Plan

## Overview

Two-stage pipeline for pelvic bone fragment segmentation on PENGWIN Task 2 (2D X-ray):

1. **Stage 1 (done):** MedSAM ViT-B produces per-fragment binary masks and a shared image embedding per X-ray.
2. **Stage 2 (this plan):** A Mixture of Experts (MoE) refines those first-stage masks using the MedSAM embedding as input, without ever seeing raw pixels.

The MoE has two experts with identical architecture but separate weights. Routing is hard: a gating dataloader partitions fragments by bounding-box area into *small* and *large* populations, and each fragment is sent to exactly one expert.

---

## What Already Exists

| Artifact | Location | Status |
|---|---|---|
| MedSAM inference + embedding extraction | `src/run_medsam_with_pengwin_boxes.py` | Done |
| Bounding box preparation | `src/prepare_pengwin_xray_boxes_for_medsam_inference.py` | Done |
| First-stage binary masks | `data/medsam-predictions/binary_masks/` | Done (50k .npz) |
| ViT-B image embeddings | `data/medsam-predictions/embeddings/` | Done (50k .npy, float16, 256×64×64) |
| Per-image + per-fragment metadata | `data/medsam-predictions/metadata.jsonl` | Done |
| First-stage evaluation script | `evaluation/evaluate_medsam_pengwin.py` | Done |
| MoE model file | `src/cnnMoE.py` | Empty placeholder |
| Gating dataloader (area-based split) | — | **Pending (teammate)** |

---

## What Needs to Be Built

| Component | Owner | File(s) |
|---|---|---|
| T1 — SDF preprocessing (per-fragment) | You | `src/sdf_utils.py` |
| T2 — Gating dataloader (area split) | Teammate | `src/moe_dataset.py` |
| T3 — CNN expert model | You | `src/cnnMoE.py` |
| T4 — Boundary-focused loss | You | `src/losses.py` |
| T5 — Training script | You | `src/train_moe.py` |
| T6 — Inference / refinement script | You | `src/infer_moe.py` |
| T7 — Second-stage evaluation | You | `evaluation/evaluate_moe_pengwin.py` |

---

## Task Breakdown

---

### T1 — SDF Preprocessing

**Purpose:** Convert each MedSAM binary mask (coarse) into a signed distance transform (SDF) channel. This gives the expert boundary-proximity information: negative values inside the mask, positive outside, zero at the boundary. Normalised to [−1, +1].

**Formula:**
```
sdf = dist_transform(~mask) − dist_transform(mask)
sdf_norm = sdf / max(|sdf|)   # per-instance normalisation
```

**Inputs:**
- MedSAM binary mask, cropped to RoI and resized to fixed spatial size `(H, W)` — shape `(1, H, W)`, uint8

**Outputs:**
- SDF channel — shape `(1, H, W)`, float32, range [−1, +1]

**Where it runs:** Inside the PyTorch `Dataset.__getitem__` as a transform — not a standalone precomputation script, because RoI cropping happens per-fragment at load time and it is fast (scipy on a small patch).

**Dependencies:** None. Can be implemented before the dataloader.

**Open question:** Replace the binary mask channel entirely with the SDF (→ 257 channels input) or keep both (→ 258 channels)? Keeping both is safer to start; ablate later.

---

### T2 — Gating Dataloader (Teammate)

**Purpose:** Build a PyTorch `Dataset` that yields one training sample per fragment, and exposes a pre-computed `is_large` flag (or separate `DataLoader` instances) so the training loop can route each sample to the correct expert.

**Inputs per sample:**
- `embedding_path` (.npy, float16 256×64×64) — image-level, shared across fragments of the same X-ray
- `binary_masks_path` (.npz, uint8 N×1024×1024) — index into mask array with `medsam_instance_id`
- `bbox_xyxy_1024` — bounding box on the 1024-grid, from `metadata.jsonl`
- `original_label_path` — GT label .tif, for SDF of GT (used in loss)
- `category_id`, `fragment_id` — for GT decoding (same logic as `evaluate_medsam_pengwin.py`)

**Outputs per sample (dict):**
```
{
  "embedding_crop": Tensor (256, H, W),   # RoI-aligned from full embedding
  "coarse_mask_crop": Tensor (1, H, W),   # initial MedSAM mask, resized to RoI
  "gt_mask_crop": Tensor (1, H, W),       # GT mask, cropped to same RoI
  "bbox": Tensor (4,),
  "area": int,                            # fragment area in pixels on 1024-grid
  "is_large": bool,                       # gating flag
  "sample_name": str,
  "fragment_index": int,
}
```

**RoI-align note:** Use `torchvision.ops.roi_align` to crop the embedding. Scale bbox from 1024-grid → 64-grid by multiplying by `64/1024`. Choose a fixed output size (e.g. `16×16`). See README for working example.

**Area threshold:** To be decided by the group — see Open Questions.

**Dependencies:** T1 (SDF transform will be applied inside or after this dataset).

---

### T3 — CNN Expert Model

**Purpose:** A lightweight spatial decoder that takes RoI-aligned embedding + coarse mask (+ SDF channel if used) and outputs a refined binary mask in RoI space.

**Architecture (per expert):**
```
Input: (C_in, H, W)  where C_in = 257 (embed + binary mask) or 258 (embed + mask + SDF)

Conv(C_in → 128, 3×3, pad=1) + BN + ReLU
Conv(128   →  64, 3×3, pad=1) + BN + ReLU
Conv( 64   →   1, 1×1)
Sigmoid

Output: (1, H, W)  — refined mask in RoI space
```

Two instances of this class (one per expert) with independent weights, trained separately on their respective fragment populations.

**Design rationale:** Convolutions here act as a *spatial decoder*, not a feature extractor — MedSAM already extracted the features. No need for a backbone (ResNet etc.), which would be redundant and overparameterised (~11M params vs ~200–500K here).

**Inputs:**
- `embedding_crop` (256, H, W)
- `coarse_mask_crop` (1, H, W)
- SDF channel (1, H, W) if included

**Outputs:**
- Refined mask (1, H, W), float32, range [0, 1] (sigmoid)

**Dependencies:** T2 (to know `C_in` and `H, W`).

---

### T4 — Boundary-Focused Loss

**Purpose:** Steer the expert to focus capacity on shape refinement at fragment edges, where MedSAM is known to be imprecise.

**Components:**

1. **Standard Dice loss** — global foreground overlap.
2. **Boundary-weighted BCE** — standard pixel-wise BCE scaled by a per-pixel weight map `W`:
   - Compute GT boundary: `boundary = dilate(gt) XOR erode(gt)` (morphological difference)
   - Create weight map: `W[px] = w_boundary` if pixel within ~3px of boundary, else `1.0`
   - Apply: `loss = mean(W * BCE_per_pixel(pred, gt))`
3. **Optional boundary-band Dice** — Dice computed only on a dilated band around the GT boundary.

**Implementation note:** All morphological ops run on CPU tensors via `scipy.ndimage` or a small custom PyTorch impl. Precompute `W` in the dataloader (or lazily in loss) from the GT mask crop.

**Inputs:**
- `pred`: (B, 1, H, W) refined mask logits or probabilities
- `gt`: (B, 1, H, W) ground-truth binary mask
- `w_boundary`: float, upweighting factor for boundary pixels (e.g. 3–5×)
- `boundary_radius`: int, dilation radius in pixels (e.g. 3)

**Outputs:** Scalar loss tensor.

**Dependencies:** T2 (GT mask crop).

---

### T5 — Training Script

**Purpose:** Train the two experts end-to-end (separately, one at a time) on their respective fragment populations.

**Logic:**
```
for expert_id in ["small", "large"]:
    dataset = MoEDataset(split=expert_id, ...)
    model = CNNExpert(C_in=...).cuda()
    optimizer = AdamW(...)
    for epoch in range(N_EPOCHS):
        for batch in DataLoader(dataset, ...):
            embedding_crop = batch["embedding_crop"].cuda()
            coarse_mask    = batch["coarse_mask_crop"].cuda()
            sdf_channel    = compute_sdf(coarse_mask)          # T1
            x = torch.cat([embedding_crop, coarse_mask, sdf_channel], dim=1)
            pred = model(x)
            loss = boundary_dice_bce_loss(pred, batch["gt_mask_crop"].cuda())  # T4
            loss.backward(); optimizer.step()
```

**Inputs:** Outputs of T2 (dataset), T3 (model), T4 (loss).

**Outputs:** Two checkpoint files: `checkpoints/expert_small.pth`, `checkpoints/expert_large.pth`.

**Dependencies:** T1, T2, T3, T4 all complete.

---

### T6 — Inference Script

**Purpose:** Run both experts on a held-out split, reconstruct full 1024×1024 refined masks per image, and write output .npz files in the same format as the first-stage masks.

**Logic:**
- Load both expert checkpoints.
- For each sample: gate each fragment to the correct expert via its area.
- Run expert forward pass in RoI space.
- Paste refined mask back into the full 1024×1024 canvas at the fragment's bounding box.
- Save as `data/moe-predictions/binary_masks/XRAY_PENGWIN_*.npz`.

**Dependencies:** T2 (gating logic), T3 (model), T5 (trained checkpoints).

---

### T7 — Second-Stage Evaluation

**Purpose:** Compare MoE refined masks against GT using the same metrics as `evaluate_medsam_pengwin.py` (Dice, IoU, HD95, ASSD). Also compare directly against first-stage MedSAM to quantify the improvement.

**Changes from existing evaluator:**
- Point `--pred-mask-root` at `data/moe-predictions/binary_masks/` instead of `data/medsam-predictions/binary_masks/`.
- Optionally extend to output a delta CSV (MoE − MedSAM) per fragment.

**Dependencies:** T6 (inference output).

---

## Implementation Order

```
T1 (SDF utils)           ──┐
                           ├──► T3 (CNN model) ──┐
T2 (dataloader, teammate) ─┘                     ├──► T5 (training) ──► T6 (inference) ──► T7 (eval)
                           └──► T4 (loss)   ──────┘
```

**Suggested sequence:**

1. **T1** — SDF utils. No dependencies, standalone, fast to test with a single fragment.
2. **T3** — CNN model definition. Only needs knowledge of `C_in` and `H, W`; can stub those as constants while T2 is in progress.
3. **T4** — Loss functions. Only needs GT mask tensors; can be unit-tested independently.
4. **T2** — Gating dataloader (teammate, running in parallel with T1/T3/T4).
5. **T5** — Training script. Requires T1–T4 all complete.
6. **T6** — Inference script. Requires T5 (trained weights).
7. **T7** — Evaluation. Requires T6.

---

## Open Questions

| # | Question | Who decides | Impact |
|---|---|---|---|
| OQ1 | **Area threshold for small/large split.** What pixel-area cutoff on the 1024-grid defines "small" vs "large"? Could be median of training set, or a task-specific value. | Group | Gating logic in T2 |
| OQ2 | **RoI output size.** `7×7` (SAM default) vs `16×16` vs `32×32`. Larger → more spatial detail but more parameters. | You + teammate | C_in computation, model size |
| OQ3 | **SDF: replace binary mask or concatenate?** Starting with concatenation (258 channels) is safer; can ablate. | You | C_in in T3 |
| OQ4 | **Boundary weight value.** How much to upweight boundary pixels in BCE (e.g. 3×, 5×)? Tune on validation. | You | T4 |
| OQ5 | **Train experts jointly or separately?** Separately (one DataLoader per expert) is simpler. Joint training with a shared batch would require dynamic routing. | Group | T5 architecture |
| OQ6 | **Evaluation split.** Is a train/val/test split defined for PENGWIN Task 2? Need to confirm to avoid evaluating on training data. | Group | T7 validity |
| OQ7 | **Output resolution.** Is RoI-space output (pasted back) sufficient, or do we need explicit upsampling to 448×448 (original GT resolution)? The existing evaluator resizes predictions to GT space so pasting back at 1024-grid should be fine. | You | T6 |
