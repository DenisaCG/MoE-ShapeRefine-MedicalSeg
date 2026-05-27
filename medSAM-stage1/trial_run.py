import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry

# Note: Scripts were adapted from the tutorial_quickstart.ipynb notebook.

# Utils (from MedSAM notebook logic)
def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                      edgecolor="red", facecolor=(0,0,0,0), lw=2)
    )
# MedSAM inference
@torch.no_grad()
def run_medsam(model, image, box, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))

    img_tensor = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(device)

    box_np = np.array(box, dtype=np.float32)
    box_np = box_np[None, :]

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=torch.tensor(box_np).to(device),
        masks=None,
    )

    image_embedding = model.image_encoder(img_tensor)

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    masks = torch.sigmoid(low_res_masks)
    masks = torch.nn.functional.interpolate(
        masks, size=(1024, 1024), mode="bilinear", align_corners=False
    )

    return masks[0, 0].cpu().numpy()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()

    # Load image
    image = cv2.imread(args.image)

    # Box format: x1,y1,x2,y2
    box = list(map(int, args.box.split(",")))

    mask = run_medsam(sam, image, box, device)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mask.npy")
    np.save(out_path, mask)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    show_mask(mask > 0.5, ax)
    show_box(box, ax)
    ax.axis("off")

    plt.savefig(os.path.join(args.out_dir, "overlay.png"), bbox_inches="tight")
    print(f"Saved results to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="input image path")
    parser.add_argument("-o", "--out_dir", required=True, help="output directory")
    parser.add_argument("--box", required=True, help="bbox: x1,y1,x2,y2")
    parser.add_argument("--checkpoint", required=True, help="MedSAM checkpoint path")

    args = parser.parse_args()
    main(args)
