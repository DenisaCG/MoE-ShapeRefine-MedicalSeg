import json, numpy as np

with open("data/medsam-predictions/metadata.jsonl") as f:
    rec = json.loads(next(f))

masks = np.load(rec["binary_masks_path"])["masks"]
emb = np.load(rec["embedding_path"])

print(rec["sample_name"])
print(masks.shape, masks.dtype)
print(emb.shape, emb.dtype)
print(len(rec["fragments"]))