import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MEDSAM_METADATA = Path("/gpfs/home5/scur0509/projects/MoE-ShapeRefine-MedicalSeg/data/medsam-predictions/metadata.jsonl")
SPLITS_DIR      = Path("/gpfs/home5/scur0509/projects/data/pengwin/splits")
CSV_DIR         = Path(__file__).parent
IMAGE_SIZE      = 1024
TOTAL_PX        = IMAGE_SIZE * IMAGE_SIZE


def load_metadata_index(metadata_path: Path) -> dict:
    index = {}
    with metadata_path.open() as f:
        for line in f:
            record = json.loads(line.strip())
            index[record["case_id"]] = record
    return index


def route_to_expert(area, threshold=5402):
    return "expert_small" if area <= threshold else "expert_large"


def gate_case(case_id: str, metadata_index: dict, threshold=5402) -> list[dict]:
    """
    Load MedSAM binary masks for case_id and route each fragment to an expert.

    Masks shape : (N, 1024, 1024) binary.
    Fragment i  : masks[i] corresponds to metadata fragments[i] (medsam_instance_id = i+1).
    """
    if case_id not in metadata_index:
        raise KeyError(f"case_id '{case_id}' not in metadata")

    record = metadata_index[case_id]
    masks  = np.load(record["binary_masks_path"])["masks"]

    records = []
    for i, frag in enumerate(record["fragments"]):
        binary_mask = masks[i]
        area        = int(binary_mask.sum())

        records.append({
            "case_id":            case_id,
            "sample_name":        record["sample_name"],
            "medsam_instance_id": frag["medsam_instance_id"],
            "category_id":        frag["category_id"],
            "category_name":      frag["category_name"],
            "fragment_id":        frag["fragment_id"],
            "area":               area,
            "expert":             route_to_expert(area, threshold),
            "embedding_path":     record["embedding_path"],
            "binary_masks_path":  record["binary_masks_path"],
        })

    return records


def run_gating_and_save_csv(smoke=False):
    """Run gating over train/val/test splits and save one CSV per split. Run once."""
    metadata_index = load_metadata_index(MEDSAM_METADATA)
    limit = 5 if smoke else None

    for split in ("train", "val", "test"):
        case_ids    = list(pd.read_csv(SPLITS_DIR / f"{split}.csv")["case_id"])
        if limit:
            case_ids = case_ids[:limit]
        all_records = []

        for case_id in tqdm(case_ids, desc=f"[{split}]", unit="case"):
            if case_id not in metadata_index:
                continue
            all_records.extend(gate_case(case_id, metadata_index))

        df      = pd.DataFrame(all_records)
        out_csv = CSV_DIR / f"gated_{split}_records.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} {split} fragments -> {out_csv}")
        print(df["expert"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on 5 cases per split only")
    args = parser.parse_args()
    run_gating_and_save_csv(smoke=args.smoke)
