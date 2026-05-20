import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_utils import load_pengwin_label, decode_pengwin_fragment_from_record, resolve_existing_path, load_prediction_masks

CSV_DIR = Path(__file__).parent
GT_DIR  = Path("/gpfs/home5/scur0509/projects/data/pengwin/original/task2_xray/train/output/images/x-ray")


class FragmentDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        embedding   = np.load(resolve_existing_path(rec["embedding_path"]))        # (256, 64, 64)
        masks       = load_prediction_masks(rec["binary_masks_path"])                      # (N, 1024, 1024)
        binary_mask = masks[rec["medsam_instance_id"] - 1]                         # (1024, 1024)

        gt_path = GT_DIR / f"{rec['sample_name'].replace('XRAY_PENGWIN_', '')}.tif"
        gt      = load_pengwin_label(gt_path)                                      # (H, W) uint32
        gt_mask = decode_pengwin_fragment_from_record(gt, rec)                     # (H, W) uint8

        return {
            "embedding":   torch.from_numpy(embedding).float(),                    # (256, 64, 64)
            "binary_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),     # (1, 1024, 1024)
            "gt_mask":     torch.from_numpy(gt_mask).unsqueeze(0).float(),         # (1, 448, 448)
        }


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    """Subsample n records from df preserving the SA/LI/RI class distribution."""
    return (
        df.groupby("category_name", group_keys=False)
        .apply(lambda x: x.sample(frac=n / len(df), random_state=seed))
        .reset_index(drop=True)
        .to_dict("records")
    )


def build_expert_dataloaders(
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    large_subsample: int | None = None,
):
    """
    Load gated_{split}_records.csv and return one DataLoader per expert.
    Run gating_mechanism.py first if the CSV does not exist.

    Args:
        large_subsample: if set, subsample expert_large to this many fragments
                         while preserving SA/LI/RI class proportions.
    """
    csv_path = CSV_DIR / f"gated_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python gating_mechanism.py` first."
        )

    df      = pd.read_csv(csv_path)
    shuffle = split == "train"

    loaders = {}
    for expert in ["expert_small", "expert_large"]:
        subset = df[df.expert == expert]

        if expert == "expert_large" and large_subsample is not None:
            n = min(large_subsample, len(subset))
            records = stratified_subsample(subset, n=n)
        else:
            records = subset.to_dict("records")

        print(f"[{split}] {expert}: {len(records)} fragments")
        loaders[expert] = DataLoader(
            FragmentDataset(records),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders
