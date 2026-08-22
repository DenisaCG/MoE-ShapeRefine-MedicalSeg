import pandas as pd

THRESHOLD_1024 = 5402
THRESHOLD_REL = THRESHOLD_1024 / (1024 * 1024)
JOIN_KEYS = ["case_id", "medsam_instance_id", "fragment_id"]

GT_CSV = "/home/aramautar2/projects/MoE-ShapeRefine-MedicalSeg/data_distribution_results/data_analysis_gt_with_medsam/features.csv"
MEDSAM_CSV = "/home/aramautar2/projects/MoE-ShapeRefine-MedicalSeg/data_distribution_results/data_analysis_medsam/features.csv"
SPLITS_DIR = "/home/aramautar2/projects/data/pengwin/splits"


def route(relative_area):
    return relative_area.le(THRESHOLD_REL).map({True: "expert_small", False: "expert_large"})


# features.csv's own "split" column is unpopulated (hardcoded "unknown" in
# analyze_dataset.py), so rebuild real train/val/test membership from the
# case-level split files used by gating_mechanism.py / build_expert_dataloaders().
case_to_split = {}
for split_name in ("train", "val", "test"):
    case_ids = pd.read_csv(f"{SPLITS_DIR}/{split_name}.csv")["case_id"]
    case_to_split.update({cid: split_name for cid in case_ids})

gt = pd.read_csv(GT_CSV, usecols=JOIN_KEYS + ["relative_area"])
medsam = pd.read_csv(MEDSAM_CSV, usecols=JOIN_KEYS + ["relative_area"])
gt["split"] = gt["case_id"].map(case_to_split)

gt["route_gt"] = route(gt["relative_area"])
medsam["route_medsam"] = route(medsam["relative_area"])

merged = gt.merge(
    medsam[JOIN_KEYS + ["route_medsam", "relative_area"]],
    on=JOIN_KEYS,
    how="inner",
    suffixes=("_gt", "_medsam"),
)

n_gt = len(gt)
n_medsam = len(medsam)
n_joined = len(merged)
mismatches = merged["route_gt"] != merged["route_medsam"]
n_mismatch = int(mismatches.sum())
rate = 100.0 * n_mismatch / n_joined

print(f"GT rows: {n_gt}, MedSAM rows: {n_medsam}, joined (matched by {JOIN_KEYS}): {n_joined}")
print(f"Threshold: {THRESHOLD_1024} px @1024x1024  ==  relative_area <= {THRESHOLD_REL:.6f} -> expert_small")
print(f"Mismatched fragments: {n_mismatch} / {n_joined}")
print(f"Misrouting rate = {rate:.3f}%")

print("\nConfusion (GT route x MedSAM route):")
print(pd.crosstab(merged["route_gt"], merged["route_medsam"]))

print("\nBy split:")
by_split = merged.assign(mismatch=mismatches).groupby("split")["mismatch"].agg(["sum", "count"])
by_split["rate_%"] = 100.0 * by_split["sum"] / by_split["count"]
print(by_split)

print("\nMismatch cases: distance from threshold (relative_area, both routes), describe():")
mism = merged[mismatches]
print(mism[["relative_area_gt", "relative_area_medsam"]].describe())
