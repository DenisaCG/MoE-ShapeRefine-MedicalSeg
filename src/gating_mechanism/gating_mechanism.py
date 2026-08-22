import json
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PROJECT_ROOT, TOPLEVEL_DATA_ROOT

MEDSAM_METADATA   = PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl"
SPLITS_DIR        = TOPLEVEL_DATA_ROOT / "pengwin" / "splits"
CSV_DIR           = Path(__file__).parent
IMAGE_SIZE        = 1024
TOTAL_PX          = IMAGE_SIZE * IMAGE_SIZE

# Shape-descriptor table produced by data_distribution/analyze_dataset.py
# --mask-source medsam. Required by --routing learned.
DEFAULT_FEATURES_CSV = (
    PROJECT_ROOT / "data_distribution_results" / "data_analysis_medsam" / "features.csv"
)
# Default classifier trained by learned_gating/train_classifier.py. Must be
# retrained per backend (cnnroi/flowsdf) -- "the better expert" is
# architecture-specific, see learned_gating/oracle_route.py.
DEFAULT_CLASSIFIER_PATH = (
    PROJECT_ROOT / "checkpoints" / "gating_classifier" / "cnnroi_logreg.joblib"
)
LEARNED_FEATURE_COLS = [
    "area", "aspect_ratio", "elongation", "compactness", "connected_components",
]
LEARNED_JOIN_KEYS = ["case_id", "medsam_instance_id", "fragment_id"]

# Canonical area routing threshold (foreground pixel count). This is the
# "current routing" arm of the gating ablation — random routing always
# matches the expert_small ratio this threshold produces, regardless of
# the --threshold value used for area routing.
DEFAULT_THRESHOLD = 5402


def load_metadata_index(metadata_path: Path) -> dict:
    index = {}
    with metadata_path.open() as f:
        for line in f:
            record = json.loads(line.strip())
            index[record["case_id"]] = record
    return index


def route_to_expert(area, threshold=DEFAULT_THRESHOLD):
    return "expert_small" if area <= threshold else "expert_large"


def gate_case(case_id: str, metadata_index: dict) -> list[dict]:
    """
    Load MedSAM binary masks for case_id and compute per-fragment area.

    Masks shape : (N, 1024, 1024) binary.
    Fragment i  : masks[i] corresponds to metadata fragments[i] (medsam_instance_id = i+1).

    Does NOT assign an expert — the "expert" key is a placeholder here so the
    CSV column order matches the original schema exactly. Routing is applied
    afterward, over the whole split, by `compute_expert_column()`, since
    random routing needs to see the whole split to preserve a target ratio.
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
            "expert":             None,  # filled in by compute_expert_column()
            "embedding_path":     record["embedding_path"],
            "binary_masks_path":  record["binary_masks_path"],
        })

    return records


def route_random(n: int, small_ratio: float, seed: int) -> np.ndarray:
    """Randomly assign n fragments to expert_small/expert_large.

    Selects exactly round(small_ratio * n) fragments (by position) as
    expert_small using a seeded Generator, so the assignment is reproducible.
    """
    rng = np.random.default_rng(seed)
    n_small = int(round(small_ratio * n))
    labels = np.full(n, "expert_large", dtype=object)
    small_positions = rng.choice(n, size=n_small, replace=False)
    labels[small_positions] = "expert_small"
    return labels


def route_learned(
    df: pd.DataFrame, features_path: Path, classifier_path: Path,
    learned_threshold: float = 0.5,
) -> pd.Series:
    """Route fragments with a trained shape-descriptor classifier.

    Joins df (case_id, medsam_instance_id, fragment_id) against the shape-
    descriptor table on those same keys, then predicts with the sklearn
    Pipeline saved by learned_gating/train_classifier.py.

    learned_threshold: probability cutoff for expert_small (default 0.5,
    sklearn's own .predict() decision boundary). The classifier's own
    training (see train_classifier.py) is imbalance-sensitive -- with
    class_weight=None it collapses toward almost always predicting
    expert_large (test recall on true-small fragments as low as 0.03-0.06).
    Lowering this threshold is a cheap way to trade some of that back
    without retraining: it makes the classifier flag expert_small more
    readily, at the cost of more false expert_small calls on fragments
    that are actually large.
    """
    import joblib

    if not features_path.exists():
        raise FileNotFoundError(
            f"Shape-descriptor CSV not found: {features_path}\n"
            "Run data_distribution/analyze_dataset.py --mask-source medsam first."
        )
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Gating classifier not found: {classifier_path}\n"
            "Run learned_gating/train_classifier.py first."
        )

    features = pd.read_csv(features_path)
    merged = df[LEARNED_JOIN_KEYS].merge(
        features[LEARNED_JOIN_KEYS + LEARNED_FEATURE_COLS],
        on=LEARNED_JOIN_KEYS, how="left",
    )
    if len(merged) != len(df):
        raise ValueError(
            f"join produced {len(merged)} rows from {len(df)} input fragments -- "
            f"{features_path} likely has duplicate {LEARNED_JOIN_KEYS} keys."
        )

    # "area" is never NaN in features.csv for a genuine match (only the shape
    # ratios can be NaN, for degenerate geometry -- see below), so an area
    # NaN here means the join found no row at all: a real integrity problem
    # (stale/incomplete features.csv), not a degenerate-fragment edge case.
    no_match = merged["area"].isna().to_numpy()
    if no_match.any():
        raise ValueError(
            f"{int(no_match.sum())} fragments have no matching row in {features_path} "
            "at all (stale or incomplete shape-descriptor table -- recompute it)."
        )

    bundle = joblib.load(classifier_path)
    # A tiny fraction of fragments are geometrically degenerate (near-zero
    # area, zero height/width) and shape_features.py leaves aspect_ratio/
    # elongation/compactness as NaN for them by design -- not a data error.
    # A learned prediction isn't meaningful there anyway, so fall back to
    # area-threshold routing for just those (every fragment must get *some*
    # expert -- unlike train_classifier.py, dropping rows isn't an option).
    degenerate = merged[LEARNED_FEATURE_COLS].isna().any(axis=1).to_numpy()
    predictions = np.empty(len(df), dtype=object)

    if degenerate.any():
        print(
            f"WARNING: {int(degenerate.sum())} fragments have degenerate shape "
            "descriptors (near-zero-area or zero-height/width masks); falling back "
            "to area-threshold routing for just those."
        )
        area_values = df["area"].to_numpy()[degenerate]
        predictions[degenerate] = [route_to_expert(a, DEFAULT_THRESHOLD) for a in area_values]

    clean = ~degenerate
    if clean.any():
        clean_X = merged.loc[clean, bundle["feature_cols"]].to_numpy()
        pipeline = bundle["pipeline"]
        if learned_threshold == 0.5:
            predictions[clean] = pipeline.predict(clean_X)
        else:
            classes = pipeline.classes_
            small_idx = list(classes).index("expert_small")
            proba_small = pipeline.predict_proba(clean_X)[:, small_idx]
            predictions[clean] = np.where(
                proba_small >= learned_threshold, "expert_small", "expert_large",
            )

    return pd.Series(predictions, index=df.index)


def compute_expert_column(
    df: pd.DataFrame,
    routing: str,
    threshold: float,
    seed: int,
    features_path: Path = DEFAULT_FEATURES_CSV,
    classifier_path: Path = DEFAULT_CLASSIFIER_PATH,
    learned_threshold: float = 0.5,
) -> pd.Series:
    """Compute the "expert" column for one split's fragment dataframe.

    routing == "area":    deterministic area-threshold routing. Matches today's
                           behaviour exactly when threshold == DEFAULT_THRESHOLD.
    routing == "random":  reproducible random assignment that preserves the
                           same expert_small ratio DEFAULT_THRESHOLD (5402) area
                           routing would produce for this split — --threshold is
                           ignored in this mode.
    routing == "learned": predicts the expert per-fragment from shape
                           descriptors with a trained classifier (see
                           learned_gating/train_classifier.py). --threshold
                           and --seed are ignored in this mode.
    """
    if routing == "area":
        return df["area"].apply(lambda area: route_to_expert(area, threshold))

    if routing == "random":
        baseline    = df["area"].apply(lambda area: route_to_expert(area, DEFAULT_THRESHOLD))
        small_ratio = (baseline == "expert_small").mean()
        labels      = route_random(len(df), small_ratio, seed)
        return pd.Series(labels, index=df.index)

    if routing == "learned":
        return route_learned(df, features_path, classifier_path, learned_threshold)

    raise ValueError(f"unknown routing mode: {routing!r} (expected 'area', 'random', or 'learned')")


def compute_split_stats(split: str, df: pd.DataFrame) -> dict:
    """Count expert_small/expert_large fragments and their percentages for one split."""
    total   = len(df)
    n_small = int((df["expert"] == "expert_small").sum())
    n_large = int((df["expert"] == "expert_large").sum())
    return {
        "split":     split,
        "total":     total,
        "n_small":   n_small,
        "pct_small": (100.0 * n_small / total) if total else float("nan"),
        "n_large":   n_large,
        "pct_large": (100.0 * n_large / total) if total else float("nan"),
    }


def print_split_stats(stats: dict) -> None:
    print(
        f"[{stats['split']}] total={stats['total']}  "
        f"expert_small={stats['n_small']} ({stats['pct_small']:.2f}%)  "
        f"expert_large={stats['n_large']} ({stats['pct_large']:.2f}%)"
    )


def print_routing_summary_table(all_stats: list[dict]) -> None:
    """Combined small/large ratio table across train/val/test.

    Particularly useful for --routing random, to confirm the random
    assignment reproduced the same expert_small ratio the 5402-threshold
    area routing would have produced for each split.
    """
    header = (
        f"{'split':<8}{'total':>10}{'small_n':>10}{'small_%':>10}"
        f"{'large_n':>10}{'large_%':>10}"
    )
    print("\nRouting split summary (all splits):")
    print(header)
    print("-" * len(header))
    for stats in all_stats:
        print(
            f"{stats['split']:<8}{stats['total']:>10}"
            f"{stats['n_small']:>10}{stats['pct_small']:>9.2f}%"
            f"{stats['n_large']:>10}{stats['pct_large']:>9.2f}%"
        )
    print()


def run_gating_and_save_csv(
    smoke: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    routing: str = "area",
    seed: int = 42,
    out_dir: Path | None = None,
    metadata_path: Path | None = None,
    features_path: Path = DEFAULT_FEATURES_CSV,
    classifier_path: Path = DEFAULT_CLASSIFIER_PATH,
    learned_threshold: float = 0.5,
) -> None:
    """Run gating over train/val/test splits and save one CSV per split. Run once.

    Default arguments (threshold=DEFAULT_THRESHOLD, routing="area", out_dir=None,
    metadata_path=None) reproduce the original CSVs exactly — same columns, same
    values, same destination. Passing --threshold, --routing random, --out-dir,
    and/or --metadata-path is additive: it only changes output when explicitly
    requested (e.g. the gating ablation study, or gating a different model's
    predictions such as data/sam-predictions/metadata.jsonl).
    """
    metadata_index = load_metadata_index(metadata_path or MEDSAM_METADATA)
    limit   = 5 if smoke else None
    out_dir = out_dir or CSV_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    is_default_run = (routing == "area" and threshold == DEFAULT_THRESHOLD)
    if out_dir == CSV_DIR and not is_default_run:
        print(
            f"WARNING: routing={routing!r} threshold={threshold} will be written into "
            f"the canonical CSV directory {CSV_DIR}, overwriting the CSVs used by "
            "default training/inference runs. Pass --out-dir to write ablation CSVs "
            "elsewhere instead."
        )

    all_split_stats = []

    for split in ("train", "val", "test"):
        case_ids    = list(pd.read_csv(SPLITS_DIR / f"{split}.csv")["case_id"])
        if limit:
            # Random (seeded) rather than case_ids[:limit] -- the first N cases
            # per split are a fixed, non-representative slice (e.g. they can
            # systematically miss the rare ~8% expert_small class), so every
            # --smoke run would otherwise reproduce the same skew.
            rng = np.random.default_rng(seed)
            n = min(limit, len(case_ids))
            case_ids = list(rng.choice(case_ids, size=n, replace=False))
        all_records = []

        for case_id in tqdm(case_ids, desc=f"[{split}]", unit="case"):
            if case_id not in metadata_index:
                continue
            all_records.extend(gate_case(case_id, metadata_index))

        df           = pd.DataFrame(all_records)
        df["expert"] = compute_expert_column(
            df, routing, threshold, seed, features_path, classifier_path, learned_threshold,
        )

        if not is_default_run:
            # Provenance columns — only added for non-canonical routing so the
            # default CSVs stay byte-for-byte identical to the original schema.
            df["routing_mode"]      = routing
            df["routing_seed"]      = seed if routing == "random" else pd.NA
            df["routing_threshold"] = threshold if routing == "area" else pd.NA

        out_csv = out_dir / f"gated_{split}_records.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} {split} fragments -> {out_csv}")
        print(df["expert"].value_counts().to_string())

        stats = compute_split_stats(split, df)
        print_split_stats(stats)
        all_split_stats.append(stats)

    print_routing_summary_table(all_split_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ if __doc__ else None)
    parser.add_argument("--smoke", action="store_true", help="Run on 5 cases per split only")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=(
            "Foreground-pixel-area threshold for --routing area "
            f"(default {DEFAULT_THRESHOLD}, matches the canonical CSVs). Ignored "
            "when --routing random is used."
        ),
    )
    parser.add_argument(
        "--routing", choices=["area", "random", "learned"], default="area",
        help=(
            "'area' (default): route by foreground pixel area vs --threshold. "
            "'random': randomly assign fragments to expert_small/expert_large, "
            f"preserving the same expert_small ratio the default area-"
            f"{DEFAULT_THRESHOLD} routing produces for each split (computed "
            "independently per split), using --seed for reproducibility. "
            "'learned': predict per-fragment from shape descriptors with a "
            "trained classifier (--features-csv, --classifier-path)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --routing random (ignored otherwise).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Directory to write gated_{split}_records.csv into. Defaults to "
            "this file's directory (the canonical CSV location). Pass a "
            "different directory for ablation runs so the canonical CSVs "
            "are not overwritten."
        ),
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=None,
        help=(
            f"Path to the predictions metadata.jsonl to gate (default "
            f"{MEDSAM_METADATA}). Pass e.g. data/sam-predictions/metadata.jsonl "
            "to gate a different model's predictions instead of MedSAM's."
        ),
    )
    parser.add_argument(
        "--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
        help="Shape-descriptor CSV for --routing learned (default: %(default)s).",
    )
    parser.add_argument(
        "--classifier-path", type=Path, default=DEFAULT_CLASSIFIER_PATH,
        help=(
            "Trained gating classifier for --routing learned (default: "
            "%(default)s). Must match the expert backend (cnnroi/flowsdf) "
            "used downstream -- see learned_gating/train_classifier.py."
        ),
    )
    parser.add_argument(
        "--learned-threshold", type=float, default=0.5,
        help=(
            "Probability cutoff for expert_small under --routing learned "
            "(default 0.5, sklearn's own .predict() boundary). Lower it to "
            "make the classifier flag expert_small more readily -- useful "
            "since class_weight=None training collapses toward almost "
            "always predicting expert_large (see train_classifier.py)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gating_and_save_csv(
        smoke=args.smoke,
        threshold=args.threshold,
        routing=args.routing,
        seed=args.seed,
        out_dir=args.out_dir,
        metadata_path=args.metadata_path,
        features_path=args.features_csv,
        classifier_path=args.classifier_path,
        learned_threshold=args.learned_threshold,
    )
