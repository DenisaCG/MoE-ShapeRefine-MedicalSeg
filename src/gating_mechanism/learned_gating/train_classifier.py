#!/usr/bin/env python3
"""Train a logistic-regression gating classifier on shape descriptors.

Learns to predict oracle_expert (the actually-better expert, per
oracle_route.py) from 5 shape descriptors already extracted per fragment:
area, aspect_ratio, elongation, compactness, connected_components.

Input:
    data_distribution_results/data_analysis_medsam/features.csv
        (shape descriptors, one row per MedSAM fragment)
    learned_gating/oracle_labels_{backend}_{split}.csv
        (ground-truth "correct expert" labels, from oracle_route.py)

Output:
    checkpoints/gating_classifier/{backend}_logreg.joblib
        {"pipeline": fitted sklearn Pipeline(StandardScaler, LogisticRegression),
         "feature_cols": [...]}
        Fit on the VAL split, not train -- see the fit-split comment in
        main() for why: each train-split fragment was already used to train
        one of the two experts (build_expert_dataloaders splits train by the
        fixed-threshold "expert" column), so oracle labels on train are
        biased toward whichever expert already memorized that fragment. Val
        was never used for gradient updates (only _best.pth checkpoint
        selection), so it's the cleanest split available to fit on; test is
        held out entirely for reporting. Ready for gating_mechanism.py's
        --routing learned --classifier-path <this file>.

Usage:
    python train_classifier.py --backend cnnroi
    python train_classifier.py --backend flowsdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEARNED_DIR  = PROJECT_ROOT / "src" / "gating_mechanism" / "learned_gating"

DEFAULT_FEATURES_CSV = (
    PROJECT_ROOT / "data_distribution_results" / "data_analysis_medsam" / "features.csv"
)
FEATURE_COLS = ["area", "aspect_ratio", "elongation", "compactness", "connected_components"]
JOIN_KEYS    = ["case_id", "medsam_instance_id", "fragment_id"]


def load_split(backend: str, split: str, oracle_dir: Path, features: pd.DataFrame) -> pd.DataFrame:
    oracle_path = oracle_dir / f"oracle_labels_{backend}_{split}.csv"
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"missing {oracle_path}; run oracle_route.py --backend {backend} --split {split} first"
        )
    oracle = pd.read_csv(oracle_path)
    merged = oracle.merge(features[JOIN_KEYS + FEATURE_COLS], on=JOIN_KEYS, how="left")

    # "area" is never NaN for a genuine match -- an area NaN means the join
    # found no row at all (stale/incomplete features.csv), a real problem.
    no_match = merged["area"].isna()
    if no_match.any():
        raise ValueError(
            f"{int(no_match.sum())} fragments in {oracle_path} have no matching row "
            f"in the shape-descriptor CSV at all -- recompute features.csv first"
        )

    # A tiny fraction of fragments are geometrically degenerate (near-zero
    # area) and legitimately have NaN aspect_ratio/elongation/compactness
    # (see shape_features.py). Unlike gating_mechanism.py's route_learned()
    # (which must assign every fragment an expert), dropping a handful of
    # these from the training/reporting set is harmless here.
    degenerate = merged[FEATURE_COLS].isna().any(axis=1)
    if degenerate.any():
        print(
            f"[{oracle_path.name}] dropping {int(degenerate.sum())}/{len(merged)} "
            "fragments with degenerate shape descriptors (near-zero-area masks)"
        )
        merged = merged.loc[~degenerate].reset_index(drop=True)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["cnnroi", "cnnnoroi", "flowsdf"], default="cnnroi",
                        help="Which oracle_labels_{backend}_*.csv set to train against.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=LEARNED_DIR,
                        help="Directory containing oracle_labels_{backend}_{split}.csv.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "gating_classifier")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength. Ignored if --tune-C is set.")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                        help="LogisticRegression class_weight. 'none' disables balancing.")
    parser.add_argument("--tune-C", action="store_true",
                        help="Pick C via k-fold CV on the val split (LogisticRegressionCV) "
                             "instead of using --C directly. Train is contaminated and test "
                             "must stay held out, so val is the only split it's sound to "
                             "cross-validate on.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds for --tune-C.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features_csv.exists():
        raise FileNotFoundError(
            f"missing {args.features_csv}; run "
            "data_distribution/analyze_dataset.py --mask-source medsam first"
        )
    features = pd.read_csv(args.features_csv)

    train = load_split(args.backend, "train", args.oracle_dir, features)
    val   = load_split(args.backend, "val",   args.oracle_dir, features)
    test  = load_split(args.backend, "test",  args.oracle_dir, features)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"val oracle_expert distribution:\n{val['oracle_expert'].value_counts()}\n")

    # Fit on VAL, not train. Each train-split fragment was already assigned to
    # exactly one expert by the fixed threshold (see dataset.py's
    # build_expert_dataloaders: `df[df.expert == expert]`), and that expert
    # did gradient updates on it -- so oracle_route.py's "run both experts,
    # keep the higher-dice one" is biased toward whichever expert the
    # fragment was already routed to on train (a memorization artifact, not a
    # genuine shape signal). Val fragments were never used for gradient
    # updates by either expert -- only for _best.pth checkpoint selection
    # (lowest val loss), which is the standard, accepted kind of val "leakage"
    # -- so val's oracle labels are clean(ish) and safe to fit on. Test is
    # untouched by either expert during training or checkpoint selection and
    # is kept fully held out for reporting.
    class_weight = None if args.class_weight == "none" else "balanced"
    if args.tune_C:
        clf = LogisticRegressionCV(
            class_weight=class_weight, cv=args.cv_folds, max_iter=1000,
            random_state=args.seed, scoring="accuracy",
        )
    else:
        clf = LogisticRegression(
            class_weight=class_weight, C=args.C, max_iter=1000, random_state=args.seed,
        )
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipeline.fit(val[FEATURE_COLS].to_numpy(), val["oracle_expert"].to_numpy())
    if args.tune_C:
        print(f"tuned C (via {args.cv_folds}-fold CV on val) = {clf.C_[0]:.4g}")

    for name, split_df, caveat in (
        ("train", train, " [CONTAMINATED -- each expert already trained on ~half these "
                          "fragments; oracle labels are biased toward the existing fixed "
                          "routing. Reference only -- do not use to judge the classifier "
                          "or the fixed_expert-vs-oracle comparison below.]"),
        ("val",   val,   " [fit split -- in-sample, not a generalization estimate]"),
        ("test",  test,  " [held out -- the trustworthy number]"),
    ):
        preds = pipeline.predict(split_df[FEATURE_COLS].to_numpy())
        acc = accuracy_score(split_df["oracle_expert"], preds)
        print(f"\n[{name}] accuracy={acc:.4f}  n={len(split_df)}{caveat}")
        print(classification_report(split_df["oracle_expert"], preds, zero_division=0))

        # For free: how the *current* gating CSV's routing (fixed_expert,
        # carried through from oracle_route.py) compares to the same oracle
        # ground truth, on the same rows.
        fixed_acc = accuracy_score(split_df["oracle_expert"], split_df["fixed_expert"])
        print(f"[{name}] fixed_expert (input gating CSV) accuracy vs oracle: {fixed_acc:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.backend}_logreg.joblib"
    joblib.dump({"pipeline": pipeline, "feature_cols": FEATURE_COLS}, out_path)
    print(f"\nSaved classifier -> {out_path}")


if __name__ == "__main__":
    main()
