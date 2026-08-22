# Learned Gating — Full Retrain (not an ablation)

## Why this is a separate directory from `gating_ablation/`

`gating_ablation/strategy4_learned_*` only ever swaps the *routing rule* while reusing
the original size-trained experts unchanged. That's a real ablation — it isolates the
routing rule's effect by holding everything else fixed — but it structurally can't
show a shape-based partition's true potential: an expert that was only ever trained
on the size-based split has never seen the kinds of fragments a different partition
would send it, so it's being tested unfairly regardless of how good the new grouping
actually is. A bad `strategy4` result rules out a free win from swapping the router
alone; it does not rule out that a shape-based partition is fundamentally better.

This directory is the actual test of that: retrain both experts from scratch on the
classifier's own partition, so no confound is left. It changes two things at once
(routing + what each expert learned), so it can't cleanly attribute a result to
routing alone the way `gating_ablation` could — but it's the only way to see the
learned partition's real ceiling.

## Pipeline

1. Shape features from MedSAM masks — done, `data_distribution_results/data_analysis_medsam/features.csv`
2. Oracle labels (run both experts on every fragment, keep whichever wins) — done, `learned_gating/oracle_labels_{backend}_{split}.csv`
3. Classifier trained on val oracle labels (train is leakage-contaminated) — done, `checkpoints/gating_classifier/{backend}_logreg.joblib`
4. **Apply the classifier to the TRAIN split** to get new `expert_small`/`expert_large` assignments, then **retrain both experts from scratch** on that new partition — not started, this directory's job.

Step 4 needs a decision on which classifier config to relabel train with — see
`INTERPRETATION.md`'s "Oracle agreement" section (and its "Follow-up: threshold
calibration" sub-section) for the candidates and their tradeoffs before picking one.

## Status

Not started. See `/home/aramautar2/projects/ToDOs.md` for the current decision gate
(waiting on the `thresh040` downstream-Dice check in `gating_ablation/` before
committing to this retrain's GPU cost).
