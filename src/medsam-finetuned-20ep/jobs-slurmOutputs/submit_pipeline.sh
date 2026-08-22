#!/bin/bash
# Submit the full 20-epoch fine-tuned MedSAM evaluation pipeline as chained SLURM jobs:
#   train (already running)  →  infer  →  evaluate  →  summarize
#
# Usage (from any directory):
#   bash src/medsam-finetuned-20ep/jobs-slurmOutputs/submit_pipeline.sh <train_job_id>
#
# <train_job_id> defaults to 25935579 (2_train_medsam_pengwin_20ep.job) if omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JID_TRAIN="${1:-25935579}"

JOB_INFER="${SCRIPT_DIR}/1_infer_medsam_finetuned_20ep.job"
JOB_EVAL="${SCRIPT_DIR}/2_evaluate_medsam_finetuned_20ep.job"
JOB_SUM="${SCRIPT_DIR}/3_summarize_medsam_finetuned_20ep.job"

echo "Chaining 20-epoch MedSAM eval pipeline after train job ${JID_TRAIN}..."

JID_INFER=$(sbatch --parsable --dependency=afterok:"${JID_TRAIN}" "$JOB_INFER")
echo "  [1/3] infer      → job ${JID_INFER}  (after ${JID_TRAIN})"

JID_EVAL=$(sbatch --parsable --dependency=afterok:"${JID_INFER}" "$JOB_EVAL")
echo "  [2/3] evaluate   → job ${JID_EVAL}  (after ${JID_INFER})"

JID_SUM=$(sbatch --parsable --dependency=afterok:"${JID_EVAL}" "$JOB_SUM")
echo "  [3/3] summarize  → job ${JID_SUM}  (after ${JID_EVAL})"

echo ""
echo "Monitor with:  squeue -j ${JID_TRAIN},${JID_INFER},${JID_EVAL},${JID_SUM}"
echo "Summary CSVs will be written to:"
echo "  data/medsam-finetuned-20ep-predictions/summary_full.csv"
echo "  data/medsam-finetuned-20ep-predictions/summary_overlap.csv"
