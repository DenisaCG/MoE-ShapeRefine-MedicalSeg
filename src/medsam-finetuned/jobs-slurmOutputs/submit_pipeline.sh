#!/bin/bash
# Submit the full fine-tuned MedSAM evaluation pipeline as chained SLURM jobs:
#   1_infer  →  2_evaluate  →  3_summarize
#
# Usage (from any directory):
#   bash src/medsam-finetuned/jobs-slurmOutputs/submit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_INFER="${SCRIPT_DIR}/1_infer_medsam_finetuned.job"
JOB_EVAL="${SCRIPT_DIR}/2_evaluate_medsam_finetuned.job"
JOB_SUM="${SCRIPT_DIR}/3_summarize_medsam_finetuned.job"

echo "Submitting fine-tuned MedSAM pipeline..."

JID_INFER=$(sbatch --parsable "$JOB_INFER")
echo "  [1/3] infer      → job ${JID_INFER}"

JID_EVAL=$(sbatch --parsable --dependency=afterok:"${JID_INFER}" "$JOB_EVAL")
echo "  [2/3] evaluate   → job ${JID_EVAL}  (after ${JID_INFER})"

JID_SUM=$(sbatch --parsable --dependency=afterok:"${JID_EVAL}" "$JOB_SUM")
echo "  [3/3] summarize  → job ${JID_SUM}  (after ${JID_EVAL})"

echo ""
echo "Monitor with:  squeue -j ${JID_INFER},${JID_EVAL},${JID_SUM}"
echo "Summary CSVs will be written to:"
echo "  data/medsam-finetuned-predictions/summary_full.csv"
echo "  data/medsam-finetuned-predictions/summary_overlap.csv"
