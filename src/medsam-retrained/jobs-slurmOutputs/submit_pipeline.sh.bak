#!/bin/bash
# Submit the full retrained MedSAM pipeline as chained SLURM jobs:
#   train  →  infer  →  evaluate  →  summarize
#
# Usage (from any directory):
#   bash src/medsam-retrained/jobs-slurmOutputs/submit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_JOB="/gpfs/home5/scur0509/projects/MoE-ShapeRefine-MedicalSeg/medSAM-stage1/scripts/jobs-slurmOutputs/1_train_medsam_pengwin.job"

JOB_INFER="${SCRIPT_DIR}/1_infer_medsam_retrained.job"
JOB_EVAL="${SCRIPT_DIR}/2_evaluate_medsam_retrained.job"
JOB_SUM="${SCRIPT_DIR}/3_summarize_medsam_retrained.job"

echo "Submitting retrained MedSAM pipeline..."

JID_TRAIN=$(sbatch --parsable "$TRAIN_JOB")
echo "  [1/4] train      → job ${JID_TRAIN}"

JID_INFER=$(sbatch --parsable --dependency=afterok:"${JID_TRAIN}" "$JOB_INFER")
echo "  [2/4] infer      → job ${JID_INFER}  (after ${JID_TRAIN})"

JID_EVAL=$(sbatch --parsable --dependency=afterok:"${JID_INFER}" "$JOB_EVAL")
echo "  [3/4] evaluate   → job ${JID_EVAL}  (after ${JID_INFER})"

JID_SUM=$(sbatch --parsable --dependency=afterok:"${JID_EVAL}" "$JOB_SUM")
echo "  [4/4] summarize  → job ${JID_SUM}  (after ${JID_EVAL})"

echo ""
echo "Monitor with:  squeue -j ${JID_TRAIN},${JID_INFER},${JID_EVAL},${JID_SUM}"
echo "Summary CSVs will be written to:"
echo "  data/medsam-retrained-predictions/summary_full.csv"
echo "  data/medsam-retrained-predictions/summary_overlap.csv"
