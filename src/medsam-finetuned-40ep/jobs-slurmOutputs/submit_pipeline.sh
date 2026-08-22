#!/bin/bash
# Submit the 40-epoch fine-tuned MedSAM continuation + evaluation pipeline as chained
# SLURM jobs:
#   train (afterok: 20ep train job)  →  infer  →  evaluate  →  summarize
#
# The train step (3_train_medsam_pengwin_40ep.job) resumes from the 20-epoch run's
# checkpoint and continues to 40 epochs total; it only starts once the 20ep train job
# has actually completed (--dependency=afterok), since that's what guarantees a real
# 20-epoch checkpoint exists to resume from.
#
# Usage (from any directory):
#   bash src/medsam-finetuned-40ep/jobs-slurmOutputs/submit_pipeline.sh <train20ep_job_id>
#
# <train20ep_job_id> defaults to 25935579 (2_train_medsam_pengwin_20ep.job) if omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JID_TRAIN20="${1:-25935579}"

JOB_TRAIN40="/gpfs/home5/aramautar2/projects/MoE-ShapeRefine-MedicalSeg/medSAM-stage1/scripts/jobs-slurmOutputs/3_train_medsam_pengwin_40ep.job"
JOB_INFER="${SCRIPT_DIR}/1_infer_medsam_finetuned_40ep.job"
JOB_EVAL="${SCRIPT_DIR}/2_evaluate_medsam_finetuned_40ep.job"
JOB_SUM="${SCRIPT_DIR}/3_summarize_medsam_finetuned_40ep.job"

echo "Chaining 40-epoch MedSAM continuation + eval pipeline after 20ep train job ${JID_TRAIN20}..."

JID_TRAIN40=$(sbatch --parsable --dependency=afterok:"${JID_TRAIN20}" "$JOB_TRAIN40")
echo "  [0/3] train      → job ${JID_TRAIN40}  (after ${JID_TRAIN20})"

JID_INFER=$(sbatch --parsable --dependency=afterok:"${JID_TRAIN40}" "$JOB_INFER")
echo "  [1/3] infer      → job ${JID_INFER}  (after ${JID_TRAIN40})"

JID_EVAL=$(sbatch --parsable --dependency=afterok:"${JID_INFER}" "$JOB_EVAL")
echo "  [2/3] evaluate   → job ${JID_EVAL}  (after ${JID_INFER})"

JID_SUM=$(sbatch --parsable --dependency=afterok:"${JID_EVAL}" "$JOB_SUM")
echo "  [3/3] summarize  → job ${JID_SUM}  (after ${JID_EVAL})"

echo ""
echo "Monitor with:  squeue -j ${JID_TRAIN20},${JID_TRAIN40},${JID_INFER},${JID_EVAL},${JID_SUM}"
echo "Summary CSVs will be written to:"
echo "  data/medsam-finetuned-40ep-predictions/summary_full.csv"
echo "  data/medsam-finetuned-40ep-predictions/summary_overlap.csv"
