#!/bin/bash

set -euo pipefail

cd /gpfs/home5/scur0509/projects/MoE-ShapeRefine-MedicalSeg/src/FlowSDF

JOB_DIR="jobs-slurmOutputs"

echo "Submitting FlowSDF test workflow..."
INFER_JOB_ID=$(sbatch --parsable "${JOB_DIR}/2_infer_flowsdf.job")
echo "Submitted inference job: ${INFER_JOB_ID}"

EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:"${INFER_JOB_ID}" "${JOB_DIR}/4_evaluate_flowsdf.job")
echo "Submitted evaluation job: ${EVAL_JOB_ID}"

VIS_JOB_ID=$(sbatch --parsable --dependency=afterok:"${EVAL_JOB_ID}" "${JOB_DIR}/5_visualize_flowsdf.job")
echo "Submitted visualization job: ${VIS_JOB_ID}"

echo ""
echo "FlowSDF workflow submitted:"
echo "  inference     ${INFER_JOB_ID}"
echo "  evaluation    ${EVAL_JOB_ID}"
echo "  visualization ${VIS_JOB_ID}"
