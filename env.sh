#!/bin/bash
# Single source of truth for account-specific paths used by every job script
# in this repo. Source this near the top of any .job/.sh script:
#
#     source "$HOME/projects/MoE-ShapeRefine-MedicalSeg/env.sh"
#
# Nothing here is hardcoded to a specific Snellius account — PROJECT_ROOT and
# SCRATCH_DATA_ROOT are derived from $HOME/$USER, which SLURM sets correctly
# for whichever account submits the job. If this project is ever migrated to
# a new account again, no changes are needed here or in any script that
# sources this file.

export PROJECT_ROOT="$HOME/projects/MoE-ShapeRefine-MedicalSeg"
export MEDSAM_ROOT="$HOME/projects/MedSAM"
export TOPLEVEL_DATA_ROOT="$HOME/projects/data"

export SCRATCH_DATA_ROOT="/scratch-shared/$USER"
export SCRATCH_DATA="$SCRATCH_DATA_ROOT/data"
export SCRATCH_CHECKPOINTS="$SCRATCH_DATA_ROOT/checkpoints"
