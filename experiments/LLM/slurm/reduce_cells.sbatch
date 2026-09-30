#!/bin/bash
#SBATCH --job-name=reduce
#SBATCH --array=0-5
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=reduce-%A_%a.out
# One array task per (site, t) cell of a reduction over the 32-seed captures:
# lm_head and mlp_c_proj at t=6, 7, 8. The reductions are NumPy only and
# memory-bandwidth bound, so each task takes a whole node and runs on the
# host's python3 (PYTHON overrides it). Submit from experiments/LLM with
# HEAD_DECOMP_ROOT=$CAPTURE_ROOT/256/tok1024 in the environment:
#
#   sbatch slurm/reduce_cells.sbatch logit_drift_map.py 32 --out <dir>
#   sbatch slurm/reduce_cells.sbatch logit_fluct_map.py 32 --drift-dir <logit_drift dir> --out <dir>
#   sbatch slurm/reduce_cells.sbatch full_grid_drift_boxes.py --seeds 32 --out <dir>
#   sbatch slurm/reduce_cells.sbatch token_fisher_analysis.py 32 --out <dir>
#
# logit_fluct_map.py needs the logit_drift_map.py bundles, so run that first.
set -euo pipefail
cells=("lm_head 6" "lm_head 7" "lm_head 8" "mlp_c_proj 6" "mlp_c_proj 7" "mlp_c_proj 8")
read -r site t <<<"${cells[$SLURM_ARRAY_TASK_ID]}"
script=${1:?reduction script}
shift
exec "${PYTHON:-python3}" "$SLURM_SUBMIT_DIR/$script" "$site" "$t" "$@"
