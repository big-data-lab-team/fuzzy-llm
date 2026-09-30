#!/bin/bash
#SBATCH --job-name=prism-vs-mca
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
# One timing run of bench/prism_vs_mca.sh per array task. --exclusive gives
# each run its own node, so concurrent runs share no cache, memory bandwidth or
# AVX-512 frequency licence. Submit from experiments/LLM:
#
#   apptainer build ~/sif/fuzzy-v2.6.0-avx512.sif docker://verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx512
#   apptainer exec ~/sif/fuzzy-v2.6.0-avx512.sif bench/prism_vs_mca.sh build <dir>
#   sbatch --array=1-30 slurm/prism_vs_mca.sbatch <dir>
#
# Restrict the partition or nodes (-p, --nodelist, --exclude) to one CPU model
# so all runs time the same hardware; each log records the model it ran on.
set -euo pipefail
dir=$(realpath "${1:?output directory from the build step}")
sif=${SIF:-$HOME/sif/fuzzy-v2.6.0-avx512.sif}
bench=$SLURM_SUBMIT_DIR/bench
apptainer exec --containall --bind "$bench:/bench:ro" --bind "$dir:/out" "$sif" \
    bash /bench/prism_vs_mca.sh run /out "$SLURM_ARRAY_TASK_ID"
