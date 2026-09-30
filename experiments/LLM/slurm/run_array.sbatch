#!/bin/bash
#SBATCH --job-name=fuzzy-llm
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
# One array task per line of a configuration list, run by run_config.sh with
# Apptainer. Submit from experiments/LLM:
#
#   sbatch --array=1-$(wc -l < <list>) slurm/run_array.sbatch <list>
#
# Add %N to the array range to cap concurrent tasks, and --mem=8G for the
# capture lists. IMAGE overrides the default ~/sif/fuzzy-llm-experiments.sif.
# One CPU per task on purpose: a run on eight threads is only about 7% faster
# than on one, so the cores are better spent on more concurrent runs.
exec bash "$SLURM_SUBMIT_DIR/run_config.sh" "$(realpath "${1:?config file}")" "$SLURM_ARRAY_TASK_ID"
