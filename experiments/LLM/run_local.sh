#!/bin/bash
# Run every line of a configuration list on this machine, <jobs> at a time.
#
#   ENGINE=podman run_local.sh <config file> [jobs]
#
# Each line is one single-threaded container (see run_config.sh), so <jobs>
# defaults to the number of cores. Allow 3 GB of memory per job, 8 GB for the
# capture lists. Rerunning resumes: finished lines are skipped.
set -u
here=$(cd "$(dirname "$0")" && pwd)
config=${1:?usage: run_local.sh <config file> [jobs]}
jobs=${2:-$(nproc)}
n=$(wc -l <"$config")
seq 1 "$n" | xargs -P "$jobs" -I{} bash "$here/run_config.sh" "$config" {}
failed=$(cut -d'|' -f1 "$config" | sed 's/$/.failed/' | xargs -r ls 2>/dev/null | wc -l)
echo "$n configurations, $failed failed (see *.log.failed)"
