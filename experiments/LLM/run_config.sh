#!/bin/bash
# Run one line of a configuration list in the fuzzy PyTorch image.
#
#   run_config.sh <config file> <line number>
#
# Each line of a list written by configs/gen_*.sh is
#
#   <log file>|<VFC_BACKENDS string>|<python arguments>
#
# and runs as one single-threaded, offline container. The image supplies the
# instrumented PyTorch; this directory (the harness), ../python (fuzzy_torch)
# and the DistilGPT-2 cache written by fetch_model.sh are bound into it.
#
#   ENGINE    apptainer (default), podman or docker
#   IMAGE     a .sif for apptainer (default ~/sif/fuzzy-v2.6.0-avx2.sif), an
#             image name otherwise (default the avx2 image below)
#   HF_CACHE  the model cache (default $FUZZY_RUNS/hf_cache)
#
# A line whose log already holds a "Result ->" line is skipped, so a list can be
# resubmitted to resume. A run that ends without one keeps its output as
# <log>.failed and leaves no partial capture arrays, since either would
# otherwise be mistaken for a result.
set -u
here=$(cd "$(dirname "$0")" && pwd)
config=${1:?usage: run_config.sh <config file> <line number>}
n=${2:?usage: run_config.sh <config file> <line number>}
engine=${ENGINE:-apptainer}
if [ "$engine" = apptainer ]; then
    image=${IMAGE:-$HOME/sif/fuzzy-v2.6.0-avx2.sif}
else
    image=${IMAGE:-docker.io/verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx2}
fi
hf_cache=${HF_CACHE:-${FUZZY_RUNS:-$HOME/fuzzy-llm-runs}/hf_cache}

line=$(sed -n "${n}p" "$config")
[ -n "$line" ] || { echo "no configuration at line $n of $config" >&2; exit 1; }
log=${line%%|*}
rest=${line#*|}
backend=${rest%%|*}
args=${rest#*|}

grep -q "Result ->" "$log" 2>/dev/null && exit 0
mkdir -p "$(dirname "$log")"

binds=("$here:/workspace:ro" "$here/../python:/pkg:ro" "$hf_cache:/hf_cache:ro")
# Capture runs write arrays under --out-dir and read the reference capture from
# --ref: bind both at the same path so the paths in the list need no
# translation.
out_dir=$(sed -n 's/.*--out-dir \([^ ]*\).*/\1/p' <<<"$args")
ref_dir=$(sed -n 's/.*--ref \([^ ]*\).*/\1/p' <<<"$args")
if [ -n "$out_dir" ]; then
    mkdir -p "$out_dir"
    binds+=("$out_dir:$out_dir")
fi
[ -n "$ref_dir" ] && binds+=("$ref_dir:$ref_dir:ro")
env=(PYTHONPATH=/workspace:/pkg VFC_BACKENDS_LOGGER=False "VFC_BACKENDS=$backend"
     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
     HF_HOME=/hf_cache HF_HUB_OFFLINE=1)

case $engine in
apptainer)
    opts=(--containall --pwd /workspace)
    for b in "${binds[@]}"; do opts+=(--bind "$b"); done
    for e in "${env[@]}"; do opts+=(--env "$e"); done
    apptainer exec "${opts[@]}" "$image" python3 -u $args >"$log" 2>&1
    ;;
podman | docker)
    opts=(--rm --network=none --cpus=1 -w /workspace)
    [ "$engine" = docker ] && opts+=(-u "$(id -u):$(id -g)")
    for b in "${binds[@]}"; do opts+=(-v "$b"); done
    for e in "${env[@]}"; do opts+=(-e "$e"); done
    $engine run "${opts[@]}" "$image" python3 -u $args >"$log" 2>&1
    ;;
*)
    echo "unknown ENGINE=$engine (apptainer, podman or docker)" >&2
    exit 2
    ;;
esac
status=$?

if ! grep -q "Result ->" "$log" 2>/dev/null; then
    [ -n "$out_dir" ] && rm -f "$out_dir"/{delta.npy,logits.npy,scalars.npz,meta.json}
    mv -f "$log" "$log.failed"
    exit $((status ? status : 1))
fi
