#!/bin/bash
# Download the DistilGPT-2 weights and tokenizer once, into the cache that
# run_config.sh binds read-only into every (offline) run. Needs network access,
# so on a cluster run it on a login node. ENGINE, IMAGE and HF_CACHE as in
# run_config.sh.
set -euo pipefail
engine=${ENGINE:-apptainer}
if [ "$engine" = apptainer ]; then
    image=${IMAGE:-$HOME/sif/fuzzy-v2.6.0-avx2.sif}
else
    image=${IMAGE:-docker.io/verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx2}
fi
hf_cache=${HF_CACHE:-${FUZZY_RUNS:-$HOME/fuzzy-llm-runs}/hf_cache}
mkdir -p "$hf_cache"
fetch="from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')
print('distilgpt2 cached')"
case $engine in
apptainer)
    apptainer exec --containall --bind "$hf_cache:/hf_cache" --env HF_HOME=/hf_cache \
        --env VFC_BACKENDS_LOGGER=False "$image" python3 -c "$fetch" ;;
podman | docker)
    opts=(--rm -v "$hf_cache:/hf_cache" -e HF_HOME=/hf_cache -e VFC_BACKENDS_LOGGER=False)
    [ "$engine" = docker ] && opts+=(-u "$(id -u):$(id -g)")
    $engine run "${opts[@]}" "$image" python3 -c "$fetch" ;;
*)
    echo "unknown ENGINE=$engine (apptainer, podman or docker)" >&2
    exit 2 ;;
esac
