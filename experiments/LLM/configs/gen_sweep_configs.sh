#!/bin/bash
# Configuration list for the perplexity sweeps of Section 6: the global,
# component, sublayer, cumulative and blockwise figures, and Supplementary
# Fig. S1 and Tables S1-S2.
#
#   gen_sweep_configs.sh [paper|rn-replication] > sweeps.txt
#
# paper           five SR seeds and one RN run of every swept point, plus the
#                 t=24 RN reference the figures draw as a dotted line
#                 (811 runs)
# rn-replication  four more RN runs of every point (seeds 2-5), which check
#                 that RN is bitwise reproducible (540 runs)
#
# Every run scores the first 1024 tokens of the WikiText-2 test slice in four
# 256-token contexts. The target runs at precision t under the given rounding
# mode; everything else runs at t=24 under RN, so VFC_BACKENDS always starts in
# RN and the harness switches the target. Logs go under
# $RESULTS_ROOT/256/<level directory>/, the layout scripts/make_figures.py reads
# (--logs $RESULTS_ROOT/256); the directory names name the protocol, which
# scripts/parse_sweep_logs.py maps to the arm each figure draws.
set -euo pipefail

set_name=${1:-paper}
results_root=${RESULTS_ROOT:-${FUZZY_RUNS:-$HOME/fuzzy-llm-runs}/perplexity_logs}
context=256
max_tokens=1024
groups="attention mlp lm_head attn_qkav"
sub_layers="attn_c_attn attn_c_proj mlp_c_fc mlp_c_proj"
prefixes="0 0-1 0-2 0-3 0-4 0-5"
blocks="0 1 2 3 4 5"
budget="--context_length $context --max_tokens $max_tokens"

emit() {  # log file, seed, python arguments...
    local log=$1 seed=$2
    shift 2
    printf '%s|libinterflop_prism.so --seed=%s --mode=rn|%s\n' "$log" "$seed" "$*"
}

emit_all_points() {  # mode seed
    local mode=$1 seed=$2 d t group layer prefix block
    local modes="--target_mode $mode --default_mode rn"

    d=$results_root/$context/global_fixed_$mode
    for t in $(seq 4 14); do
        emit "$d/prec_${t}_seed_$seed.log" "$seed" \
            test_global_perplexity.py --precision "$t" $modes $budget
    done

    d=$results_root/$context/scoped_ppl_percomponents_${mode}_rndefault
    for group in $groups; do
        for t in $(seq 4 14); do
            emit "$d/group_${group}_prec_${t}_seed_$seed.log" "$seed" \
                test_percomponent_perplexity.py --group "$group" --precision "$t" $modes $budget
        done
    done

    d=$results_root/$context/scoped_fine_${mode}_rndefault_block_all
    for layer in $sub_layers; do
        for t in $(seq 4 14); do
            emit "$d/layer_${layer}_prec_${t}_seed_$seed.log" "$seed" \
                test_fine_perplexity.py --layer "$layer" --precision "$t" --block_idx all $modes $budget
        done
    done

    d=$results_root/$context/scoped_cumulative_${mode}_rndefault
    for prefix in $prefixes; do
        for t in 4 5 6; do
            emit "$d/blocks_${prefix//-/_}_layer_mlp_c_proj_prec_${t}_seed_$seed.log" "$seed" \
                test_fine_perplexity.py --layer mlp_c_proj --precision "$t" --block_idx "$prefix" $modes $budget
        done
    done

    d=$results_root/$context/scoped_blockwise_${mode}_rndefault
    for block in $blocks; do
        for t in 4 5 6; do
            emit "$d/block_${block}_layer_mlp_c_proj_prec_${t}_seed_$seed.log" "$seed" \
                test_fine_perplexity.py --layer mlp_c_proj --precision "$t" --block_idx "$block" $modes $budget
        done
    done
}

case $set_name in
paper)
    for seed in 1 2 3 4 5; do emit_all_points sr "$seed"; done
    emit_all_points rn 1
    # The full-precision reference, PPL_0 = 62.51.
    emit "$results_root/$context/scoped_fine_rn_rndefault_block_all/layer_attn_c_attn_prec_24_seed_1.log" 1 \
        test_fine_perplexity.py --layer attn_c_attn --precision 24 --block_idx all \
        --target_mode rn --default_mode rn $budget
    ;;
rn-replication)
    for seed in 2 3 4 5; do emit_all_points rn "$seed"; done
    ;;
*)
    echo "unknown set: $set_name (paper or rn-replication)" >&2
    exit 2
    ;;
esac
