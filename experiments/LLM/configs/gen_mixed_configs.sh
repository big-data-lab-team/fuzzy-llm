#!/bin/bash
# Configuration list for tab:accuracy-matched-precision: mixed rounding against
# all-RN at identical per-site precisions.
#
#   gen_mixed_configs.sh > mixed.txt          (19 runs)
#
# For each MLP down-projection precision k in {8, 7, 6}, the mixed recipe (SR at
# the two output projections, RN elsewhere) with SR seeds 101-105, and the
# all-RN control at the same precisions; plus the t=24 reference. Logs go to
# $MIXED_ROOT/<run>.log, named as in results/mixed_followup_20260915/outcomes/,
# and scripts/mixed_table.py builds the table from either.
set -euo pipefail

root=${MIXED_ROOT:-${FUZZY_RUNS:-$HOME/fuzzy-llm-runs}/mixed_logs}
budget="--context_length 256 --max_tokens 1024 --default_mode rn"

sites() {  # k, mode at the output projections
    echo "attn_c_attn=8:rn,attn_c_proj=6:$2,mlp_c_fc=8:rn,mlp_c_proj=$1:$2,lm_head=11:rn"
}

emit() {  # run name, seed, site assignment
    printf '%s/%s.log|libinterflop_prism.so --seed=%s --mode=rn|test_mixed_perplexity.py --sites %s %s\n' \
        "$root" "$1" "$2" "$3" "$budget"
}

emit reference 1 "attn_c_attn=24:rn,attn_c_proj=24:rn,mlp_c_fc=24:rn,mlp_c_proj=24:rn,lm_head=24:rn"
for k in 8 7 6; do
    emit "mlp${k}_rn" 1 "$(sites "$k" rn)"
    for seed in 101 102 103 104 105; do
        emit "mlp${k}_mixed_seed$seed" "$seed" "$(sites "$k" sr)"
    done
done
