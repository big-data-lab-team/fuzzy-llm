#!/bin/bash
# Configuration list for the logit captures behind the loss decomposition
#
#   E_m[dL] = D_m + Q_m + V_m + R_m          (eq:head-full-comparison)
#
# of tab:decomposition, fig:drift-distribution and fig:head-decomposition-terms.
#
#   gen_capture_configs.sh <stage> > captures-<stage>.txt
#
# Each run stores the per-token logit perturbation Delta = lambda - lambda_ref
# of one (site, t, mode, seed) cell under $CAPTURE_ROOT/256/tok1024/, the root
# that decompose.py, distributions.py, drift_study.py and the logit_*_map.py
# reductions read. The cells take hundreds of megabytes each; allow 8 GB of
# memory per run. Run the stages in order:
#
#   ref    the t=24 RN reference every Delta is taken against, plus a second
#          RN seed and an IEEE run; `decompose.py refcheck` gates on them (3)
#   null   no site lowered: the t=24 background every cell carries (9)
#   head   lm_head at t=4..14, SR_SEEDS seeds and one RN run (99)
#   sites  the other sites of SITES at t=4..14 (495)
#
# The paper uses eight SR seeds everywhere and 32 at lm_head and mlp_c_proj for
# t=4..10. Seeds 9-32 are added by rerunning with a larger pool; cells already
# captured are skipped:
#
#   SR_SEEDS="$(seq 1 32)" T_MAX=10 gen_capture_configs.sh head
#   SR_SEEDS="$(seq 1 32)" T_MAX=10 SITES=mlp_c_proj gen_capture_configs.sh sites
#
# SR cells run their untouched operations at t=24 under SR, RN cells under RN,
# so the null stage measures the background the SR cells include.
set -u

STAGE=${1:?usage: $0 ref|null|head|sites}
CAPTURE_ROOT=${CAPTURE_ROOT:-${FUZZY_RUNS:-$HOME/fuzzy-llm-runs}/captures}
LOG_ROOT=${LOG_ROOT:-$CAPTURE_ROOT/logs}
CONTEXT=256
TOKENS=1024
T_MIN=${T_MIN:-4}
T_MAX=${T_MAX:-14}
SR_SEEDS=${SR_SEEDS:-1 2 3 4 5 6 7 8}
NULL_SEEDS=${NULL_SEEDS:-1 2 3 4 5 6 7 8}
SITES=${SITES:-attn_c_attn attn_qkav attn_c_proj mlp_c_fc mlp_c_proj}

root=$CAPTURE_ROOT/$CONTEXT/tok$TOKENS
logs=$LOG_ROOT/$CONTEXT/tok$TOKENS
budget="--context_length $CONTEXT --max_tokens $TOKENS"

backend() {  # mode seed
    if [ "$1" = rn ]; then
        echo "libinterflop_prism.so --seed=$2 --mode=rn"
    else
        echo "libinterflop_prism.so --seed=$2"
    fi
}

emit() {  # log file, backend, python arguments...
    local log=$1 backend=$2
    shift 2
    echo "$log|$backend|$*"
}

emit_cell() {  # site precision mode seed
    local rel
    rel=$(printf '%s/t%02d/%s/seed%s' "$1" "$2" "$3" "$4")
    emit "$logs/${rel//\//_}.log" "$(backend "$3" "$4")" \
        capture_logits.py --site "$1" --precision "$2" \
        --out-dir "$root/$rel" --ref "$root/reference" $budget
}

emit_site() {  # site
    local t seed
    for t in $(seq "$T_MIN" "$T_MAX"); do
        for seed in $SR_SEEDS; do emit_cell "$1" "$t" sr "$seed"; done
        emit_cell "$1" "$t" rn 1
    done
}

case $STAGE in
ref)
    # The reference is PRISM at t=24 under RN: the arithmetic every cell's
    # untouched operations perform, and deterministic, so a fixed origin. PRISM
    # RN under a second seed must match it bit for bit; the IEEE run is
    # advisory, since PRISM's rounding path is not IEEE's.
    emit "$logs/reference.log" "$(backend rn 1)" \
        capture_logits.py --site none --precision 24 --out-dir "$root/reference" $budget
    emit "$logs/reference_check_rn_seed2.log" "$(backend rn 2)" \
        capture_logits.py --site none --precision 24 --out-dir "$root/reference_check/rn_seed2" $budget
    emit "$logs/reference_check_ieee.log" "libinterflop_ieee.so" \
        capture_logits.py --site none --precision 24 --out-dir "$root/reference_check/ieee" $budget
    ;;
null)
    for seed in $NULL_SEEDS; do
        emit "$logs/null_t24_sr_seed$seed.log" "$(backend sr "$seed")" \
            capture_logits.py --site none --precision 24 \
            --out-dir "$root/null/t24/sr/seed$seed" --ref "$root/reference" $budget
    done
    emit "$logs/null_t24_rn_seed1.log" "$(backend rn 1)" \
        capture_logits.py --site none --precision 24 \
        --out-dir "$root/null/t24/rn/seed1" --ref "$root/reference" $budget
    ;;
head)
    emit_site lm_head
    ;;
sites)
    for site in $SITES; do emit_site "$site"; done
    ;;
*)
    echo "unknown stage: $STAGE (ref, null, head or sites)" >&2
    exit 2
    ;;
esac
