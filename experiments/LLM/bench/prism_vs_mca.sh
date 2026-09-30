#!/bin/bash
# Time elementwise add, mul, div and fma (binary32 and binary64, 512-bit
# vectors) natively, under PRISM SR and under Verificarlo MCA quad RR.
#
#   prism_vs_mca.sh build <dir>       compile the three binaries into <dir>
#   prism_vs_mca.sh run <dir> <k>     time run <k>, written to <dir>/run_<k>.log
#
# Runs inside verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx512 on an AVX-512 CPU.
# Each run times all three binaries at full precision (t=24/53) and at reduced
# precision (t=8/24). scripts/prism_bench_table.py turns the run logs into
# tab:prism-vector-perf.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cmd=${1:?usage: $0 build|run <dir> [k]}
dir=${2:?output directory}
export VFC_BACKENDS_LOGGER=False
flags="-O3 -march=skylake-avx512 -mprefer-vector-width=512 -ffp-contract=off"

header() {
    printf 'Date: %s\nHost: %s\n' "$(date -u +%FT%TZ)" "$(hostname)"
    lscpu | grep -E 'Model name'
}

case $cmd in
build)
    mkdir -p "$dir"
    cd "$dir"
    exec >build.log 2>&1
    header
    clang++ --version | head -1
    verificarlo --version | head -1
    cp "$here/prism_vs_mcaquad_bench.cpp" bench.cpp
    clang++ $flags bench.cpp -o native
    verificarlo-c++ --inst-fma --prism-backend=sr --prism-backend-dispatch=static $flags bench.cpp -o prism
    verificarlo-c++ --inst-fma $flags bench.cpp -o mca
    # Every kernel, vector fma included, must call into its backend.
    prism_calls=$(nm -D --undefined-only prism | c++filt |
        grep -oE 'prism::sr::vector::[a-z_:]+::(add|mul|div|fma)f(32x16|64x8)' | sort -u)
    mca_calls=$(nm -D --undefined-only mca |
        grep -oE '_(16xfloat|8xdouble)(add|mul|div|fma)_avx512' | sort -u)
    printf 'PRISM calls:\n%s\nMCA calls:\n%s\n' "$prism_calls" "$mca_calls"
    [ "$(wc -l <<<"$prism_calls")" = 8 ] || { echo "ERROR: PRISM does not instrument all 8 kernels"; exit 1; }
    [ "$(wc -l <<<"$mca_calls")" = 8 ] || { echo "ERROR: MCA does not instrument all 8 kernels"; exit 1; }
    echo BUILD_OK
    ;;
run)
    k=${3:?run index}
    cd "$dir"
    exec >"run_$k.log" 2>&1
    header
    for p in "24 53" "8 24"; do
        set -- $p
        echo "=== run $k native p=$1/$2 ==="
        ./native 5000
        echo "=== run $k prism p=$1/$2 ==="
        VFC_BACKENDS="libinterflop_prism.so --precision-binary32=$1 --precision-binary64=$2 --seed=$k" ./prism 100
        echo "=== run $k mca p=$1/$2 ==="
        VFC_BACKENDS="libinterflop_mca.so --mode=rr --precision-binary32=$1 --precision-binary64=$2 --seed=$k" ./mca 30
    done
    echo RUN_OK
    ;;
*)
    echo "usage: $0 build|run <dir> [k]" >&2
    exit 2
    ;;
esac
