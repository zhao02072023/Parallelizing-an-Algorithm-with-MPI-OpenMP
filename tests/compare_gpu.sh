#!/usr/bin/env bash
# GPU build vs serial CPU build. On a real GPU the device exp() differs from the host's
# in the last bits, so this compares with a relative tolerance instead of byte for byte.
# usage: tests/compare_gpu.sh [cpu_build_dir] [gpu_build_dir] [mpirun command]
set -euo pipefail
CPU=${1:-build}
GPU=${2:-build-gpu}
MPIRUN=${3:-mpirun}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
fail=0
for case in "--d 1 --Q 30" "--d 2 --Q 60 --rho 0.5 --k 0.3,0.4" "--d 3 --Q 12 --rho 0.3 --sigma 0.3,0.4,0.2"; do
    echo "== $case"
    rm -rf "$OUT"/*; mkdir -p "$OUT/ref"
    OMP_NUM_THREADS=1 "$CPU/hjb_serial" $case --snapshots 3 --out "$OUT/ref" > /dev/null
    for np in ${GPU_RANKS:-1 2}; do
        mkdir -p "$OUT/np$np"
        OMP_NUM_THREADS=1 $MPIRUN -n "$np" "$GPU/hjb_gpu" $case --snapshots 3 --out "$OUT/np$np" | grep '^#' | head -1
        echo -n "np $np: "; python3 python/compare.py "$OUT/ref" "$OUT/np$np" 1e-9 || fail=1
    done
done
[ $fail = 0 ] && echo "ALL PASSED" || { echo "SOME FAILED"; exit 1; }
