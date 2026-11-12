#!/usr/bin/env bash
# Parallel == serial test: every MPI variant must write byte-identical snapshots.
# usage: tests/compare.sh [build_dir] [mpirun command]
set -euo pipefail
BUILD=${1:-build}
MPIRUN=${2:-"mpirun --oversubscribe"}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
fail=0

check() {  # name, ranks, threads, solver args...
    local name=$1 np=$2 nt=$3; shift 3
    mkdir -p "$OUT/$name"
    OMP_NUM_THREADS=$nt $MPIRUN -np "$np" "$BUILD/hjb_mpi" "$@" --out "$OUT/$name" > /dev/null
    for f in "$OUT/ref"/theta_*.bin; do
        if ! cmp -s "$f" "$OUT/$name/$(basename "$f")"; then
            echo "FAIL $name ($(basename "$f") differs)"; fail=1; return
        fi
    done
    echo "ok   $name"
}

for case in "--d 1 --Q 40" "--d 2 --Q 25 --rho 0.5 --k 0.3,0.4" "--d 3 --Q 8 --rho 0.3 --sigma 0.3,0.4,0.2"; do
    echo "== $case"
    rm -rf "$OUT"/*; mkdir -p "$OUT/ref"
    OMP_NUM_THREADS=1 "$BUILD/hjb_serial" $case --snapshots 3 --out "$OUT/ref" > /dev/null
    for np in 1 2 3 4 6 8; do
        for decomp in block slab; do
            for halo in blocking overlap; do
                check "np$np-$decomp-$halo" $np 1 $case --snapshots 3 --decomp $decomp --halo $halo
            done
        done
    done
    check "np2-block-overlap-omp3" 2 3 $case --snapshots 3 --halo overlap
done
[ $fail = 0 ] && echo "ALL PASSED" || { echo "SOME FAILED"; exit 1; }
