#!/usr/bin/env bash
# Every accuracy check for the report, in one go (a few minutes on a laptop).
# usage: tests/validate.sh [build_dir]
set -euo pipefail
BUILD=${1:-build}
OUT=${OUT:-out/validate}
PY="python3 python"
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

echo "== 1. d = 1 against the paper's closed form (Proposition 2), paper parameters"
mkdir -p "$OUT/d1"; "$BUILD/hjb_serial" --d 1 --Q 30 --snapshots 60 --out "$OUT/d1" > /dev/null
$PY/check_1d.py "$OUT/d1"

echo "== 2. convergence in the time step (expect error halving with cfl: first order)"
for c in 0.8 0.4 0.2 0.1 0.05; do
    mkdir -p "$OUT/cfl$c"; "$BUILD/hjb_serial" --d 1 --Q 30 --snapshots 10 --cfl $c --out "$OUT/cfl$c" > /dev/null
    echo -n "cfl $c: "; $PY/check_1d.py "$OUT/cfl$c"
done

echo "== 3. d = 2, 3 against exact answers"
mkdir -p "$OUT/sep2" "$OUT/sep3" "$OUT/lin2"
"$BUILD/hjb_serial" --d 2 --Q 30 --k 0.3,0.5 --A 0.9,0.6 --sigma 0.3,0.4 --snapshots 4 --out "$OUT/sep2" > /dev/null
$PY/check_nd.py "$OUT/sep2" separable
"$BUILD/hjb_serial" --d 3 --Q 10 --k 0.3,0.5,0.4 --sigma 0.3,0.4,0.2 --snapshots 4 --out "$OUT/sep3" > /dev/null
$PY/check_nd.py "$OUT/sep3" separable
"$BUILD/hjb_serial" --d 2 --Q 20 --rho 0.6 --A 0.9,0.6 --sigma 0.3,0.4 --snapshots 4 --out "$OUT/lin2" > /dev/null
$PY/check_nd.py "$OUT/lin2" linear

echo "== 4. parallel == serial, byte for byte"
tests/compare.sh "$BUILD" "${MPIRUN:-mpirun --oversubscribe}"
