#!/usr/bin/env bash
# Scheduler-agnostic benchmark driver. Run it inside a job allocation:
#   jobs/bench.sh <experiment> [results.csv]
# experiments: serial | serialfull | strong | strong3d | weak | hybrid | variants | binding
#              gpu_strong | gpu_weak | gpu_vs_cpu   (GPU nodes, BIN_GPU built with -DGPU=ON)
#
# Defaults fit Vanda CPU nodes (2 sockets x 36 cores). Environment:
#   NODES           nodes in the allocation                     (default 1)
#   CORES_PER_NODE  physical cores per node                     (default 72)
#   CORES_PER_SOCKET                                            (default CORES_PER_NODE / 2)
#   CORE_COUNTS     total core counts to sweep (default 1 2 4 9 18 36 72 144 ... up to the allocation)
#   THREADS         OpenMP threads per rank in hybrid runs      (default CORES_PER_SOCKET)
#   REPEATS         runs per configuration                      (default 3)
#   STEPS           time steps per run                          (default 200)
#   MPIRUN          launcher                                    (default mpirun)
#   MPI_FLAVOR      openmpi | intel | other       (default: detected from mpirun --version)
#   BIND            launcher flags, overriding the detected ones; $1 = ranks, $2 = threads,
#                   $3 = ranks per node
set -euo pipefail
EXP=${1:?experiment}
CSV=${2:-results/raw/${EXP}.csv}
HERE=$(cd "$(dirname "$0")/.." && pwd)      # root directory
BIN=${BIN:-$HERE/build/hjb_mpi}             # cpu parallel program
SERIAL=${SERIAL:-$HERE/build/hjb_serial}    # cpu sequential program
BIN_GPU=${BIN_GPU:-$HERE/build-gpu/hjb_gpu} # gpu accelerated program
GPN=${GPUS_PER_NODE:-2}
NODES=${NODES:-1}
CPN=${CORES_PER_NODE:-72}
CPS=${CORES_PER_SOCKET:-$((CPN / 2))}
THREADS=${THREADS:-$CPS} # OpenMP threads spawned per MPI rank i.e. half of cores
REPEATS=${REPEATS:-3}    # repeatability of results
STEPS=${STEPS:-200}      # 200 time steps in each program
MPIRUN=${MPIRUN:-mpirun}
MAXC=$((NODES * CPN))    # maximum hardware boundary
if [ -z "${MPI_FLAVOR:-}" ]; then
    v=$($MPIRUN --version 2>&1 || true)
    case "$v" in
        *"Open MPI"*|*"OpenRTE"*) MPI_FLAVOR=openmpi ;;
        *Intel*) MPI_FLAVOR=intel ;;
        *) MPI_FLAVOR=other ;;
    esac
fi
mkdir -p "$(dirname "$CSV")"
[ -s "$CSV" ] || echo "experiment,variant,ranks,threads,d,Q,points,steps,total_s,compute_s,halo_s,reduce_s,io_s,t_final,halo_msgs,halo_bytes,copy_s" > "$CSV"

# benchmark model: 2 correlated assets, gamma small enough that the explicit
# step stays reasonable at Q = 1000 (see README, "Stiffness")
MODEL="--gamma 1e-4 --rho 0.5 --A 0.9,0.7 --sigma 0.3,0.4 --k 0.3,0.35"

if [ -z "${CORE_COUNTS:-}" ]; then
    CORE_COUNTS=""
    for c in 1 2 4 9 18 36 72; do [ $c -le $MAXC ] && CORE_COUNTS+="$c "; done
    c=$((2 * CPN)); while [ $c -le $MAXC ]; do CORE_COUNTS+="$c "; c=$((2 * c)); done
    [ $MAXC -gt $CPN ] && [[ " $CORE_COUNTS " != *" $MAXC "* ]] && CORE_COUNTS+="$MAXC"
fi

launch_flags() {  # ranks threads -> flags that fill nodes one at a time, ranks pinned to their cores
    local np=$1 nt=$2 rpn=$((CPN / $2))
    [ $rpn -gt $np ] && rpn=$np
    if [ -n "${BIND:-}" ]; then set -- "$np" "$nt" "$rpn"; eval "echo $BIND"; return; fi
    case $MPI_FLAVOR in
        openmpi) echo "--map-by ppr:${rpn}:node:PE=${nt} --bind-to core ${REPORT:+--report-bindings}" ;;
        intel)   echo "-ppn ${rpn} -genv I_MPI_PIN_DOMAIN=${nt}:compact -genv I_MPI_PIN_ORDER=compact ${REPORT:+-genv I_MPI_DEBUG=4}" ;;
        *)       echo "" ;;
    esac
}

run() {  # ranks threads extra-args...
    local np=$1 nt=$2; shift 2
    for r in $(seq "$REPEATS"); do
        line=$(OMP_NUM_THREADS=$nt OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=cores \
               $MPIRUN -n "$np" $(launch_flags "$np" "$nt") "$BIN" $MODEL --max-steps "$STEPS" "$@" \
               | grep '^RESULT' | tail -1 | cut -d, -f2-)
        echo "$EXP,$line" | tee -a "$CSV"
    done
}

run_serial() {  # extra-args...  (one core, no MPI)
    for r in $(seq "$REPEATS"); do
        line=$(OMP_NUM_THREADS=1 OMP_PROC_BIND=true OMP_PLACES=cores "$SERIAL" $MODEL "$@" \
               | grep '^RESULT' | tail -1 | cut -d, -f2-)
        echo "$EXP,$line" | tee -a "$CSV"
    done
}

run_gpu() {  # ranks extra-args...  one rank per GPU, each pinned to its own socket
    local np=$1; shift
    for r in $(seq "$REPEATS"); do
        line=$(OMP_TARGET_OFFLOAD=MANDATORY OMP_NUM_THREADS=1 \
               $MPIRUN -n "$np" $(launch_flags "$np" $((CPN / GPN))) "$BIN_GPU" $MODEL --max-steps "$STEPS" "$@" \
               | grep '^RESULT' | tail -1 | cut -d, -f2-)
        echo "$EXP,$line" | tee -a "$CSV"
    done
}

gpu_counts() { local g=1; while [ $g -le $((NODES * GPN)) ]; do echo $g; g=$((2 * g)); done; }

hybrid_ok() { [ $(($1 % THREADS)) = 0 ] && [ $1 -ge "$THREADS" ]; }

case "$EXP" in
serial)    # cost per step vs problem size: lattice points (like matrix dimension) and stocks d,
           # which sets the 2d neighbours per point (like nonzeros per row)
    for Q in 1000 10000 100000 1000000; do run_serial --d 1 --Q $Q --max-steps "$STEPS"; done
    for Q in 50 100 200 400 800 1600; do run_serial --d 2 --Q $Q --max-steps "$STEPS"; done
    for Q in 10 20 40 80 160; do run_serial --d 3 --Q $Q --max-steps "$STEPS"; done ;;
serialfull) # whole horizon: the number of stable steps also grows with Q (stiffness)
    for g in 0.01 1e-4; do
        for Q in 30 100 300 1000; do run_serial --d 1 --Q $Q --gamma $g; done
        for Q in 15 30 60 120; do run_serial --d 2 --Q $Q --gamma $g; done
    done ;;
binding)   # one short hybrid run that prints where every rank is pinned; check it before the sweeps
    REPORT=1 REPEATS=1 STEPS=5 run $((MAXC / THREADS)) "$THREADS" --d 2 --Q 1000 ;;
strong)    # fixed 4.0 M points, pure MPI and MPI x THREADS
    for c in $CORE_COUNTS; do
        run "$c" 1 --d 2 --Q 1000
        hybrid_ok "$c" && run $((c / THREADS)) "$THREADS" --d 2 --Q 1000
    done ;;
strong3d)  # fixed 8.1 M points in 3D: six halo faces per block instead of four
    for c in $CORE_COUNTS; do
        run "$c" 1 --d 3 --Q 100
        hybrid_ok "$c" && run $((c / THREADS)) "$THREADS" --d 3 --Q 100
    done ;;
weak)      # about 250k points per core
    for c in $CORE_COUNTS; do
        Q=$(awk -v c="$c" 'BEGIN { printf "%d", (sqrt(250000 * c) - 1) / 2 + 0.5 }')
        run "$c" 1 --d 2 --Q "$Q"
        hybrid_ok "$c" && run $((c / THREADS)) "$THREADS" --d 2 --Q "$Q"
    done ;;
hybrid)    # every ranks x threads split of the whole allocation
    for nt in 1 2 3 4 6 9 12 18 36 72; do
        [ $nt -le $CPN ] && [ $((CPN % nt)) = 0 ] && run $((MAXC / nt)) "$nt" --d 2 --Q 1000
    done ;;
variants)  # decomposition x halo mode, pure MPI, plus fixed dt (no allreduce)
    for c in $CORE_COUNTS; do
        [ $c -lt 2 ] && continue
        for decomp in block slab; do
            for halo in blocking overlap; do
                run "$c" 1 --d 2 --Q 1000 --decomp $decomp --halo $halo
            done
        done
        run "$c" 1 --d 2 --Q 1000 --dt 0.2
    done ;;
gpu_strong) # fixed problems on 1, 2, 4, ... GPUs; the large one is where GPUs pay off
    for g in $(gpu_counts); do
        run_gpu "$g" --d 2 --Q 1000
        run_gpu "$g" --d 2 --Q 4000
        run_gpu "$g" --d 3 --Q 100
    done ;;
gpu_weak)   # about 16 M points per GPU
    for g in $(gpu_counts); do
        Q=$(awk -v g="$g" 'BEGIN { printf "%d", (sqrt(16000000 * g) - 1) / 2 + 0.5 }')
        run_gpu "$g" --d 2 --Q "$Q"
    done ;;
gpu_vs_cpu) # one GPU node: its 72 CPU cores (MPI x OpenMP) against its 1 and 2 GPUs
    for Q in 250 1000 4000; do
        run $((CPN / THREADS)) "$THREADS" --d 2 --Q $Q
        run_gpu 1 --d 2 --Q $Q
        run_gpu "$GPN" --d 2 --Q $Q
    done ;;
*) echo "unknown experiment $EXP"; exit 1 ;;
esac
