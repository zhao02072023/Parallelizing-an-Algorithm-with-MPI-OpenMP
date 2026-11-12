#!/bin/bash
# Queue every benchmark for the report, after jobs/pbs.sh EXP=binding looked right.
# These are the exact resource requests used for the report runs on Vanda (Nov 2026).
# Vanda limits: at most 360 CPU cores and 2 GPUs per job, and 4 jobs at a time per user,
# so qsub refuses the later lines while 4 are queued or running: rerun those once some finish.
set -euo pipefail
cpu() { echo "select=$1:ncpus=72:mpiprocs=72:mem=200gb"; }   # $1 whole CPU nodes
for e in strong strong3d weak variants; do                  # 4 nodes = 288 cores
    qsub -l "$(cpu 4)" -l place=scatter -v EXP=$e jobs/pbs.sh
done
qsub -l "$(cpu 1)" -l place=scatter -v EXP=hybrid jobs/pbs.sh
qsub -l "$(cpu 2)" -l place=scatter -v EXP=hybrid jobs/pbs.sh
qsub -l select=1:ncpus=36:mpiprocs=36:mem=100gb -v EXP=serial     jobs/pbs.sh
qsub -l select=1:ncpus=36:mpiprocs=36:mem=100gb -v EXP=serialfull jobs/pbs.sh
# GPU: one whole GPU node = 2 sockets x (36 cores + 1 A40), the 2-GPU per-job maximum
for e in gpu_vs_cpu gpu_strong gpu_weak; do
    qsub -l select=2:ncpus=36:mpiprocs=36:ngpus=1:mem=200gb -l place=pack -v EXP=$e jobs/pbs_gpu.sh
done
