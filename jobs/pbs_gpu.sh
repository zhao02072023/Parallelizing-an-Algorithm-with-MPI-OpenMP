#!/bin/bash
# PBS wrapper for Vanda GPU nodes (2 x A40 + 2 x 36 cores each): one MPI rank per GPU.
#   qsub -v EXP=gpu_vs_cpu jobs/pbs_gpu.sh     # one whole GPU node (2 GPUs, the per-job maximum)
# Vanda's chunk is one socket (36 cores + 1 GPU), so a whole node is 2 chunks packed together.
#PBS -N hjb-gpu
#PBS -l select=2:ncpus=36:mpiprocs=36:ngpus=1:mem=200gb
#PBS -l place=pack
#PBS -l walltime=01:00:00
#PBS -j oe
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/modules.sh
export NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
export CORES_PER_NODE=${CORES_PER_NODE:-72}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
mkdir -p results
HOSTS=$PBS_O_WORKDIR/results/hosts.$PBS_JOBID
sort -u "$PBS_NODEFILE" | sed "s/\$/ slots=$CORES_PER_NODE/" > "$HOSTS"
export MPIRUN="$MPIRUN_CMD --hostfile $HOSTS"
nvidia-smi -L
echo "job $PBS_JOBID: $NODES node(s), experiment ${EXP:-gpu_vs_cpu}"
jobs/bench.sh "${EXP:-gpu_vs_cpu}"
