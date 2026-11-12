#!/bin/bash
# PBS wrapper for Vanda CPU nodes (2 x 36-core Xeon 8452Y). Submit from the repo root:
#   qsub -v EXP=binding  jobs/pbs.sh                                   # 1 node, check pinning first
#   qsub -l select=4:ncpus=72:mpiprocs=72 -v EXP=strong jobs/pbs.sh   # 4 nodes = 288 cores
# The -l on the command line overrides the select line below.
#PBS -N hjb-bench
#PBS -l select=1:ncpus=72:mpiprocs=72:mem=200gb
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00
#PBS -j oe
# (remove the place=...:excl line if the site does not allow exclusive nodes; timings
#  are noisier when other jobs share the node, so say so in the report)
set -euo pipefail
cd "$PBS_O_WORKDIR"
source jobs/modules.sh
export NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
export CORES_PER_NODE=${CORES_PER_NODE:-72}
HOSTS=$PBS_O_WORKDIR/results/hosts.$PBS_JOBID
mkdir -p results
case "$($MPIRUN_CMD --version 2>&1)" in
    *"Open MPI"*) sort -u "$PBS_NODEFILE" | sed "s/\$/ slots=$CORES_PER_NODE/" > "$HOSTS"
                  export MPIRUN="$MPIRUN_CMD --hostfile $HOSTS" ;;
    *)            sort -u "$PBS_NODEFILE" > "$HOSTS"
                  export MPIRUN="$MPIRUN_CMD -f $HOSTS" ;;
esac
echo "job $PBS_JOBID: $NODES node(s), experiment ${EXP:-strong}"
jobs/bench.sh "${EXP:-strong}"
