# Vanda modules, sourced by every job script.
#   CPU build (build/):     g++ 13.1 + HPC-X MPI (Open MPI) from the NVIDIA HPC SDK module
#   GPU build (build-gpu/): nvc++ from the same module, OpenMP offload to the A40 (cc86)
# Python (tests and plots) needs numpy, scipy, matplotlib:  python3 -m pip install --user numpy scipy matplotlib
module purge 2>/dev/null || true
module load nvhpc-hpcx-cuda12/26.1
module load gcc/13.1.0
export OMPI_CXX=g++                 # make HPC-X's mpicxx wrap g++ for the CPU build
# Vanda has 100 GbE only (one NIC, ens6f1), no InfiniBand/RoCE. HPC-X's defaults (hcoll and
# UCX RDMA transports) hang at the first multi-node run, so turn hcoll off and run UCX over TCP.
export MPIRUN_CMD=${MPIRUN_CMD:-"mpirun --mca coll ^hcoll -x HCOLL_ENABLE=0 -x UCX_TLS=self,sm,tcp -x UCX_NET_DEVICES=ens6f1 --mca oob_tcp_if_include ens6f1"}
# Vanda compute nodes have no cmake module: install it once with
#   python3 -m pip install --user cmake
export PATH=$HOME/.local/bin:$PATH
