## Build

```bash
source jobs/modules.sh     # Vanda: nvhpc-hpcx-cuda12/26.1 (MPI + nvc++) and gcc/13.1.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
# GPU build (OpenMP target offload to the A40, compute capability 86)
OMPI_CXX=nvc++ cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=nvc++ -DGPU=ON
OMPI_CXX=nvc++ cmake --build build-gpu -j --target hjb_gpu
```

This gives `build/hjb_serial` (OpenMP only), `build/hjb_mpi` (MPI + OpenMP) and
`build-gpu/hjb_gpu` (MPI + OpenMP offload, one rank per GPU). On Vanda, `qsub jobs/build.pbs`
does all of this on a GPU node and runs every check.
Add `-DNATIVE=OFF` if building on a node with a different CPU from the compute nodes.
`-ffast-math` is deliberately off: the parallel runs are tested to be byte-identical to serial.

## Run

```bash
# the paper's example (Figures 1-3), 60 snapshots in time
./build/hjb_serial --d 1 --Q 30 --snapshots 60 --out out/d1

# two correlated stocks, 4 ranks x 2 threads, 2D block decomposition with overlapped halos
OMP_NUM_THREADS=2 mpirun -n 4 ./build/hjb_mpi --d 2 --Q 200 --rho 0.5 --halo overlap --out out/d2
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--d` | 1 | number of stocks (1–3) |
| `--Q` | 30 | inventory limit per stock; lattice has (2Q+1)^d points |
| `--T` | 600 | horizon [s] |
| `--gamma` | 0.01 | risk aversion [1/Tick] |
| `--A`, `--k`, `--sigma` | 0.9, 0.3, 0.3 | per stock, comma list (last value repeats) |
| `--rho` | 0 | correlation between every pair of stocks |
| `--cfl` | 0.5 | adaptive step dt = cfl / max_q r(q), see below |
| `--dt` | off | fixed time step instead (skips the global reduction) |
| `--max-steps` | off | stop after n steps (benchmarks) |
| `--snapshots` | 1 | write θ at n equally spaced times, the last at t = 0 |
| `--out` | none | output directory |
| `--decomp` | block | MPI only: `block` (MPI_Dims_create over all d) or `slab` (split q_1 only) |
| `--halo` | blocking | MPI only: `blocking` (Sendrecv) or `overlap` (Isend/Irecv, interior computed while messages fly) |

Each run prints one `RESULT,...` CSV line: variant, ranks, threads, d, Q, points, steps,
total time, and the max over ranks of compute, halo, reduction and I/O time.

Output: `meta.txt` plus `theta_XXX.bin`, raw float64 in row-major (q_1 slowest) order,
written with MPI-IO by the parallel code. `python/hjbio.py` reads them.

## Algorithm and parallel design

* **Time stepping.** Explicit Euler backward in time. A step is two sweeps:
  (1) F(θ) and the local stiffness rate r(q) = Σ η_i e^{−k_i Δθ} at every point,
  (2) θ_new = θ + dt·F once dt is known. The Jacobian of F has its eigenvalues in [−2 r_max, 0],
  so dt = cfl / r_max with cfl ≤ 1 is stable. The step is cut to land exactly on output times.
* **Decomposition.** `MPI_Cart_create` over the d inventory dimensions; each rank owns a
  block plus one ghost layer per face. Faces are `MPI_Type_create_subarray` types, so no packing code.
* **Per-step communication.** 2d face exchanges (star stencil, no corners) and one
  `MPI_Allreduce(MAX)` of r for the adaptive dt. The reduction is also where load imbalance
  shows up, since faster ranks wait there.
* **Overlap variant.** Post all Irecv/Isend, compute the block interior (needs no ghosts),
  `MPI_Waitall`, then compute the one-point-thick shell.
* **OpenMP.** `collapse(2)` static loops inside each rank, with a max reduction;
  arrays are first touched with the same schedule. MPI is called only from the master
  thread (`MPI_THREAD_FUNNELED`).
* **GPU (`hjb_gpu`).** One MPI rank per GPU (ranks on a node take devices round robin).
  θ stays in device memory (`omp_target_alloc`) for the whole run; the same two sweeps run as
  `omp target teams distribute parallel for collapse(3)`, the first with a max reduction.
  Halo exchange per step: pack each face on the GPU into a contiguous buffer, copy it to the
  host, `MPI_Sendrecv`, copy the received face back and unpack it into the ghost layer.
  The copies are timed separately (`copy_s`). GPU-aware MPI would remove them; not used here.
* **Determinism.** Every point's arithmetic is identical in every configuration and max is
  order-independent, so all variants write byte-identical files (`tests/compare.sh`).

### Stiffness: why the benchmarks use γ = 1e−4

The step limit is set by the fastest order arrivals, which come from the deep, inventory-dumping
quotes at large |q|. Those grow roughly like e^{k·c·Q} with c ∝ √(γσ²/(kA)). Measured full-horizon
step counts for d = 1:

| Q | γ = 0.01 | γ = 0.001 | γ = 0.0001 |
| --- | --- | --- | --- |
| 30 | 904 | 802 | 795 |
| 300 | 14 021 | 1 717 | 822 |
| 1000 | 154 334 | 12 435 | 1 093 |

So the paper's γ = 0.01 is fine at Q = 30 (validation), and the large benchmark lattices
use γ = 1e−4. Benchmarks run a fixed number of steps anyway (`--max-steps`).
An implicit scheme would remove this limit; that is out of scope here.

## Validation

```bash
pip install numpy scipy matplotlib
tests/validate.sh        # set MPIRUN if plain "mpirun --oversubscribe" is wrong for your MPI
```

1. d = 1 against the paper's exact solution v(t) = expm(−M(T−t))·1 (`python/reference.py`):
   relative max error 3e−5 at cfl 0.5.
2. Convergence: the error halves each time cfl halves (first order, as expected for Euler).
3. d = 2, 3 with ρ = 0 (any k_i): θ is exactly the sum of 1D solutions. With equal k and ρ ≠ 0:
   the linearised system v' = Mv on the whole lattice via `scipy.sparse.linalg.expm_multiply`.
   Relative errors 2e−6 to 8e−6.
4. Parallel = serial byte for byte, for 1–8 ranks, both decompositions, both halo modes, and with threads.
5. GPU = serial within a relative 1e−9 (`tests/compare_gpu.sh`, run by `jobs/build.pbs` with
   `OMP_TARGET_OFFLOAD=MANDATORY` so it cannot silently run on the host). The device `exp()`
   differs from glibc's in the last bits, so byte equality is not expected.

## Running on Vanda (PBS)

CPU nodes: 2 × 36-core Xeon 8452Y, 512 GB. GPU nodes: the same plus 2 × A40.
The scripts default to 72 cores per node, one MPI rank per socket (36 threads) for hybrid
runs, and one rank per GPU pinned to its own socket for GPU runs.

Vanda's network is 100 GbE only, with no InfiniBand or RoCE. HPC-X's defaults (hcoll
collectives and UCX RDMA transports) hang at the first multi-node run, so `jobs/modules.sh`
sets `MPIRUN_CMD` to turn hcoll off and run UCX over TCP on the nodes' NIC (`ens6f1`).
Every inter-node message therefore pays TCP latency (tens of µs), which shows up in the scaling
results. Vanda's per-job limits are 360 CPU cores and 2 GPUs, with 4 jobs at a time.

1. `qsub jobs/build.pbs`.
   It builds both versions and runs every check. Look for `ALL PASSED` twice in its output.
2. `qsub -v EXP=binding jobs/pbs.sh`: one 5-step run that prints where each rank is pinned.
   Each rank should own the 36 cores of one socket. Fix `BIND` in `jobs/bench.sh` if not.
3. queues every experiment below with the resource requests used for
   the report (only 4 are accepted at a time, so resubmit the rest as jobs finish):
   ```bash
   for e in strong strong3d variants weak; do
      qsub -l select=4:ncpus=72:mpiprocs=72:mem=200gb -l place=scatter -v EXP=$e jobs/pbs.sh
   done

   for e in gpu_vs_cpu gpu_strong gpu_weak; do
      qsub -l select=2:ncpus=36:mpiprocs=36:ngpus=1:mem=200gb -l place=pack -v EXP=$e jobs/pbs_gpu.sh
   done

   qsub -l select=1:ncpus=36:mpiprocs=36:mem=100gb -v EXP=serial jobs/pbs.sh
   qsub -l select=1:ncpus=36:mpiprocs=36:mem=100gb -v EXP=serialfull jobs/pbs.sh
   qsub -l select=1:ncpus=72:mpiprocs=72:mem=200gb -l place=scatter -v EXP=hybrid jobs/pbs.sh
   qsub -l select=2:ncpus=72:mpiprocs=72:mem=200gb -l place=scatter -v EXP=hybrid jobs/pbs.sh
   ```
4. Report figures and tables from all CSVs in `results/raw/`:
   `python3 python/report_figs.py results/raw figs` (strong/weak scaling, time breakdown,
   variants, hybrid split, GPU vs CPU, `tables.md`) and
   `python3 python/plots.py serial <serial+serialfull csv> figs` for the serial figure.
   `python3 python/plots.py scaling|gpu|table <csv> figs` gives quick per-experiment plots.

| Experiment | What it runs | Nodes |
| --- | --- | --- |
| `serial` | 1 core, cost per step vs lattice size: d = 1 (2 k – 2 M points), d = 2 (10 k – 10 M), d = 3 (9 k – 33 M) | 1 CPU |
| `serialfull` | 1 core, whole horizon: stable step count vs Q for γ = 0.01 and 1e−4 | 1 CPU |
| `strong` | d = 2, Q = 1000 (4.0 M points) on 1, 2, 4, 9, 18, 36, 72, 144, 288 cores; pure MPI and MPI × 36 threads | 4 CPU |
| `strong3d` | d = 3, Q = 100 (8.1 M points), same core counts | 4 CPU |
| `weak` | ~250 k points per core, Q grows with the core count | 4 CPU |
| `hybrid` | every ranks × threads split that fills the nodes (72×1 … 1×72 per node) | 1 and 2 CPU |
| `variants` | block/slab × blocking/overlap halo, plus fixed dt (no allreduce), pure MPI | 4 CPU |
| `gpu_vs_cpu` | one GPU node: its 72 cores (2 × 36) vs 1 GPU vs 2 GPUs, Q = 250, 1000, 4000 | 1 GPU |
| `gpu_strong` | d = 2 Q = 1000 and 4000 (64 M points), d = 3 Q = 100, on 1 and 2 GPUs | 1 GPU |
| `gpu_weak` | ~16 M points per GPU, 1 and 2 GPUs | 1 GPU |

The lattice size plays the role of the matrix dimension and d sets the neighbours per point
(2d), the analogue of nonzeros per row, so `serial` covers both. Core counts below one node
run on the first node only, so the jump after 72 cores in the plots is the network.
Each configuration is repeated 3 times (`REPEATS`) for 200 steps (`STEPS`); tables use the median.
Every CSV row also has `halo_msgs` and `halo_bytes` (sent per step by the busiest rank) for
the communication analysis. `jobs/bench.sh` also works outside PBS: set `NODES`,
`CORES_PER_NODE`, `MPIRUN`.

`python/plots.py paper out/d1 figs` re-draws the paper's Figures 1–3 and
`python/plots.py quotes2d <out> figs` the 2-stock quote map.

## Layout

```
src/params.hpp       model + run parameters, CLI
src/grid.hpp         block geometry and ghost layers
src/kernel.hpp       the time-step kernel shared by both builds
src/main_serial.cpp  serial driver
src/main_mpi.cpp     MPI + OpenMP driver: halos (blocking / overlapped), allreduce
src/main_gpu.cpp     MPI + OpenMP target offload driver, one rank per GPU
src/mpi_common.hpp   Cartesian topology, face datatypes, MPI-IO snapshots
python/              reference solutions, checks, plots, tables
tests/               validate.sh, compare.sh (byte-for-byte), compare_gpu.sh (tolerance)
jobs/                bench.sh driver, PBS wrappers for Vanda, modules.sh
```

## AI usage

This code was written with Claude (Anthropic) from a plan agreed with the author; the
author reviewed, ran and is responsible for it. Describe the details in the report's AI-usage section.
