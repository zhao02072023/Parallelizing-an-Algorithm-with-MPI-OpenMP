// MPI + OpenMP solver: Cartesian domain decomposition of the inventory lattice,
// one ghost layer exchanged with each face neighbour every step, and a global
// max reduction for the adaptive time step. OpenMP threads share each block.
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel.hpp"
#include "mpi_common.hpp"

static void halo_blocking(const Params& p, const Decomp& D, double* th) {
    for (int a = 0; a < p.d; ++a) {
        MPI_Sendrecv(th, 1, D.send_hi[a], D.nb_hi[a], 2 * a, th, 1, D.recv_lo[a], D.nb_lo[a], 2 * a, D.cart,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(th, 1, D.send_lo[a], D.nb_lo[a], 2 * a + 1, th, 1, D.recv_hi[a], D.nb_hi[a], 2 * a + 1,
                     D.cart, MPI_STATUS_IGNORE);
    }
}

// The stencil is a star (no corners), so all faces can be in flight at once.
static int halo_start(const Params& p, const Decomp& D, double* th, MPI_Request* req) {
    int n = 0;
    for (int a = 0; a < p.d; ++a) {
        MPI_Irecv(th, 1, D.recv_lo[a], D.nb_lo[a], 2 * a, D.cart, &req[n++]);
        MPI_Irecv(th, 1, D.recv_hi[a], D.nb_hi[a], 2 * a + 1, D.cart, &req[n++]);
        MPI_Isend(th, 1, D.send_hi[a], D.nb_hi[a], 2 * a, D.cart, &req[n++]);
        MPI_Isend(th, 1, D.send_lo[a], D.nb_lo[a], 2 * a + 1, D.cart, &req[n++]);
    }
    return n;
}

// F on the points that touch a face of the block (they need ghost values).
// A point belongs to the slab of the first dimension in which it lies on a face.
static double rhs_shell(const Params& p, const Kernel& K, const Block& b, const double* th, double* F) {
    double rmax = 0.0;
    for (int a = 0; a < p.d; ++a) {
        int lo[MAXD], hi[MAXD];
        for (int j = 0; j < MAXD; ++j) {
            if (j < a) { lo[j] = b.g[j]; hi[j] = b.n[j] - b.g[j]; }
            else { lo[j] = 0; hi[j] = b.n[j]; }
        }
        lo[a] = 0; hi[a] = 1;
        rmax = std::max(rmax, K.rhs_range(b, th, F, lo, hi));
        if (b.n[a] > 1) {
            lo[a] = b.n[a] - 1; hi[a] = b.n[a];
            rmax = std::max(rmax, K.rhs_range(b, th, F, lo, hi));
        }
    }
    return rmax;
}

static double rhs_interior(const Kernel& K, const Block& b, const double* th, double* F) {
    int lo[MAXD], hi[MAXD];
    for (int j = 0; j < MAXD; ++j) { lo[j] = b.g[j]; hi[j] = b.n[j] - b.g[j]; }
    for (int j = 0; j < MAXD; ++j) if (hi[j] <= lo[j]) return 0.0;
    return K.rhs_range(b, th, F, lo, hi);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Params p = parse_args(argc, argv);
    if (p.halo != "blocking" && p.halo != "overlap") {
        std::fprintf(stderr, "--halo must be blocking or overlap\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const bool overlap = p.halo == "overlap";
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    Decomp D;
    Block b = setup(p, D);
    Kernel K(p);
    std::vector<double> a(b.size()), c(b.size());
    Kernel::first_touch(b, a.data());
    Kernel::first_touch(b, c.data());
    double *th = a.data(), *nx = c.data();

    double traffic[2];  // messages, bytes sent per step
    halo_traffic(p, D, b, traffic);

    const std::vector<double> times = output_times(p);
    if (!p.out.empty() && D.rank == 0) write_meta(p, times, D.size, nthreads);

    double t = p.T, t_compute = 0, t_halo = 0, t_reduce = 0, t_io = 0;
    long steps = 0;
    MPI_Request req[4 * MAXD];
    MPI_Barrier(D.cart);
    const double t0 = MPI_Wtime();
    for (size_t j = 0; j < times.size(); ++j) {
        const double t_next = times[j];
        while (t > t_next && (p.max_steps < 0 || steps < p.max_steps)) {
            double rmax, w = MPI_Wtime();
            if (overlap) {
                int nreq = halo_start(p, D, th, req);
                double w1 = MPI_Wtime();
                t_halo += w1 - w;
                rmax = rhs_interior(K, b, th, nx);
                double w2 = MPI_Wtime();
                t_compute += w2 - w1;
                MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
                double w3 = MPI_Wtime();
                t_halo += w3 - w2;
                rmax = std::max(rmax, rhs_shell(p, K, b, th, nx));
                t_compute += MPI_Wtime() - w3;
            } else {
                halo_blocking(p, D, th);
                double w1 = MPI_Wtime();
                t_halo += w1 - w;
                rmax = K.rhs(b, th, nx);
                t_compute += MPI_Wtime() - w1;
            }
            w = MPI_Wtime();
            if (p.dt <= 0) MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_DOUBLE, MPI_MAX, D.cart);
            double w1 = MPI_Wtime();
            t_reduce += w1 - w;
            const double dt = choose_dt(p, rmax, t, t_next);
            Kernel::apply_dt(b, th, nx, dt);
            std::swap(th, nx);
            t = (dt == t - t_next) ? t_next : t - dt;
            ++steps;
            t_compute += MPI_Wtime() - w1;
        }
        if (t > t_next) break;  // stopped by --max-steps
        if (!p.out.empty()) {
            double w = MPI_Wtime();
            write_snapshot(p, D, b, th, int(j));
            t_io += MPI_Wtime() - w;
        }
    }
    MPI_Barrier(D.cart);
    const double total = MPI_Wtime() - t0;
    double ph[4] = {t_compute, t_halo, t_reduce, t_io}, phmax[4];
    MPI_Reduce(ph, phmax, 4, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    double trmax[2];
    MPI_Reduce(traffic, trmax, 2, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    if (D.rank == 0) {
        std::string variant = p.decomp + "-" + p.halo + (p.dt > 0 ? "-fixeddt" : "");
        std::printf("# process grid %d x %d x %d\n", D.dims[0], D.dims[1], D.dims[2]);
        std::printf("%s\n", RESULT_HEADER);
        std::printf("RESULT,%s,%d,%d,%d,%d,%ld,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6g,%.0f,%.0f,0\n", variant.c_str(),
                    D.size, nthreads, p.d, p.Q, long(p.N()) * (p.d > 1 ? p.N() : 1) * (p.d > 2 ? p.N() : 1), steps,
                    total, phmax[0], phmax[1], phmax[2], phmax[3], t, trmax[0], trmax[1]);
    }
    for (int i = 0; i < p.d; ++i) {
        MPI_Type_free(&D.send_lo[i]);
        MPI_Type_free(&D.send_hi[i]);
        MPI_Type_free(&D.recv_lo[i]);
        MPI_Type_free(&D.recv_hi[i]);
    }
    MPI_Comm_free(&D.cart);
    MPI_Finalize();
    return 0;
}
