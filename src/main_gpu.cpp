// MPI + OpenMP target offload: one MPI rank per GPU. theta lives on the device for
// the whole run; each step packs the block faces on the GPU, copies them to the host,
// exchanges them with MPI, and copies the received ghost layers back.
//
// The arithmetic per point is the same as kernel.hpp. Results are not bitwise equal
// to the CPU build on a real GPU (the device exp() differs in the last bits), so the
// GPU check compares with a tolerance (tests/compare_gpu.sh).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernel.hpp"
#include "mpi_common.hpp"

// Plain-old-data copy of the model so target regions capture it by value.
struct Coef {
    int d, Q;
    int off0, off1, off2, n0, n1, n2;
    long s0, s1, s2;
    double half_gamma;
    double eta[MAXD], k[MAXD], S[MAXD][MAXD];
};

static Coef make_coef(const Params& p, const Block& b) {
    Coef c{};
    c.d = p.d; c.Q = p.Q;
    c.off0 = b.off[0]; c.off1 = b.off[1]; c.off2 = b.off[2];
    c.n0 = b.n[0]; c.n1 = b.n[1]; c.n2 = b.n[2];
    c.s0 = b.s[0]; c.s1 = b.s[1]; c.s2 = b.s[2];
    c.half_gamma = 0.5 * p.gamma;
    for (int i = 0; i < MAXD; ++i) {
        c.eta[i] = i < p.d ? p.eta[i] : 0.0;
        c.k[i] = p.k[i];
        for (int j = 0; j < MAXD; ++j) c.S[i][j] = (i < p.d && j < p.d) ? p.Sigma[i][j] : 0.0;
    }
    return c;
}

// Same formula as Kernel::rhs_range, over the whole block. Returns max rate.
static double rhs_gpu(const Coef c, const double* th, double* F, int dev) {
    double rmax = 0.0;
    const int g1 = c.d > 1, g2 = c.d > 2;
#pragma omp target teams distribute parallel for collapse(3) reduction(max : rmax) map(tofrom : rmax) map(to : c) \
    is_device_ptr(th, F) device(dev)
    for (int i0 = 0; i0 < c.n0; ++i0)
        for (int i1 = 0; i1 < c.n1; ++i1)
            for (int i2 = 0; i2 < c.n2; ++i2) {
                const long ix = (i0 + 1) * c.s0 + (i1 + g1) * c.s1 + (i2 + g2);
                const int q[MAXD] = {c.off0 + i0 - c.Q, g1 ? c.off1 + i1 - c.Q : 0, g2 ? c.off2 + i2 - c.Q : 0};
                const long st[MAXD] = {c.s0, c.s1, c.s2};
                double quad = 0.0;
                for (int i = 0; i < c.d; ++i)
                    for (int j = 0; j < c.d; ++j) quad += q[i] * c.S[i][j] * q[j];
                const double t = th[ix];
                double h = 0.0, r = 0.0;
                for (int i = 0; i < c.d; ++i) {
                    if (q[i] < c.Q) {
                        const double e = c.eta[i] * exp(-c.k[i] * (t - th[ix + st[i]]));
                        h += e / c.k[i];
                        r += e;
                    }
                    if (q[i] > -c.Q) {
                        const double e = c.eta[i] * exp(-c.k[i] * (t - th[ix - st[i]]));
                        h += e / c.k[i];
                        r += e;
                    }
                }
                F[ix] = h - c.half_gamma * quad;
                rmax = r > rmax ? r : rmax;
            }
    return rmax;
}

static void apply_dt_gpu(const Coef c, const double* th, double* out, double dt, int dev) {
    const int g1 = c.d > 1, g2 = c.d > 2;
#pragma omp target teams distribute parallel for collapse(3) is_device_ptr(th, out) map(to : c) device(dev)
    for (int i0 = 0; i0 < c.n0; ++i0)
        for (int i1 = 0; i1 < c.n1; ++i1)
            for (int i2 = 0; i2 < c.n2; ++i2) {
                const long ix = (i0 + 1) * c.s0 + (i1 + g1) * c.s1 + (i2 + g2);
                out[ix] = th[ix] + dt * out[ix];
            }
}

static void zero_gpu(double* a, long n, int dev) {
#pragma omp target teams distribute parallel for is_device_ptr(a) device(dev)
    for (long i = 0; i < n; ++i) a[i] = 0.0;
}

// Copy one face of the block to/from a contiguous buffer, all on the device.
// layer: local index along dimension a (-1 and n_a are the ghost layers).
// Dimension a's extent is 1, the others are the interior extents.
static void face_copy(const Coef c, double* th, double* buf, int a, int layer, bool pack, int dev) {
    const int g1 = c.d > 1, g2 = c.d > 2;
    const int e0 = a == 0 ? 1 : c.n0, e1 = a == 1 ? 1 : c.n1, e2 = a == 2 ? 1 : c.n2;
#pragma omp target teams distribute parallel for collapse(3) is_device_ptr(th, buf) map(to : c) device(dev)
    for (int j0 = 0; j0 < e0; ++j0)
        for (int j1 = 0; j1 < e1; ++j1)
            for (int j2 = 0; j2 < e2; ++j2) {
                const int i0 = a == 0 ? layer : j0, i1 = a == 1 ? layer : j1, i2 = a == 2 ? layer : j2;
                const long ix = (i0 + 1) * c.s0 + (i1 + g1) * c.s1 + (i2 + g2);
                const long k = (long(j0) * e1 + j1) * e2 + j2;
                if (pack) buf[k] = th[ix];
                else th[ix] = buf[k];
            }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Params p = parse_args(argc, argv);

    // one GPU per rank: ranks on the same node take devices round robin
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int local_rank;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
    const int ndev = omp_get_num_devices();
    const int dev = ndev > 0 ? local_rank % ndev : omp_get_initial_device();
    const int host = omp_get_initial_device();

    Decomp D;
    Block b = setup(p, D);
    const Coef c = make_coef(p, b);
    if (D.rank == 0)
        std::printf("# %d device(s) visible on rank 0%s\n", ndev, ndev ? "" : ": running on the host (fallback)");

    const size_t bytes = b.size() * sizeof(double);
    double* th = static_cast<double*>(omp_target_alloc(bytes, dev));
    double* nx = static_cast<double*>(omp_target_alloc(bytes, dev));
    long max_face = 0;
    for (int a = 0; a < p.d; ++a) max_face = std::max(max_face, b.points() / b.n[a]);
    // device and host buffers: [send lo, send hi, recv lo, recv hi]
    double* dbuf = static_cast<double*>(omp_target_alloc(4 * max_face * sizeof(double), dev));
    std::vector<double> hbuf(4 * max_face), host_theta(b.size());
    if (!th || !nx || !dbuf) {
        std::fprintf(stderr, "rank %d: device allocation failed\n", D.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    zero_gpu(th, b.size(), dev);
    zero_gpu(nx, b.size(), dev);

    double traffic[2];
    halo_traffic(p, D, b, traffic);
    const std::vector<double> times = output_times(p);
    if (!p.out.empty() && D.rank == 0) write_meta(p, times, D.size, 1);

    auto halo = [&](double& t_copy, double& t_mpi) {
        for (int a = 0; a < p.d; ++a) {
            const long fp = b.points() / b.n[a];
            const size_t fb = fp * sizeof(double);
            double* hs_lo = hbuf.data();
            double* hs_hi = hs_lo + max_face;
            double* hr_lo = hs_hi + max_face;
            double* hr_hi = hr_lo + max_face;
            double w = MPI_Wtime();
            if (D.nb_lo[a] != MPI_PROC_NULL) {
                face_copy(c, th, dbuf, a, 0, true, dev);
                omp_target_memcpy(hs_lo, dbuf, fb, 0, 0, host, dev);
            }
            if (D.nb_hi[a] != MPI_PROC_NULL) {
                face_copy(c, th, dbuf + max_face, a, b.n[a] - 1, true, dev);
                omp_target_memcpy(hs_hi, dbuf + max_face, fb, 0, 0, host, dev);
            }
            double w1 = MPI_Wtime();
            t_copy += w1 - w;
            MPI_Sendrecv(hs_hi, int(fp), MPI_DOUBLE, D.nb_hi[a], 2 * a, hr_lo, int(fp), MPI_DOUBLE, D.nb_lo[a],
                         2 * a, D.cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(hs_lo, int(fp), MPI_DOUBLE, D.nb_lo[a], 2 * a + 1, hr_hi, int(fp), MPI_DOUBLE,
                         D.nb_hi[a], 2 * a + 1, D.cart, MPI_STATUS_IGNORE);
            double w2 = MPI_Wtime();
            t_mpi += w2 - w1;
            if (D.nb_lo[a] != MPI_PROC_NULL) {
                omp_target_memcpy(dbuf + 2 * max_face, hr_lo, fb, 0, 0, dev, host);
                face_copy(c, th, dbuf + 2 * max_face, a, -1, false, dev);
            }
            if (D.nb_hi[a] != MPI_PROC_NULL) {
                omp_target_memcpy(dbuf + 3 * max_face, hr_hi, fb, 0, 0, dev, host);
                face_copy(c, th, dbuf + 3 * max_face, a, b.n[a], false, dev);
            }
            t_copy += MPI_Wtime() - w2;
        }
    };

    double t = p.T, t_compute = 0, t_halo = 0, t_copy = 0, t_reduce = 0, t_io = 0;
    long steps = 0;
    MPI_Barrier(D.cart);
    const double t0 = MPI_Wtime();
    for (size_t j = 0; j < times.size(); ++j) {
        const double t_next = times[j];
        while (t > t_next && (p.max_steps < 0 || steps < p.max_steps)) {
            halo(t_copy, t_halo);
            double w = MPI_Wtime();
            double rmax = rhs_gpu(c, th, nx, dev);  // returns after the kernel has finished
            double w1 = MPI_Wtime();
            t_compute += w1 - w;
            if (p.dt <= 0) MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_DOUBLE, MPI_MAX, D.cart);
            double w2 = MPI_Wtime();
            t_reduce += w2 - w1;
            const double dt = choose_dt(p, rmax, t, t_next);
            apply_dt_gpu(c, th, nx, dt, dev);
            std::swap(th, nx);
            t = (dt == t - t_next) ? t_next : t - dt;
            ++steps;
            t_compute += MPI_Wtime() - w2;
        }
        if (t > t_next) break;
        if (!p.out.empty()) {
            double w = MPI_Wtime();
            omp_target_memcpy(host_theta.data(), th, bytes, 0, 0, host, dev);
            write_snapshot(p, D, b, host_theta.data(), int(j));
            t_io += MPI_Wtime() - w;
        }
    }
    MPI_Barrier(D.cart);
    const double total = MPI_Wtime() - t0;
    double ph[5] = {t_compute, t_halo, t_reduce, t_io, t_copy}, phmax[5], trmax[2];
    MPI_Reduce(ph, phmax, 5, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    MPI_Reduce(traffic, trmax, 2, MPI_DOUBLE, MPI_MAX, 0, D.cart);
    if (D.rank == 0) {
        std::string variant = std::string(ndev ? "gpu-" : "gpuhostfallback-") + p.decomp + (p.dt > 0 ? "-fixeddt" : "");
        std::printf("# process grid %d x %d x %d\n", D.dims[0], D.dims[1], D.dims[2]);
        std::printf("%s\n", RESULT_HEADER);
        std::printf("RESULT,%s,%d,%d,%d,%d,%ld,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6g,%.0f,%.0f,%.6f\n", variant.c_str(),
                    D.size, 1, p.d, p.Q, long(p.N()) * (p.d > 1 ? p.N() : 1) * (p.d > 2 ? p.N() : 1), steps, total,
                    phmax[0], phmax[1], phmax[2], phmax[3], t, trmax[0], trmax[1], phmax[4]);
    }
    omp_target_free(th, dev);
    omp_target_free(nx, dev);
    omp_target_free(dbuf, dev);
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
