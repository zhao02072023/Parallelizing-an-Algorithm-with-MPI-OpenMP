// The time-step kernel, shared by the serial and MPI builds so both do exactly
// the same arithmetic at every lattice point.
//
// One backward Euler-in-time step from t to t - dt is
//   theta_new = theta + dt * F(theta),   F = sum_i H_i - (gamma/2) q' Sigma q
// and is done in two sweeps:
//   1. rhs_range: F into the output buffer, and the largest "stiffness rate"
//      r(q) = sum_i eta_i exp(-k_i (theta(q) - theta(q +- e_i))). The Jacobian of
//      F has its eigenvalues in [-2 r_max, 0], so explicit Euler is stable for
//      dt <= 1 / r_max.
//   2. apply_dt: out = theta + dt * out, once dt is known (it needs a global max).
#pragma once

#include <algorithm>
#include <cmath>

#include "grid.hpp"

struct Kernel {
    int d, Q;
    double half_gamma;
    double eta[MAXD], eta_over_k[MAXD], k[MAXD], Sigma[MAXD][MAXD];

    explicit Kernel(const Params& p) : d(p.d), Q(p.Q), half_gamma(0.5 * p.gamma) {
        for (int i = 0; i < MAXD; ++i) {
            eta[i] = i < d ? p.eta[i] : 0.0;
            k[i] = p.k[i];
            eta_over_k[i] = eta[i] / k[i];
            for (int j = 0; j < MAXD; ++j) Sigma[i][j] = (i < d && j < d) ? p.Sigma[i][j] : 0.0;
        }
    }

    // F over local interior points lo <= l < hi (per dimension). Returns max rate.
    double rhs_range(const Block& b, const double* __restrict th, double* __restrict F,
                     const int lo[MAXD], const int hi[MAXD]) const {
        double rmax = 0.0; // stiffness
        const long s0 = b.s[0], s1 = b.s[1], s2 = b.s[2];
        const long st[MAXD] = {s0, s1, s2};
#pragma omp parallel for collapse(2) reduction(max : rmax) schedule(static)
        for (int i0 = lo[0]; i0 < hi[0]; ++i0) {
            for (int i1 = lo[1]; i1 < hi[1]; ++i1) {
                long c = b.at(i0, i1, lo[2]);
                const int q0 = b.off[0] + i0 - Q;
                const int q1 = d > 1 ? b.off[1] + i1 - Q : 0;
                for (int i2 = lo[2]; i2 < hi[2]; ++i2, ++c) {
                    const int q2 = d > 2 ? b.off[2] + i2 - Q : 0;
                    const int q[MAXD] = {q0, q1, q2};
                    double quad = 0.0;
                    for (int i = 0; i < d; ++i) {
                        for (int j = 0; j < d; ++j) {
                             quad += q[i] * Sigma[i][j] * q[j];
                        }
                    }
                            
                    const double t = th[c];
                    double h = 0.0, r = 0.0;
                    for (int i = 0; i < d; ++i) { // compute the optimal execution intensities
                        if (q[i] < Q) {  // can still buy: bid side
                            const double e = eta[i] * std::exp(-k[i] * (t - th[c + st[i]]));
                            h += e / k[i];
                            r += e;
                        }
                        if (q[i] > -Q) {  // can still sell: ask side
                            const double e = eta[i] * std::exp(-k[i] * (t - th[c - st[i]]));
                            h += e / k[i];
                            r += e;
                        }
                    }
                    F[c] = h - half_gamma * quad; // update the output vector F
                    rmax = std::max(rmax, r);
                }
            }
        }
        return rmax;
    }

    double rhs(const Block& b, const double* th, double* F) const {
        const int lo[MAXD] = {0, 0, 0};
        const int hi[MAXD] = {b.n[0], b.n[1], b.n[2]};
        return rhs_range(b, th, F, lo, hi);
    }

    // out = theta + dt * out on interior points
    static void apply_dt(const Block& b, const double* __restrict th, double* __restrict out, double dt) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < b.n[0]; ++i0)
            for (int i1 = 0; i1 < b.n[1]; ++i1) {
                const long c = b.at(i0, i1, 0);
                for (int i2 = 0; i2 < b.n[2]; ++i2) {
                    out[c + i2] = th[c + i2] + dt * out[c + i2];
                }
            }
    }

    // zero everything (ghosts included) with the same thread layout as the sweeps,
    // so pages are first touched by the thread that will use them
    static void first_touch(const Block& b, double* a) {
#pragma omp parallel for schedule(static)
        for (long i0 = 0; i0 < b.ext[0]; ++i0)
            for (long j = 0; j < b.s[0]; ++j) a[i0 * b.s[0] + j] = 0.0;
    }
};

// Next time step size: adaptive from the global max rate, never stepping past the next output time.
inline double choose_dt(const Params& p, double rmax, double t, double t_next) {
    double dt = p.dt > 0 ? p.dt : (rmax > 0 ? p.cfl / rmax : t - t_next);
    if (t - dt < t_next + 1e-12 * p.T) dt = t - t_next;
    return dt;
}
