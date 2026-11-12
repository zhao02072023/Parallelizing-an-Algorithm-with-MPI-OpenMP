// Model and run parameters for the multi-asset market-making HJB solver.
//
// Model (Gueant, Lehalle, Fernandez-Tapia 2013, extended to d assets):
//   theta(T, q) = 0 and, backward in time,
//   d theta / dt = (gamma/2) q' Sigma q
//                  - sum_i (eta_i / k_i) [ exp(-k_i (theta(q) - theta(q + e_i)))  if q_i <  Q
//                                        + exp(-k_i (theta(q) - theta(q - e_i)))  if q_i > -Q ]
//   eta_i = A_i (1 + gamma/k_i)^-(1 + k_i/gamma)
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

constexpr int MAXD = 3;

// One CSV line per run. halo_msgs / halo_bytes: sent per step by the busiest rank;
// copy_s: host <-> device transfers (GPU build only).
constexpr const char* RESULT_HEADER =
    "RESULT,variant,ranks,threads,d,Q,points,steps,total_s,compute_s,halo_s,reduce_s,io_s,t_final,"
    "halo_msgs,halo_bytes,copy_s";

struct Params {
    // model
    int d = 1;
    int Q = 30;
    double T = 600.0;
    double gamma = 0.01;
    double A[MAXD] = {0.9, 0.9, 0.9};
    double k[MAXD] = {0.3, 0.3, 0.3};
    double sigma[MAXD] = {0.3, 0.3, 0.3};
    double rho = 0.0;  // same correlation for every pair of assets

    // numerics
    double cfl = 0.5;     // dt = cfl / max_q (sum of arrival rates at q)
    double dt = 0.0;      // > 0: fixed time step instead of adaptive
    long max_steps = -1;  // > 0: stop after this many steps (benchmarks)
    int snapshots = 1;    // number of equally spaced output times, the last is t = 0
    std::string out;      // output directory; empty = no output

    // parallel variants (MPI build only)
    std::string decomp = "block";  // block | slab
    std::string halo = "blocking"; // blocking | overlap

    // derived
    double eta[MAXD];
    double Sigma[MAXD][MAXD];
    int N() const { return 2 * Q + 1; }

    void derive() {
        for (int i = 0; i < d; ++i)
            eta[i] = A[i] * std::pow(1.0 + gamma / k[i], -(1.0 + k[i] / gamma));
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Sigma[i][j] = (i == j ? 1.0 : rho) * sigma[i] * sigma[j];
    }
};

inline void parse_list(const char* s, double* v, int n) {
    // "0.3" sets all entries, "0.3,0.5" sets the first two and repeats the last
    std::stringstream ss(s);
    std::string tok;
    int i = 0;
    double last = 0;
    while (std::getline(ss, tok, ',') && i < n) v[i++] = last = std::atof(tok.c_str());
    for (; i < n; ++i) v[i] = last;
}

inline void usage() {
    std::fprintf(stderr,
        "options: --d 1|2|3 --Q int --T sec --gamma g --A a[,a..] --k k[,k..] --sigma s[,s..]\n"
        "         --rho r --cfl c --dt fixed --max-steps n --snapshots n --out dir\n"
        "         --decomp block|slab --halo blocking|overlap\n");
}

inline Params parse_args(int argc, char** argv) {
    Params p;
    for (int a = 1; a < argc; ++a) {
        std::string key = argv[a];
        if (key == "-h" || key == "--help") { usage(); std::exit(0); }
        if (a + 1 >= argc) { usage(); std::exit(1); }
        const char* v = argv[++a];
        if (key == "--d") p.d = std::atoi(v);
        else if (key == "--Q") p.Q = std::atoi(v);
        else if (key == "--T") p.T = std::atof(v);
        else if (key == "--gamma") p.gamma = std::atof(v);
        else if (key == "--A") parse_list(v, p.A, MAXD);
        else if (key == "--k") parse_list(v, p.k, MAXD);
        else if (key == "--sigma") parse_list(v, p.sigma, MAXD);
        else if (key == "--rho") p.rho = std::atof(v);
        else if (key == "--cfl") p.cfl = std::atof(v);
        else if (key == "--dt") p.dt = std::atof(v);
        else if (key == "--max-steps") p.max_steps = std::atol(v);
        else if (key == "--snapshots") p.snapshots = std::atoi(v);
        else if (key == "--out") p.out = v;
        else if (key == "--decomp") p.decomp = v;
        else if (key == "--halo") p.halo = v;
        else { std::fprintf(stderr, "unknown option %s\n", key.c_str()); usage(); std::exit(1); }
    }
    if (p.d < 1 || p.d > MAXD) { std::fprintf(stderr, "d must be 1..3\n"); std::exit(1); }
    p.derive();
    return p;
}

// Times (descending from T) at which theta is written; the last one is 0.
inline std::vector<double> output_times(const Params& p) {
    std::vector<double> t;
    for (int j = 1; j <= p.snapshots; ++j) t.push_back(p.T * (1.0 - double(j) / p.snapshots));
    return t;
}

// Text header next to the binary snapshots, read by python/hjbio.py.
inline void write_meta(const Params& p, const std::vector<double>& times, int nranks, int nthreads) {
    std::string path = p.out + "/meta.txt";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "d %d\nQ %d\nT %.17g\ngamma %.17g\nrho %.17g\n", p.d, p.Q, p.T, p.gamma, p.rho);
    for (int i = 0; i < p.d; ++i)
        std::fprintf(f, "asset %d A %.17g k %.17g sigma %.17g\n", i, p.A[i], p.k[i], p.sigma[i]);
    std::fprintf(f, "ranks %d\nthreads %d\n", nranks, nthreads);
    for (size_t j = 0; j < times.size(); ++j) std::fprintf(f, "snapshot %zu %.17g\n", j, times[j]);
    std::fclose(f);
}
