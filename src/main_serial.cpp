// Serial reference solver (OpenMP threads allowed, no MPI).
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel.hpp"

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void write_snapshot(const Params& p, const Block& b, const double* th, int j) {
    char name[64];
    std::snprintf(name, sizeof name, "/theta_%03d.bin", j);
    std::string path = p.out + name;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    for (int i0 = 0; i0 < b.n[0]; ++i0)
        for (int i1 = 0; i1 < b.n[1]; ++i1) std::fwrite(th + b.at(i0, i1, 0), sizeof(double), b.n[2], f);
    std::fclose(f);
}
## Entry point
int main(int argc, char** argv) { # 
    Params p = parse_args(argc, argv);
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    Block b = whole_block(p);
    Kernel K(p);
    std::vector<double> a(b.size()), c(b.size());
    Kernel::first_touch(b, a.data());
    Kernel::first_touch(b, c.data());
    double *th = a.data(), *nx = c.data();  // theta(T, q) = 0

    const std::vector<double> times = output_times(p);
    if (!p.out.empty()) write_meta(p, times, 1, nthreads);

    double t = p.T, t_compute = 0, t_io = 0;
    long steps = 0;
    const double t0 = now();
    for (size_t j = 0; j < times.size(); ++j) {
        const double t_next = times[j];
        while (t > t_next && (p.max_steps < 0 || steps < p.max_steps)) {
            double c0 = now();
            const double rmax = K.rhs(b, th, nx);            // compute F for each lattice point and maximum stiffness
            const double dt = choose_dt(p, rmax, t, t_next); // set dt
            Kernel::apply_dt(b, th, nx, dt);
            std::swap(th, nx);
            t = (dt == t - t_next) ? t_next : t - dt;
            ++steps;
            t_compute += now() - c0;
        }
        if (t > t_next) break;  // stopped by --max-steps
        if (!p.out.empty()) {
            double i0 = now();
            write_snapshot(p, b, th, int(j));
            t_io += now() - i0;
        }
    }
    const double total = now() - t0;
    std::printf("%s\n", RESULT_HEADER);
    std::printf("RESULT,serial,1,%d,%d,%d,%ld,%ld,%.6f,%.6f,0,0,%.6f,%.6g,0,0,0\n", nthreads, p.d, p.Q, b.points(),
                steps, total, t_compute, t_io, t);
    return 0;
}
