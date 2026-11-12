// MPI pieces shared by the CPU (hjb_mpi) and GPU (hjb_gpu) drivers: Cartesian
// decomposition, face datatypes, halo traffic accounting and MPI-IO snapshots.
#pragma once

#include <mpi.h>

#include <cstdio>
#include <string>

#include "grid.hpp"

struct Decomp {
    MPI_Comm cart;
    int rank, size;
    int dims[MAXD] = {1, 1, 1}, coords[MAXD] = {0, 0, 0};
    int nb_lo[MAXD], nb_hi[MAXD];  // face neighbours (MPI_PROC_NULL at the lattice edge)
    // per active dimension: faces we send from and ghost layers we receive into
    MPI_Datatype send_lo[MAXD], send_hi[MAXD], recv_lo[MAXD], recv_hi[MAXD];
};

inline MPI_Datatype face(const Block& b, int a, int start) {
    int sizes[MAXD], sub[MAXD], st[MAXD];
    for (int j = 0; j < MAXD; ++j) {
        sizes[j] = int(b.ext[j]);
        sub[j] = j == a ? 1 : b.n[j];
        st[j] = j == a ? start : b.g[j];
    }
    MPI_Datatype t;
    MPI_Type_create_subarray(MAXD, sizes, sub, st, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

inline Block setup(const Params& p, Decomp& D) {
    MPI_Comm_size(MPI_COMM_WORLD, &D.size);
    int dims[MAXD] = {0, 0, 0}, periods[MAXD] = {0, 0, 0};
    if (p.decomp == "slab") {
        dims[0] = D.size;
        for (int i = 1; i < p.d; ++i) dims[i] = 1;
    } else if (p.decomp == "block") {
        MPI_Dims_create(D.size, p.d, dims);
    } else {
        std::fprintf(stderr, "--decomp must be block or slab\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Cart_create(MPI_COMM_WORLD, p.d, dims, periods, 1, &D.cart);
    MPI_Comm_rank(D.cart, &D.rank);
    MPI_Cart_coords(D.cart, D.rank, p.d, D.coords);
    Block b;
    for (int i = 0; i < p.d; ++i) {
        D.dims[i] = dims[i];
        split(p.N(), dims[i], D.coords[i], b.n[i], b.off[i]);
        MPI_Cart_shift(D.cart, i, 1, &D.nb_lo[i], &D.nb_hi[i]);
        if (b.n[i] < 1) {
            std::fprintf(stderr, "too many ranks along dimension %d for Q=%d\n", i, p.Q);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    b.finalize(p.d);
    for (int a = 0; a < p.d; ++a) {
        D.send_lo[a] = face(b, a, 1);
        D.recv_lo[a] = face(b, a, 0);
        D.send_hi[a] = face(b, a, b.n[a]);
        D.recv_hi[a] = face(b, a, b.n[a] + 1);
    }
    return b;
}

inline void write_snapshot(const Params& p, const Decomp& D, const Block& b, const double* th, int j) {
    char name[64];
    std::snprintf(name, sizeof name, "/theta_%03d.bin", j);
    std::string path = p.out + name;
    int gsz[MAXD], lsz[MAXD], sub[MAXD], gst[MAXD], lst[MAXD];
    for (int i = 0; i < MAXD; ++i) {
        gsz[i] = i < p.d ? p.N() : 1;
        sub[i] = b.n[i];
        gst[i] = b.off[i];
        lsz[i] = int(b.ext[i]);
        lst[i] = b.g[i];
    }
    MPI_Datatype ftype, mtype;
    MPI_Type_create_subarray(MAXD, gsz, sub, gst, MPI_ORDER_C, MPI_DOUBLE, &ftype);
    MPI_Type_create_subarray(MAXD, lsz, sub, lst, MPI_ORDER_C, MPI_DOUBLE, &mtype);
    MPI_Type_commit(&ftype);
    MPI_Type_commit(&mtype);
    MPI_File f;
    MPI_File_open(D.cart, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
    MPI_File_set_size(f, 0);
    MPI_File_set_view(f, 0, MPI_DOUBLE, ftype, "native", MPI_INFO_NULL);
    MPI_File_write_all(f, th, 1, mtype, MPI_STATUS_IGNORE);
    MPI_File_close(&f);
    MPI_Type_free(&ftype);
    MPI_Type_free(&mtype);
}

// Halo traffic this rank sends per step: one message per existing face neighbour.
inline void halo_traffic(const Params& p, const Decomp& D, const Block& b, double out[2]) {
    out[0] = out[1] = 0;
    for (int i = 0; i < p.d; ++i) {
        const double face_pts = double(b.points()) / b.n[i];
        for (int nb : {D.nb_lo[i], D.nb_hi[i]})
            if (nb != MPI_PROC_NULL) { out[0] += 1; out[1] += face_pts * sizeof(double); }
    }
}
