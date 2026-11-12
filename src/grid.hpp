// A rectangular block of the inventory lattice {-Q..Q}^d with one ghost layer
// in each active dimension. Storage is always 3D, row-major (dimension 2 fastest);
// inactive dimensions have extent 1 and no ghost layer.
#pragma once

#include "params.hpp"

struct Block {
    int n[MAXD] = {1, 1, 1};    // interior points owned in each dimension
    int off[MAXD] = {0, 0, 0};  // global index (0..2Q) of the first owned point
    int g[MAXD] = {0, 0, 0};    // ghost width: 1 in active dimensions
    long ext[MAXD] = {1, 1, 1}; // storage extent n + 2g
    long s[MAXD] = {0, 0, 1};   // storage strides

    void finalize(int d) {
        for (int i = 0; i < MAXD; ++i) {
            g[i] = i < d ? 1 : 0;
            ext[i] = n[i] + 2 * g[i];
        }
        s[2] = 1;
        s[1] = ext[2];
        s[0] = ext[1] * ext[2];
    }
    long size() const { 
        return ext[0] * ext[1] * ext[2]; 
    }

    long points() const { 
        return long(n[0]) * n[1] * n[2]; 
    }

    // storage index of interior point (i0, i1, i2), 0-based local coordinates for z,y,x
    long at(int i0, int i1, int i2) const {
        return (i0 + g[0]) * s[0] + (i1 + g[1]) * s[1] + (i2 + g[2]);
    }
};

// Whole lattice as one block (serial run).
inline Block whole_block(const Params& p) {
    Block b;
    for (int i = 0; i < p.d; ++i) b.n[i] = p.N();
    b.finalize(p.d);
    return b;
}

// Split N points into parts; part r gets its share of the remainder first.
inline void split(int N, int parts, int r, int& n, int& off) {
    int base = N / parts, rem = N % parts;
    n = base + (r < rem ? 1 : 0);
    off = r * base + (r < rem ? r : rem);
}
