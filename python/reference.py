"""Exact references for the HJB solver.

theta_1d_exact: the paper's Proposition 2, v(t) = expm(-M (T - t)) 1 and theta = ln(v) / k.
theta_linear_exact: the same linearisation on a d-dimensional lattice (all k_i equal),
    dv/dt = M v with M = diag((k gamma / 2) q' Sigma q) - sum_i eta_i (shift_+i + shift_-i).
"""
import itertools
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def eta(A, k, gamma):
    return A * (1.0 + gamma / k) ** (-(1.0 + k / gamma))


def theta_1d_exact(Q, T, gamma, A, k, sigma, times):
    q = np.arange(-Q, Q + 1)
    alpha = 0.5 * k * gamma * sigma**2
    e = eta(A, k, gamma)
    M = np.diag(alpha * q**2) - e * (np.eye(2 * Q + 1, k=1) + np.eye(2 * Q + 1, k=-1))
    ones = np.ones(2 * Q + 1)
    return [np.log(scipy.linalg.expm(-M * (T - t)) @ ones) / k for t in times]


def theta_linear_exact(d, Q, T, gamma, A, k, sigma, rho, times):
    """A, sigma: per-asset sequences; k: one common value."""
    N = 2 * Q + 1
    qs = np.arange(-Q, Q + 1)
    Sigma = np.array([[(1.0 if i == j else rho) * sigma[i] * sigma[j] for j in range(d)] for i in range(d)])
    grids = np.meshgrid(*([qs] * d), indexing="ij")
    qv = np.stack([g.ravel() for g in grids], axis=1)
    diag = 0.5 * k * gamma * np.einsum("pi,ij,pj->p", qv, Sigma, qv)
    I = sp.identity(N, format="csr")
    shift = sp.diags([np.ones(N - 1), np.ones(N - 1)], [1, -1], format="csr")
    M = sp.diags(diag, format="csr")
    for i in range(d):
        ops = [I] * d
        ops[i] = shift
        K = ops[0]
        for o in ops[1:]:
            K = sp.kron(K, o, format="csr")
        M = M - eta(A[i], k, gamma) * K
    ones = np.ones(N**d)
    out = []
    for t in times:
        v = spla.expm_multiply(-M * (T - t), ones)
        out.append((np.log(v) / k).reshape((N,) * d))
    return out
