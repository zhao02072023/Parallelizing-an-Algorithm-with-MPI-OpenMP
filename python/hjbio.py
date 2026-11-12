"""Read solver output: meta.txt + theta_XXX.bin snapshots."""
import os
import numpy as np


def read_meta(out):
    meta = {"assets": [], "times": []}
    with open(os.path.join(out, "meta.txt")) as f:
        for line in f:
            w = line.split()
            if w[0] == "asset":
                meta["assets"].append({w[2]: float(w[3]), w[4]: float(w[5]), w[6]: float(w[7])})
            elif w[0] == "snapshot":
                meta["times"].append(float(w[2]))
            elif w[0] in ("d", "Q", "ranks", "threads"):
                meta[w[0]] = int(w[1])
            else:
                meta[w[0]] = float(w[1])
    return meta


def read_theta(out, j, meta=None):
    meta = meta or read_meta(out)
    N = 2 * meta["Q"] + 1
    a = np.fromfile(os.path.join(out, "theta_%03d.bin" % j), dtype=np.float64)
    return a.reshape((N,) * meta["d"])


def quotes_1d(theta, gamma, k):
    """Optimal bid / ask distances (ticks) from theta on q = -Q..Q.
    delta_b(q) = theta(q) - theta(q+1) + ln(1 + gamma/k)/gamma, undefined at q = Q (nan)."""
    c = np.log1p(gamma / k) / gamma
    db = np.full_like(theta, np.nan)
    da = np.full_like(theta, np.nan)
    db[:-1] = theta[:-1] - theta[1:] + c
    da[1:] = theta[1:] - theta[:-1] + c
    return db, da
