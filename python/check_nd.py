"""Validation 3: the d-dimensional solver against exact answers.

  separable: rho = 0, any k_i -> theta(q) = sum_i theta_i(q_i), each from the 1D closed form
  linear:    all k_i equal, any rho -> sparse matrix exponential on the whole lattice (small Q only)
"""
import sys
import numpy as np
from hjbio import read_meta, read_theta
from reference import theta_1d_exact, theta_linear_exact

out, mode = sys.argv[1], sys.argv[2]
m = read_meta(out)
d, Q, T, g, assets, times = m["d"], m["Q"], m["T"], m["gamma"], m["assets"], m["times"]
if mode == "separable":
    assert m["rho"] == 0.0, "separable check needs rho = 0"
    per_asset = [theta_1d_exact(Q, T, g, a["A"], a["k"], a["sigma"], times) for a in assets]
    ref = []
    for j in range(len(times)):
        s = np.zeros((2 * Q + 1,) * d)
        for i in range(d):
            shape = [1] * d
            shape[i] = 2 * Q + 1
            s = s + per_asset[i][j].reshape(shape)
        ref.append(s)
elif mode == "linear":
    ks = {a["k"] for a in assets}
    assert len(ks) == 1, "linear check needs equal k"
    ref = theta_linear_exact(d, Q, T, g, [a["A"] for a in assets], ks.pop(),
                             [a["sigma"] for a in assets], m["rho"], times)
else:
    sys.exit("mode is separable or linear")
err = max(np.max(np.abs(read_theta(out, j, m) - r)) for j, r in enumerate(ref))
scale = max(np.max(np.abs(r)) for r in ref)
print("%s check d=%d Q=%d: max |theta - exact| = %.3e, relative %.2e" % (mode, d, Q, err, err / scale))
