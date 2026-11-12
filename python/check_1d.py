"""Validation 1: the 1D solver against the paper's closed form (Proposition 2)."""
import sys
import numpy as np
from hjbio import read_meta, read_theta
from reference import theta_1d_exact

out = sys.argv[1]
m = read_meta(out)
a = m["assets"][0]
ref = theta_1d_exact(m["Q"], m["T"], m["gamma"], a["A"], a["k"], a["sigma"], m["times"])
err = max(np.max(np.abs(read_theta(out, j, m) - r)) for j, r in enumerate(ref))
scale = max(np.max(np.abs(r)) for r in ref)
print("max |theta - exact| = %.3e  (max |theta| = %.3e, relative %.2e)" % (err, scale, err / scale))
