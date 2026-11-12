"""Max difference between two solver outputs (all snapshots): python3 compare.py ref_dir test_dir [rel_tol]"""
import sys
import numpy as np
from hjbio import read_meta, read_theta

ref, test = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
m = read_meta(ref)
worst = 0.0
for j in range(len(m["times"])):
    a, b = read_theta(ref, j, m), read_theta(test, j, m)
    worst = max(worst, np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))
ok = worst <= tol
print("%s max relative difference %.3e (tolerance %.0e)" % ("ok  " if ok else "FAIL", worst, tol))
sys.exit(0 if ok else 1)
