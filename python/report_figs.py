"""Report figures and tables from the benchmark CSVs.  usage: python3 report_figs.py <raw_dir> <out_dir>

Reads strong, strong3d, weak, variants, hybrid, gpu_strong, gpu_weak, gpu_vs_cpu .csv from <raw_dir>."""
import csv, os, sys, collections
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Anchor paths relative to the folder where this script lives
BASE_DIR = Path.cwd()
raw = BASE_DIR / "results" / "raw"
out = BASE_DIR / "out"
os.makedirs("./out", exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 200,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load(name):
    rows = []
    file_path = raw / f"{name}.csv"
    with open(file_path, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("ranks", "threads", "d", "Q", "points", "steps"):
                r[k] = int(r[k])
            for k in ("total_s", "compute_s", "halo_s", "reduce_s", "copy_s", "halo_bytes", "halo_msgs"):
                r[k] = float(r[k] or 0)
            r["cores"] = r["ranks"] * r["threads"]
            r["ms"] = 1e3 * r["total_s"] / r["steps"]
            rows.append(r)
    return rows

def groups(rows, key):
    g = collections.OrderedDict()
    for r in rows: g.setdefault(key(r), []).append(r)
    return g

def stats(rs):
    v = np.array([r["ms"] for r in rs]); return np.median(v), v.min(), v.max()

def median_run(rs):
    return sorted(rs, key=lambda r: r["ms"])[len(rs) // 2]

strong, s3, weak, var, hyb = (load(n) for n in ("strong", "strong3d", "weak", "variants", "hybrid"))
gs, gw, gc = load("gpu_strong"), load("gpu_weak"), load("gpu_vs_cpu")

# ---------- Fig A: strong scaling, 2D and 3D, speedup + efficiency vs 1 core ----------
def series(rows, pred):
    g = groups(sorted([r for r in rows if pred(r)], key=lambda r: r["cores"]), lambda r: r["cores"])
    c = np.array(list(g)); s = np.array([stats(v) for v in g.values()])
    return c, s
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
for j, (rows, title, extra) in enumerate([(strong, "2D, Q = 1000 (4.0 M points)", True), (s3, "3D, Q = 100 (8.1 M points)", False)]):
    t1 = stats([r for r in rows if r["cores"] == 1])[0]
    sets = [("pure MPI (blocking halo)", series(rows, lambda r: r["threads"] == 1), C[0], "o"),
            ("MPI x 36 OpenMP threads", series(rows, lambda r: r["threads"] == 36), C[1], "s")]
    if extra:
        sets.append(("pure MPI, overlapped halo", series(var, lambda r: r["variant"] == "block-overlap"), C[2], "^"))
    cmax = 288
    ax[0, j].plot([1, cmax], [1, cmax], "--", color=INK2, lw=1, label="ideal")
    ax[1, j].axhline(1, ls="--", color=INK2, lw=1)
    for lab, (c, s), col, m in sets:
        sp = t1 / s[:, 0]
        err = [sp - t1 / s[:, 2], t1 / s[:, 1] - sp]
        ax[0, j].errorbar(c, sp, yerr=err, color=col, marker=m, capsize=2, label=lab)
        ax[1, j].errorbar(c, sp / c, yerr=[e / c for e in err], color=col, marker=m, capsize=2)
    ax[0, j].set_xscale("log", base=2); ax[0, j].set_yscale("log", base=2)
    ax[0, j].set_title(title, fontsize=10, color=INK)
    ax[1, j].set_ylim(0, 1.1); ax[1, j].set_xlabel("cores (ranks x threads), 36 per socket, 72 per node")
    for a in ax[:, j]:
        a.set_xticks([1, 2, 4, 9, 18, 36, 72, 144, 288]); a.set_xticklabels(["1", "2", "4", "9", "18", "36", "72", "144", "288"])
        for x in (72, 144): a.axvline(x, color=GRID, lw=3, zorder=0)
ax[0, 0].set_ylabel("speedup  T(1 core) / T(p)"); ax[1, 0].set_ylabel("parallel efficiency")
ax[0, 0].legend(fontsize=8, loc="upper left")
ax[0, 1].legend(fontsize=8, loc="upper left")
fig.text(0.5, 0.005, "markers: median of 3 runs; bars: fastest and slowest run. Shaded lines mark 1 and 2 nodes.", ha="center", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1));
# fig.savefig(os.path.join(out, "A_strong_scaling.png"));
# plt.close(fig)

# ---------- Fig B: weak scaling ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
t1 = stats([r for r in weak if r["cores"] == 1])[0]
for lab, nt, col, m in [("pure MPI", 1, C[0], "o"), ("MPI x 36 threads", 36, C[1], "s")]:
    c, s = series(weak, lambda r, nt=nt: r["threads"] == nt)
    ax[0].errorbar(c, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], color=col, marker=m, capsize=2, label=lab)
    ax[1].plot(c, t1 / s[:, 0], color=col, marker=m, label=lab)
for a in ax:
    a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 9, 18, 36, 72, 144, 288]); a.set_xticklabels(["1", "2", "4", "9", "18", "36", "72", "144", "288"])
    a.set_xlabel("cores (250 k points per core)")
    for x in (72, 144): a.axvline(x, color=GRID, lw=3, zorder=0)
ax[0].set_ylabel("time per step [ms]"); ax[0].set_ylim(0, 18); ax[0].legend(fontsize=8)
ax[1].set_ylabel("weak efficiency  T(1) / T(p)"); ax[1].set_ylim(0, 1.1)
fig.tight_layout();
# fig.savefig(os.path.join(out, "B_weak_scaling.png"));
# plt.close(fig)

# ---------- Fig C: where the time goes (median run per config) ----------
def breakdown(a, rows, pred, title):
    g = groups(sorted([r for r in rows if pred(r)], key=lambda r: r["cores"]), lambda r: r["cores"])
    cs = list(g); runs = [median_run(v) for v in g.values()]
    tot = np.array([r["total_s"] for r in runs])
    h = np.array([r["halo_s"] for r in runs]) / tot; red = np.array([r["reduce_s"] for r in runs]) / tot
    comp = 1 - h - red
    x = np.arange(len(cs)); w = 0.72
    a.bar(x, 100 * comp, w, color=C[0], label="remainder: compute", edgecolor="white", linewidth=1)
    a.bar(x, 100 * h, w, bottom=100 * comp, color=C[1], label="halo exchange (slowest rank)", edgecolor="white", linewidth=1)
    a.bar(x, 100 * red, w, bottom=100 * (comp + h), color=C[2], label="dt allreduce incl. waiting (slowest rank)", edgecolor="white", linewidth=1)
    a.set_xticks(x); a.set_xticklabels([str(c) for c in cs]); a.set_ylim(0, 100)
    a.set_title(title, fontsize=9.5, color=INK); a.set_xlabel("cores"); a.grid(axis="x", visible=False)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
breakdown(ax[0], strong, lambda r: r["threads"] == 1 and r["cores"] > 1, "2D Q=1000, pure MPI")
breakdown(ax[1], s3, lambda r: r["threads"] == 1 and r["cores"] > 1, "3D Q=100, pure MPI")
breakdown(ax[2], s3, lambda r: r["threads"] == 36, "3D Q=100, MPI x 36 threads")
ax[0].set_ylabel("share of run time [%]")
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1));
# fig.savefig(os.path.join(out, "C_time_breakdown.png"));
# plt.close(fig)

# ---------- Fig D: design variants ----------
t1 = stats([r for r in strong if r["cores"] == 1])[0]
names = [("block-blocking", "block, blocking"), ("block-overlap", "block, overlapped"),
         ("slab-blocking", "slab, blocking"), ("slab-overlap", "slab, overlapped"),
         ("block-blocking-fixeddt", "block, fixed dt (no allreduce)")]
cores = [72, 144, 288]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
w = 0.16
for i, (v, lab) in enumerate(names):
    s = np.array([stats([r for r in var if r["variant"] == v and r["cores"] == c]) for c in cores])
    sp = t1 / s[:, 0]
    ax.bar(np.arange(3) + (i - 2) * w, sp, w * 0.9, color=C[i], label=lab,
           yerr=[sp - t1 / s[:, 2], t1 / s[:, 1] - sp], capsize=2, error_kw={"lw": 0.8, "ecolor": INK2})
for k, c in enumerate(cores):
    ax.hlines(c, k - 0.45, k + 0.45, colors=INK2, linestyles="--", lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["72 cores (1 node)", "144 cores (2 nodes)", "288 cores (4 nodes)"])
ax.set_ylabel("speedup vs 1 core"); ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
ax.text(2.47, 288, "ideal", va="center", fontsize=7.5, color=INK2)
ax.set_ylim(0, 300)
fig.tight_layout(); 
# fig.savefig(os.path.join(out, "D_variants.png"));
# plt.close(fig)

# ---------- Fig E: hybrid split ----------
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for tot_c, col, m, lab in [(72, C[0], "o", "72 cores (1 node)"), (144, C[1], "s", "144 cores (2 nodes)")]:
    g = groups(sorted([r for r in hyb if r["cores"] == tot_c], key=lambda r: r["threads"]), lambda r: r["threads"])
    th = np.array(list(g)); s = np.array([stats(v) for v in g.values()])
    ax.errorbar(th, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], color=col, marker=m, capsize=2, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 3, 4, 6, 9, 12, 18, 36, 72]); ax.set_xticklabels(["1", "2", "3", "4", "6", "9", "12", "18", "36", "72"])
ax.set_xlabel("OpenMP threads per rank (ranks = cores / threads)"); ax.set_ylabel("time per step [ms]")
ax.set_ylim(0, 5); ax.legend(fontsize=8)
fig.tight_layout();
# fig.savefig(os.path.join(out, "E_hybrid_split.png")); 
# plt.close(fig)

# ---------- Fig F: GPU vs CPU node ----------
Qs = [250, 1000, 4000]
cfg = [("CPU node, 2 x 36 cores", lambda r: not r["variant"].startswith("gpu")),
       ("1 A40", lambda r: r["variant"].startswith("gpu") and r["ranks"] == 1),
       ("2 A40 (one node)", lambda r: r["variant"].startswith("gpu") and r["ranks"] == 2)]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
w = 0.26
for i, (lab, p) in enumerate(cfg):
    thr = []
    for q in Qs:
        rs = [r for r in gc if r["Q"] == q and p(r)]
        thr.append(rs[0]["points"] / (stats(rs)[0] * 1e-3) / 1e9)
    ax.bar(np.arange(3) + (i - 1) * w, thr, w * 0.9, color=C[i], label=lab)
    for k, t in enumerate(thr):
        ax.text(k + (i - 1) * w, t + 0.03, "%.2f" % t, ha="center", fontsize=7, color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(["Q = %d\n%.2f M points" % (q, (2 * q + 1) ** 2 / 1e6) for q in Qs])
ax.set_ylabel("throughput [G point-updates / s]"); ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout();
# fig.savefig(os.path.join(out, "F_gpu_vs_cpu.png"))
# plt.close(fig)

# ---------- tables ----------
L = []
def tab(title, rows, base_ms, keyf, labf, weak=False):
    L.append("\n### " + title + "\n")
    L.append("| config | cores | median ms/step | min ms/step | speedup | efficiency | halo % | allreduce % | halo KB/rank/step |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, rs in groups(sorted(rows, key=keyf), keyf).items():
        med, mn, mx = stats(rs); c = rs[0]["cores"]; mr = median_run(rs)
        if weak: eff = base_ms / med; sp = eff * c
        else: sp = base_ms / med; eff = sp / c
        L.append("| %s | %d | %.3f | %.3f | %.1f | %.2f | %.1f | %.1f | %.1f |" % (labf(rs[0]), c, med, mn, sp, eff,
                 100 * mr["halo_s"] / mr["total_s"], 100 * mr["reduce_s"] / mr["total_s"], mr["halo_bytes"] / 1024))

lab = lambda r: "%d x %d" % (r["ranks"], r["threads"])
t1 = stats([r for r in strong if r["cores"] == 1])[0]
tab("Strong scaling, 2D Q=1000 (baseline 1 core = %.2f ms/step)" % t1, strong, t1, lambda r: (r["threads"], r["cores"]), lab)
tab("Design variants, 2D Q=1000 (same 1-core baseline)", var, t1, lambda r: (r["variant"], r["cores"]), lambda r: r["variant"] + " " + lab(r))
t13 = stats([r for r in s3 if r["cores"] == 1])[0]
tab("Strong scaling, 3D Q=100 (baseline 1 core = %.2f ms/step)" % t13, s3, t13, lambda r: (r["threads"], r["cores"]), lab)
tw = stats([r for r in weak if r["cores"] == 1])[0]
tab("Weak scaling, 2D, 250k points/core (baseline %.2f ms/step)" % tw, weak, tw, lambda r: (r["threads"], r["cores"]), lambda r: lab(r) + " Q=%d" % r["Q"], weak=True)
t72 = stats([r for r in hyb if r["ranks"] == 72 and r["threads"] == 1])[0]
tab("Hybrid split, 2D Q=1000 (speedup = 72 x T(72x1) / T, with T(72x1) = %.3f ms)" % t72, hyb, t72 * 72, lambda r: (r["cores"], r["threads"]), lab)
L.append("\n### GPU (A40, one rank per GPU), ms per step\n")
L.append("| problem | CPU node 2x36 | 1 GPU | 2 GPU | 2-GPU speedup | 1 GPU vs node | halo+copy % (2 GPU) |")
L.append("|---|---|---|---|---|---|---|")
for q in Qs:
    rs = {lab_: [r for r in gc if r["Q"] == q and p(r)] for lab_, p in cfg}
    a, b, c2 = (stats(v)[0] for v in rs.values()); mr = median_run(list(rs.values())[2])
    L.append("| 2D Q=%d | %.3f | %.3f | %.3f | %.2f | %.2f | %.1f |" % (q, a, b, c2, b / c2, a / b, 100 * (mr["halo_s"] + mr["copy_s"]) / mr["total_s"]))
for q in [100]:
    b = stats([r for r in gs if r["d"] == 3 and r["ranks"] == 1])[0]; c2 = stats([r for r in gs if r["d"] == 3 and r["ranks"] == 2])[0]
    cpu = min(stats([r for r in s3 if r["cores"] == 72 and r["threads"] == t])[0] for t in (1, 36))
    L.append("| 3D Q=100 | %.3f (best 72-core run, strong3d) | %.3f | %.3f | %.2f | %.2f | |" % (cpu, b, c2, b / c2, cpu / b))
b = stats([r for r in gw if r["ranks"] == 1])[0]; c2 = stats([r for r in gw if r["ranks"] == 2])[0]
L.append("\nGPU weak scaling (16 M points per GPU): 1 GPU %.3f ms, 2 GPU %.3f ms, efficiency %.3f\n" % (b, c2, b / c2))
open(os.path.join(out, "tables.md"), "w").write(
    "Medians over 3 runs of 200 steps. Speedup is T(1 core)/T(p) with the 1-core run of the same problem.\n"
    "Halo and allreduce shares come from the median run and are max over ranks, so they include waiting for the slowest rank.\n" + "\n".join(L) + "\n")
