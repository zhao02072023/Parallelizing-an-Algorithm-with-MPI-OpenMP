"""Report figures.

  python3 plots.py paper   <out_1d_dir>  <fig_dir>   # paper Figures 1-3 (needs many --snapshots)
  python3 plots.py quotes2d <out_2d_dir> <fig_dir>   # bid quote of asset 1 over (q1, q2) at t = 0
  python3 plots.py scaling <results.csv> <fig_dir>   # strong/weak scaling + phase breakdown
  python3 plots.py serial  <serial.csv>  <fig_dir>   # serial cost vs problem size (serial + serialfull rows)
  python3 plots.py gpu     <gpu.csv>     <fig_dir>   # gpu_vs_cpu bars (gpu_strong/gpu_weak: use scaling)
  python3 plots.py table   <results.csv> <out_dir>   # speedup / efficiency / communication tables (.md + .tex)
"""
import os
import sys
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hjbio import read_meta, read_theta, quotes_1d


def paper(out, figs):
    m = read_meta(out)
    a = m["assets"][0]
    Q = m["Q"]
    q = np.arange(-Q, Q + 1)
    times = np.array([m["T"]] + m["times"])
    thetas = [np.zeros(2 * Q + 1)] + [read_theta(out, j, m) for j in range(len(m["times"]))]
    db, da = map(np.array, zip(*[quotes_1d(th, m["gamma"], a["k"]) for th in thetas]))
    tt, qq = np.meshgrid(times, q, indexing="ij")
    for name, z, label, mask in [("fig1_bid", db, r"$s - s^b$ [Tick]", q < Q),
                                 ("fig2_ask", da, r"$s^a - s$ [Tick]", q > -Q),
                                 ("fig3_spread", db + da, r"$\psi$ [Tick]", np.abs(q) < Q)]:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(tt[:, mask], qq[:, mask], z[:, mask], cmap="viridis", linewidth=0)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Inventory"); ax.set_zlabel(label)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, name + ".png"), dpi=150)
        plt.close(fig)


def quotes2d(out, figs):
    m = read_meta(out)
    Q, g, k = m["Q"], m["gamma"], m["assets"][0]["k"]
    th = read_theta(out, len(m["times"]) - 1, m)
    db = th[:-1, :] - th[1:, :] + np.log1p(g / k) / g  # bid of asset 1, q1 < Q
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(db.T, origin="lower", extent=[-Q, Q - 1, -Q, Q], aspect="auto", cmap="viridis")
    fig.colorbar(im, label=r"asset 1 bid distance $\delta^b_1$ [Tick]")
    ax.set_xlabel("$q_1$"); ax.set_ylabel("$q_2$"); ax.set_title(r"t = 0, $\rho$ = %g" % m["rho"])
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "quotes2d_rho%g.png" % m["rho"]), dpi=150)
    plt.close(fig)


def load(csv):
    import csv as csvmod
    rows = []
    with open(csv) as f:
        for r in csvmod.DictReader(f):
            if r["experiment"] == "experiment":  # header repeated by an appended file
                continue
            for key in ("ranks", "threads", "d", "Q", "points", "steps"):
                r[key] = int(r[key])
            for key in ("total_s", "compute_s", "halo_s", "reduce_s", "io_s",
                        "halo_msgs", "halo_bytes", "copy_s"):
                r[key] = float(r.get(key) or 0)
            r["cores"] = r["ranks"] * r["threads"]
            r["tps"] = r["total_s"] / r["steps"]
            rows.append(r)
    return rows


def med(xs):
    return float(np.median(xs))


def serial(csv, figs):
    rows = load(csv)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for d in (1, 2, 3):
        rs = sorted([r for r in rows if r["experiment"] == "serial" and r["d"] == d], key=lambda r: r["points"])
        if not rs:
            continue
        pts = sorted({r["points"] for r in rs})
        t = [med([r["tps"] for r in rs if r["points"] == p]) for p in pts]
        ax[0].loglog(pts, t, "o-", label="d = %d (%d neighbours per point)" % (d, 2 * d))
        ax[1].semilogx(pts, [1e9 * x / p for x, p in zip(t, pts)], "o-", label="d = %d" % d)
    p = np.array([1e3, 1e8])
    ax[0].loglog(p, p * 3e-8, "k--", label="slope 1")
    ax[0].set_xlabel("lattice points (2Q+1)^d"); ax[0].set_ylabel("time per step [s]")
    ax[1].set_xlabel("lattice points"); ax[1].set_ylabel("ns per point per step")
    ax[1].set_ylim(bottom=0)
    full = [r for r in rows if r["experiment"] == "serialfull"]
    # gamma is not in the CSV: the serialfull sweep runs gamma 0.01 first, then 1e-4
    seen = {}
    for r in full:
        key = (r["d"], r["Q"])
        seen[key] = seen.get(key, 0) + 1
        r["gamma_run"] = seen[key]
    for d in (1, 2):
        for g, lab in ((1, "paper gamma 0.01"), (2, "benchmark gamma 1e-4")):
            rs = sorted([r for r in full if r["d"] == d], key=lambda r: r["Q"])
            reps = max([1] + [seen[(d, r["Q"])] for r in rs]) // 2  # repeats per gamma
            sel = [r for r in rs if (r["gamma_run"] - 1) // max(reps, 1) == g - 1]
            if not sel:
                continue
            Qs = sorted({r["Q"] for r in sel})
            ax[2].loglog(Qs, [med([r["steps"] for r in sel if r["Q"] == q]) for q in Qs], "o-",
                         label="d = %d, %s" % (d, lab))
    ax[2].set_xlabel("inventory limit Q"); ax[2].set_ylabel("stable time steps over T = 600 s")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    plt.show(fig)
    # fig.savefig(os.path.join(figs, "serial_scaling.png"), dpi=150)
    # plt.close(fig)


def gpu(csv, figs):
    rows = [r for r in load(csv) if r["experiment"] == "gpu_vs_cpu"]
    Qs = sorted({r["Q"] for r in rows})
    cfgs = []
    for r in rows:
        name = ("%d GPU" % r["ranks"]) if r["variant"].startswith("gpu") else ("CPU node %dx%d" % (r["ranks"], r["threads"]))
        r["cfg"] = name
        if name not in cfgs:
            cfgs.append(name)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        v = [med([r["tps"] for r in rows if r["Q"] == q and r["cfg"] == c] or [np.nan]) for q in Qs]
        ax.bar(np.arange(len(Qs)) + i * w, v, w, label=c)
    ax.set_xticks(np.arange(len(Qs)) + w * (len(cfgs) - 1) / 2)
    ax.set_xticklabels(["Q = %d\n%.1f M points" % (q, (2 * q + 1) ** 2 / 1e6) for q in Qs])
    ax.set_yscale("log"); ax.set_ylabel("time per step [s]")
    ax.legend(); ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "gpu_vs_cpu.png"), dpi=150)
    plt.close(fig)


def table(csv, outdir):
    """Speedup, efficiency and communication share per configuration, medians over repeats.
    Baseline: the run with the fewest cores (or GPUs) in the same experiment and problem
    (same d and Q; weak scaling: same d)."""
    rows = load(csv)
    md, tex = [], []
    for e in sorted({r["experiment"] for r in rows}):
        weak = "weak" in e
        cfgs = {}
        for r in rows:
            if r["experiment"] == e:
                cfgs.setdefault((r["variant"], r["threads"], r["d"], r["Q"], r["ranks"]), []).append(r)
        base = {}
        for k, c in cfgs.items():
            prob = (k[2],) if weak else (k[2], k[3])
            cores = k[1] * k[4]
            if prob not in base or cores < base[prob][1]:
                base[prob] = (med([r["tps"] for r in c]), cores)
        md.append("\n### %s\n" % e)
        md.append("| variant | d | Q | ranks x threads | time/step [ms] | speedup | efficiency | halo KB/rank/step | halo % | allreduce % | copy % |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        tex.append("%% %s\n\\begin{tabular}{lrrrrrrrrrr}\nvariant & d & Q & ranks$\\times$threads & ms/step & speedup & efficiency & halo KB & halo \\%% & allreduce \\%% & copy \\%% \\\\ \\hline" % e)
        for k in sorted(cfgs, key=lambda k: (k[2], 0 if weak else k[3], k[0], k[1], k[1] * k[4])):
            c = cfgs[k]
            variant, nt, d, Q, nr = k
            tps = med([r["tps"] for r in c])
            cores = nr * nt
            b_tps, b_cores = base[(d,) if weak else (d, Q)]
            if weak:
                eff = b_tps / tps
                sp = eff * cores / b_cores
            else:
                sp = b_tps / tps * b_cores
                eff = sp / cores
            tot = med([r["total_s"] for r in c])
            fr = lambda key: 100 * med([r[key] for r in c]) / tot
            kb = med([r["halo_bytes"] for r in c]) / 1024
            vals = (variant, d, Q, nr, nt, 1e3 * tps, sp, eff, kb, fr("halo_s"), fr("reduce_s"), fr("copy_s"))
            md.append("| %s | %d | %d | %d x %d | %.3f | %.1f | %.2f | %.1f | %.1f | %.1f | %.1f |" % vals)
            tex.append("%s & %d & %d & %d$\\times$%d & %.3f & %.1f & %.2f & %.1f & %.1f & %.1f & %.1f \\\\" % vals)
        tex.append("\\end{tabular}\n")
    name = os.path.splitext(os.path.basename(csv))[0]
    with open(os.path.join(outdir, name + "_table.md"), "w") as f:
        f.write("Speedup is relative to the run with the fewest cores (GPUs) on the same problem, times its\n"
                "core count, so a 1-core baseline gives the usual T1/Tp. Weak-scaling efficiency is\n"
                "t(smallest)/t(n). Halo / allreduce / copy are % of total time (max over ranks), so they\n"
                "include time spent waiting for slower ranks.\n" + "\n".join(md) + "\n")
    with open(os.path.join(outdir, name + "_table.tex"), "w") as f:
        f.write("\n".join(tex))


def scaling(csv, figs):
    rows = load(csv)
    exps = sorted({r["experiment"] for r in rows})
    for e in exps:
        rs = [r for r in rows if r["experiment"] == e]
        series = sorted({r["variant"] + " t%d" % r["threads"] for r in rs})
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for s in series:
            pts = {}
            for r in rs:
                if r["variant"] + " t%d" % r["threads"] == s:
                    pts.setdefault(r["cores"], []).append(r["total_s"] / r["steps"])
            c = np.array(sorted(pts))
            tps = np.array([np.median(pts[x]) for x in c])
            if e.startswith("weak"):
                eff = tps[0] / tps
                ax[0].plot(c, tps, "o-", label=s)
                ax[1].plot(c, eff, "o-", label=s)
            else:
                sp = tps[0] * c[0] / tps
                ax[0].loglog(c, sp, "o-", label=s)
                ax[1].semilogx(c, sp / c, "o-", label=s)
        if e.startswith("weak"):
            ax[0].set_ylabel("time per step [s]")
            ax[1].set_ylabel("weak-scaling efficiency")
        else:
            c = np.array(sorted({r["cores"] for r in rs}))
            ax[0].loglog(c, c / c[0] * 1.0, "k--", label="ideal")
            ax[0].set_ylabel("speedup vs %d cores" % c[0])
            ax[1].set_ylabel("parallel efficiency")
        for a in ax:
            a.set_xlabel("cores (ranks x threads)")
            a.grid(True, which="both", alpha=0.3)
        ax[1].set_ylim(0, 1.1)
        ax[0].legend(fontsize=8)
        fig.suptitle(e)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "scaling_%s.png" % e), dpi=150)
        plt.close(fig)

        # phase breakdown per configuration (median over repeats)
        cfg = {}
        for r in rs:
            cfg.setdefault((r["variant"], r["ranks"], r["threads"]), []).append(r)
        keys = sorted(cfg, key=lambda k: (k[0], k[1] * k[2], k[1]))
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(keys)), 4))
        bottom = np.zeros(len(keys))
        for ph in ("compute_s", "halo_s", "reduce_s", "io_s"):
            v = np.array([np.median([r[ph] / r["steps"] for r in cfg[k]]) for k in keys])
            ax.bar(range(len(keys)), v, bottom=bottom, label=ph[:-2])
            bottom += v
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["%s\n%dx%d" % (k[0], k[1], k[2]) for k in keys], rotation=90, fontsize=7)
        ax.set_ylabel("time per step [s] (max over ranks)")
        ax.legend()
        ax.set_title(e + ": phase breakdown")
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "phases_%s.png" % e), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    # cmd, src, figs = sys.argv[1:4]
    cmd = "serial"
    src = r"E:\Development\NUS\Projects\3\hjb-mm\python\results\raw\serialall.csv"
    figs = "figs"
    os.makedirs(figs, exist_ok=True)
    {"paper": paper, "quotes2d": quotes2d, "scaling": scaling, "serial": serial, "gpu": gpu,
     "table": table}[cmd](src, figs)
