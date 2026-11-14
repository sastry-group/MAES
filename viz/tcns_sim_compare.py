# viz/tcns_sim_compare.py
"""
Offline comparison of ergodic_sim.py sweeps (one folder per <map>_M<M>_q<q>).

Per (map, M, q), default q = 3 only (--q 3 10 for more):
  <map>_M<M>_q<q>_regret.png  regret vs K | total epochs per mode | total planning time per mode
  <map>_M<M>_belief.png       RMSE | MSLL | est_err ||rho_bar - rho*||_1, per epoch vs K
Plus summary.csv (one row per map, M, q, mode).

Regret: seed mean, band = min-max over seeds. Totals: seed mean. Belief: seed 0.
Colour = mode. Planning time = GP posterior + MH build + cap, summed over epochs.

Usage (from MAES/):
  python -m viz.tcns_sim_compare results/TCNS_sim_5x5_v --out results/TCNS_sim_5x5_v/compare
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"computed": "#2a78d6", "early": "#eb6834", "fixed100": "#7a3ec2",
          "fixed1000": "#1baf7a", "fixed10000": "#8a8a85"}
STYLES = ["-", "--", ":", "-."]
BG = "#fcfcfb"
FOLDER = re.compile(r"(?P<map>.+)_M(?P<M>\d+)_q(?P<q>[\d.]+)$")


def _style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)


def _order(modes):
    return [m for m in COLORS if m in modes] + sorted(m for m in modes if m not in COLORS)


# ---------------------------------------------------------------- load
def load_folder(folder):
    """{mode: dict(K, regret (S, nK), epochs [seed][epoch], D_u, wall_s [S])}"""
    out = {}
    for f in glob.glob(os.path.join(folder, "runs", "*.npz")):
        mode, seed = os.path.basename(f)[:-4].rsplit("_", 1)
        z = np.load(f, allow_pickle=True)
        d = out.setdefault(mode, dict(seeds={}, D_u=float(z["D_u"])))
        d["seeds"][int(seed)] = (z["regret"], json.loads(str(z["epochs"])),
                                 float(z["wall_s"]) if "wall_s" in z.files else np.nan)
    for mode, d in out.items():
        K = np.load(os.path.join(folder, f"regret_{mode}.npz"))["K"]
        seeds = [d["seeds"][s] for s in sorted(d["seeds"])]
        n = min(len(r) for r, _, _ in seeds)
        d.update(K=K[:n], regret=np.array([r[:n] for r, _, _ in seeds]),
                 epochs=[e for _, e, _ in seeds], wall_s=np.array([w for _, _, w in seeds]))
    return out


def load_all(root):
    data = {}                                   # (map, M) -> {q: {mode: ...}}
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        m = FOLDER.match(os.path.basename(folder))
        if m and os.path.isdir(os.path.join(folder, "runs")):
            data.setdefault((m["map"], int(m["M"])), {})[float(m["q"])] = load_folder(folder)
    return data


# ---------------------------------------------------------------- per-epoch helpers
def epoch_end(ep):
    return np.array([x["k_start"] + x["steps"] for x in ep])


def epoch_x(ep):
    return np.maximum([x["k_start"] for x in ep], 1)


def thin(n, keep=300):
    """Indices of about `keep` evenly spaced epochs (fixed100 has 1e5 epochs)."""
    if n <= keep:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, keep).round().astype(int))


# ---------------------------------------------------------------- plots
def _fmt_s(t):
    return f"{t * 1e3:.1f} ms" if t < 0.01 else f"{t * 1e3:.0f} ms" if t < 1 else f"{t:.1f} s"


def plot_regret(name, runs, path):
    """regret vs K | total epochs per mode | total planning time per mode (one q)."""
    modes = _order(runs)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=BG,
                           gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    a = ax[0]
    for mode in modes:
        d, c = runs[mode], COLORS.get(mode, "#52514e")
        C = d["regret"]
        a.plot(d["K"], C.mean(0), color=c, lw=2, label=mode)
        if len(C) > 1:
            a.fill_between(d["K"], C.min(0), C.max(0), color=c, alpha=0.18, lw=0)
    D_u = next(iter(runs.values()))["D_u"]
    a.axhline(D_u, color="#52514e", lw=1, ls="--")
    a.text(1.2, D_u * 1.08, "uniform coverage", color="#52514e", fontsize=8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("time step K (per agent)"); a.set_ylabel("regret (1/K) Σ ‖ρ̂ₖ − ρ*‖₁")
    a.legend(frameon=False, fontsize=8)

    # totals, seed mean: epochs (= refits) and planning time (GP posterior + MH + cap)
    n = [np.mean([len(e) for e in runs[m]["epochs"]]) for m in modes]
    t = [np.mean([sum(x["plan_s"] for x in e) for e in runs[m]["epochs"]]) for m in modes]
    cols = [COLORS.get(m, "#52514e") for m in modes]
    for b, vals, lab, fmt in ((ax[1], n, "total epochs (refits)", lambda v: f"{v:.0f}"),
                              (ax[2], t, "total planning time [s]", _fmt_s)):
        b.bar(range(len(modes)), vals, 0.7, color=cols)
        for i, v in enumerate(vals):
            b.text(i, v * 1.15, fmt(v), ha="center", va="bottom", fontsize=8, color="#52514e")
        b.set_yscale("log")
        b.set_ylim(top=b.get_ylim()[1] * 4)
        b.set_xticks(range(len(modes)), modes, rotation=25, fontsize=8)
        b.set_ylabel(lab)
    for x in ax:
        _style(x)
    for b in ax[1:]:
        b.grid(False, axis="x")
    fig.suptitle(name, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_belief(name, by_q, path):
    keys = (("rmse", "RMSE (μ vs φ*)"), ("msll", "MSLL (> 0 = overconfident)"),
            ("est_err", "‖ρ̄ₑ − ρ*‖₁ (estimation)"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), facecolor=BG)
    for j, q in enumerate(sorted(by_q)):
        for mode in _order(by_q[q]):
            ep, c = by_q[q][mode]["epochs"][0], COLORS.get(mode, "#52514e")
            idx = thin(len(ep))
            x = np.r_[[ep[i]["k_start"] for i in idx], epoch_end(ep)[-1]]
            for a, (k, _) in zip(ax, keys):
                if k in ep[0]:
                    y = np.array([ep[i][k] for i in idx])
                    # the belief is held for the whole epoch -> step, not a line between refits
                    a.step(x, np.r_[y, y[-1]], where="post", color=c, lw=1.5, ls=STYLES[j],
                           label=mode if j == 0 else None)
                    a.plot(x[:-1], y, ls="none", marker="o", ms=3, color=c)   # refit times
    for a, (k, lab) in zip(ax, keys):
        a.set_xlabel("time step K (per agent)"); a.set_ylabel(lab); _style(a)       # linear axes
        if not a.lines:
            a.text(0.5, 0.5, f"no '{k}' in these runs\n(rerun with the patched ergodic_sim.py)",
                   ha="center", va="center", transform=a.transAxes, color="#52514e")
    ax[1].axhline(0, color="#52514e", lw=0.8, ls="--")
    ax[0].legend(frameon=False, fontsize=8)
    if len(by_q) > 1:
        ax[2].legend([plt.Line2D([], [], color="#52514e", ls=STYLES[j]) for j in range(len(by_q))],
                     [f"q = {q:g}" for q in sorted(by_q)], frameon=False, fontsize=8)
    fig.suptitle(name, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder holding <map>_M<M>_q<q>/ subfolders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=float, nargs="+", default=[3.0], help="q values to plot")
    a = ap.parse_args()
    data = {k: {q: v for q, v in by_q.items() if q in a.q} for k, by_q in load_all(a.root).items()}
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit(f"no <map>_M<M>_q<q>/runs folders under {a.root}")
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for (mp, M), by_q in sorted(data.items()):
        name = f"{mp}, M = {M}"
        for q in sorted(by_q):
            plot_regret(f"{name}, q = {q:g}", by_q[q], os.path.join(a.out, f"{mp}_M{M}_q{q:g}_regret.png"))
        plot_belief(name, by_q, os.path.join(a.out, f"{mp}_M{M}_belief.png"))
        for q in sorted(by_q):
            for mode in _order(by_q[q]):
                d = by_q[q][mode]
                fin = d["regret"][:, -1]
                last = d["epochs"][0][-1]
                rows.append(dict(
                    map=mp, M=M, q=q, mode=mode, seeds=len(fin),
                    regret_mean=fin.mean(), regret_min=fin.min(), regret_max=fin.max(),
                    D_u=d["D_u"], epochs=np.mean([len(e) for e in d["epochs"]]),
                    plan_s=np.mean([sum(x["plan_s"] for x in e) for e in d["epochs"]]),
                    run_s=np.mean([sum(x["run_s"] for x in e) for e in d["epochs"]]),
                    wall_s=np.nanmean(d["wall_s"]),
                    rmse_last=last.get("rmse", np.nan), msll_last=last.get("msll", np.nan),
                    est_err_last=last["est_err"]))
        print(f"plotted {name}")

    path = os.path.join(a.out, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{'map':>8} {'M':>2} {'q':>4} {'mode':>11} {'regret':>8} {'epochs':>8} {'plan_s':>8}")
    for r in rows:
        print(f"{r['map']:>8} {r['M']:>2} {r['q']:>4g} {r['mode']:>11} "
              f"{r['regret_mean']:8.4f} {r['epochs']:8.0f} {r['plan_s']:8.2f}")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()