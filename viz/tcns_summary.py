# viz/tcns_summary.py
"""
TCNS runs: regret, epoch lengths, belief error, and a summary.json in the same
format as ergodic_sim.py (colleague's script). One call = one (map, M, start);
runs with the same mode are treated as seeds and averaged.

Usage (from MAES/):
  python -m viz.tcns_summary "results/TCNS_5x5_1agent/*simple5x5_static_v0*_1_start44x1*" \
         --out results/TCNS_5x5/summary_v0_M1_44
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from viz.run_io import load_run, _str
from viz.belief_errors import rmse, msll, team_average

COLORS = {"computed": "#2a78d6", "early": "#eb6834", "fixed100": "#1baf7a",
          "fixed1000": "#eda100", "fixed10000": "#e87ba4"}
EPOCH_KEYS = ("e", "steps", "cap", "rho_min", "Gamma", "ell", "est_err",      # hers
              "k_start", "Me", "min_visits", "lam2", "beta")                   # extra
INT_KEYS = ("e", "steps", "cap", "k_start", "Me")
BG = "#fcfcfb"


# ---------------------------------------------------------------- per run
def mode_label(d):
    rule = _str(d.get("epoch_rule", "off"))
    return rule if rule != "off" else f"fixed{int(np.asarray(d['gp_update']))}"


def regret_curve(traj, rho_star):
    """(1/K) sum_{k<=K} ||rho_hat_k - rho*||_1, rho_hat_k = team visits over steps 1..k
    (same definition as ergodic_sim.py)."""
    T, N = traj.shape
    C = np.zeros((T, rho_star.size))
    np.add.at(C, (np.repeat(np.arange(T), N), traj.ravel()), 1.0)
    np.cumsum(C, axis=0, out=C)
    K = np.arange(1, T + 1)
    err = np.abs(C / (N * K)[:, None] - rho_star).sum(axis=1)
    return np.cumsum(err) / K


def raw_phi(d):
    """phi* per state. In softmax mode true_phi_map_time holds rho*, so use the grid."""
    keys, vals = d["state_index_map_keys"], d["state_index_map_vals"]
    v = np.empty(len(vals))
    v[vals] = d["true_phi_map"][keys[:, 0], keys[:, 1]]
    return v


def load_one(folder):
    d, _ = load_run(folder)
    traj = d["traj_states"]
    bad = (traj < 0).any(axis=1)
    T = int(np.argmax(bad)) if bad.any() else len(traj)          # drop unfilled rows
    traj = traj[:T].astype(int)

    rho_star = d["true_phi_map_time"][0].astype(float)
    rho_star = rho_star / rho_star.sum()
    reg = regret_curve(traj, rho_star)

    epochs = []
    if "epoch_e" in d:
        for i in range(len(d["epoch_e"])):
            ep = {k: float(d[f"epoch_{k}"][i]) for k in EPOCH_KEYS if f"epoch_{k}" in d}
            for k in INT_KEYS:
                if k in ep:
                    ep[k] = int(ep[k])
            epochs.append(ep)

    ok = d["belief_update_times"] >= 0
    tgt = np.tile(raw_phi(d), (int(ok.sum()), 1))
    mu, sd = d["belief_mu_map"][ok], d["belief_sigma_map"][ok]
    belief = dict(t=d["belief_update_times"][ok],
                  rmse=team_average(rmse(mu, tgt)),
                  msll=team_average(msll(mu, sd, tgt)))

    D_u = float(np.abs(1.0 / rho_star.size - rho_star).sum())
    return mode_label(d), reg, epochs, belief, D_u


# ---------------------------------------------------------------- helpers (as in ergodic_sim.py)
def checkpoints(T, n=2000):
    return np.unique(np.r_[np.round(np.logspace(0, math.log10(T), n)).astype(np.int64), T])


def last_passage(Kc, regret, eps):
    above = np.flatnonzero(regret > eps)
    if above.size == 0:
        return int(Kc[0])
    return int(Kc[above[-1] + 1]) if above[-1] + 1 < len(regret) else None


def _style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)


# ---------------------------------------------------------------- plots
def plot_regret(Kc, curves, runs, D_u, eps_list, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax, log in zip(axes, (True, False)):
        for mode, C in curves.items():
            c = COLORS.get(mode, "#52514e")
            S, m = C.shape[0], C.mean(0)
            ax.plot(Kc, m, color=c, lw=2, label=f"{mode} epochs", zorder=2)
            if S > 1:
                half = student_t.ppf(0.975, S - 1) * C.std(0, ddof=1) / math.sqrt(S)
                ax.fill_between(Kc, np.maximum(m - half, 1e-6), m + half, color=c, alpha=0.18, lw=0)
            starts = np.array([ep["k_start"] + 1 for ep in runs[mode][0][1]
                               if ep["e"] > 1 and ep["k_start"] + 1 <= Kc[-1]])
            if len(starts) > 200:
                starts = starts[np.unique(np.linspace(0, len(starts) - 1, 200).astype(int))]
            if len(starts):
                ax.scatter(starts, np.interp(starts, Kc, m), s=18, color=c, lw=0, zorder=3)
        ax.axhline(D_u, color="#52514e", lw=1, ls="--")
        ax.text(Kc[0], D_u * 1.03, "uniform coverage", color="#52514e", fontsize=8)
        for e in eps_list:
            ax.axhline(e, color="#9a9994", lw=0.8, ls=":")
            ax.text(Kc[0], e * 1.03, f"ε = {e:.3g}", color="#52514e", fontsize=8)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("time step K (per agent)")
        ax.set_ylabel("regret  (1/K) Σ ‖ρ̂ₖ − ρ*‖₁")
        ax.set_title("log-log" if log else "linear", fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_epochs(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for mode, lst in runs.items():
        ep = lst[0][1][:-1]                     # seed 0; drop the epoch truncated by the horizon
        if not ep:
            continue
        c = COLORS.get(mode, "#52514e")
        e = [x["e"] for x in ep]
        axes[0].plot(e, [x["steps"] for x in ep], color=c, lw=2, marker="o", ms=4, label=mode)
        axes[1].plot(e, [x["est_err"] for x in ep], color=c, lw=2, marker="o", ms=4, label=mode)
    for ax, yl in zip(axes, ("epoch length (steps per agent)", "‖ρ̄ₑ − ρ*‖₁ (est_err)")):
        ax.set_yscale("log"); ax.set_xlabel("epoch e"); ax.set_ylabel(yl); _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_belief(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for mode, lst in runs.items():
        b = lst[0][2]                           # seed 0
        c = COLORS.get(mode, "#52514e")
        axes[0].plot(b["t"], b["rmse"], color=c, lw=2, marker="o", ms=3, label=mode)
        axes[1].plot(b["t"], b["msll"], color=c, lw=2, marker="o", ms=3, label=mode)
    for ax, yl in zip(axes, ("RMSE (μ vs φ*)", "MSLL (> 0 = overconfident)")):
        ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("time step"); ax.set_ylabel(yl); _style(ax)
    axes[1].axhline(0, color="#52514e", lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders or glob patterns (one map/M/start)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    folders = sorted({f for p in a.runs for f in glob.glob(p) if os.path.isdir(f)})
    if not folders:
        raise SystemExit("no run folders matched")
    runs, D_u = {}, None
    for f in folders:
        mode, reg, ep, bel, D_u = load_one(f)
        runs.setdefault(mode, []).append((reg, ep, bel))
        print(f"{mode:>11}: {os.path.basename(f)}  T={len(reg)}  epochs={len(ep)}")
    order = [m for m in COLORS if m in runs] + sorted(m for m in runs if m not in COLORS)
    runs = {m: runs[m] for m in order}

    T = min(len(r[0]) for lst in runs.values() for r in lst)
    Kc = checkpoints(T)
    eps_list = [0.5 * D_u, 0.1 * D_u]
    os.makedirs(a.out, exist_ok=True)

    summary, curves = dict(D_u=D_u, eps=eps_list, modes={}), {}
    for mode, lst in runs.items():
        C = np.array([r[0][:T] for r in lst])[:, Kc - 1]
        curves[mode] = C
        m = C.mean(0)
        summary["modes"][mode] = dict(
            final_regret=float(m[-1]), n_seeds=len(lst),
            K_eps={f"{e:.4g}": last_passage(Kc, m, e) for e in eps_list},
            epochs_seed0=lst[0][1])
        np.savez(os.path.join(a.out, f"regret_{mode}.npz"), K=Kc, regret=C)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_regret(Kc, curves, runs, D_u, eps_list, os.path.join(a.out, "regret.png"))
    plot_epochs(runs, os.path.join(a.out, "epoch_lengths.png"))
    plot_belief(runs, os.path.join(a.out, "belief_error.png"))
    print(json.dumps({m: {k: v for k, v in d.items() if k != "epochs_seed0"}
                      for m, d in summary["modes"].items()}, indent=2))
    print(f"D_u (uniform baseline) = {D_u:.4f}\nsaved to {a.out}")


if __name__ == "__main__":
    main()