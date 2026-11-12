# viz/tcns_summary.py
"""
TCNS runs: regret, epoch lengths, belief error, and a summary.json in the same
format as ergodic_sim.py (colleague's script). One call = one (map, M, start);
runs with the same mode are treated as seeds and averaged.

Also reads ergodic_sim.py output folders (the ones with runs/*.npz), so both codes
give the same plots. Do not mix the two codes in one call: same-named modes get averaged.

Usage (from MAES/):
  python -m viz.tcns_summary "results/TCNS_5x5_1_test/*5x5_static_v0_*_1_start44x1_*" \
         --out results/TCNS_5x5/summary_v0_M1_44_test
  python -m viz.tcns_summary test/results/v0_M1_44 test/results/v0_M1_44_fixed1000 \
         --out test/results/summary_v0_M1_44_test
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


def regret_curve(traj, rho_star, chunk=200_000):
    """(1/K) sum_{k<=K} ||rho_hat_k - rho*||_1, rho_hat_k = team visits over steps 1..k
    (same definition as ergodic_sim.py). Chunked: memory O(chunk x R)."""
    T, N = traj.shape
    R = rho_star.size
    counts, cum, out = np.zeros(R), 0.0, np.empty(T)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        C = np.zeros((e - s, R))
        np.add.at(C, (np.repeat(np.arange(e - s), N), traj[s:e].ravel()), 1.0)
        np.cumsum(C, axis=0, out=C)
        C += counts
        K = np.arange(s + 1, e + 1)
        c = np.cumsum(np.abs(C / (N * K)[:, None] - rho_star).sum(axis=1)) + cum
        out[s:e] = c / K
        cum, counts = c[-1], C[-1].copy()
    return out


def decomposition(traj, rho_star, rho_bar, starts, chunk=200_000):
    """Regret <= tracking + estimation (triangle inequality through the current target):
         tracking(K)   = (1/K) sum_k ||rho_hat_k   - rho_bar_k||_1   (ergodic / mixing error)
         estimation(K) = (1/K) sum_k ||rho_bar_k  - rho*||_1         (belief error)
    rho_bar: (n_snap, R) targets; starts[i] = first step index at which snapshot i is active."""
    T, N = traj.shape
    R = rho_star.size
    est_rows = np.abs(rho_bar - rho_star).sum(axis=1)
    counts, ct, ce = np.zeros(R), 0.0, 0.0
    track, est = np.empty(T), np.empty(T)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        C = np.zeros((e - s, R))
        np.add.at(C, (np.repeat(np.arange(e - s), N), traj[s:e].ravel()), 1.0)
        np.cumsum(C, axis=0, out=C)
        C += counts
        K = np.arange(s + 1, e + 1)
        idx = np.searchsorted(starts, np.arange(s, e), side="right") - 1
        tr = np.cumsum(np.abs(C / (N * K)[:, None] - rho_bar[idx]).sum(axis=1)) + ct
        es = np.cumsum(est_rows[idx]) + ce
        track[s:e], est[s:e] = tr / K, es / K
        ct, ce, counts = tr[-1], es[-1], C[-1].copy()
    return track, est


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
    t_up = d["belief_update_times"][ok]
    snaps = d["belief_phi_map"][ok].astype(float)                     # (n, N, R)
    snaps = snaps / np.nansum(snaps, axis=2, keepdims=True)
    rho_bar = np.nanmean(snaps, axis=1)                                # team target per snapshot
    starts = np.where(t_up == 0, 0, t_up + 1)                          # refit at t1-1 -> active from t1
    track, est = decomposition(traj, rho_star, rho_bar, starts)

    tgt = np.tile(raw_phi(d), (int(ok.sum()), 1))
    mu, sd = d["belief_mu_map"][ok], d["belief_sigma_map"][ok]
    belief = dict(t=d["belief_update_times"][ok],
                  rmse=team_average(rmse(mu, tgt)),
                  msll=team_average(msll(mu, sd, tgt)))

    D_u = float(np.abs(1.0 / rho_star.size - rho_star).sum())
    return mode_label(d), np.arange(1, T + 1), reg, epochs, belief, (track, est), D_u


def is_hers(folder):
    return os.path.isdir(os.path.join(folder, "runs"))


def load_hers(folder):
    """ergodic_sim.py folder: one entry per runs/<mode>_<seed>.npz (values at log-spaced K)."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "runs", "*.npz"))):
        z = np.load(p, allow_pickle=True)
        if "K" not in z:
            raise SystemExit(f"{p}: no K/tracking saved; rerun with the updated ergodic_sim_1.py")
        ep = json.loads(str(z["epochs"]))
        mode = os.path.basename(p).rsplit("_", 1)[0]
        if mode == "fixed":
            mode = f"fixed{int(ep[0]['cap'])}"
        nan = float("nan")
        belief = dict(t=np.array([x["k_start"] for x in ep]),
                      rmse=np.array([x.get("rmse", nan) for x in ep]),
                      msll=np.array([x.get("msll", nan) for x in ep]))
        out.append((mode, z["K"], z["regret"], ep, belief,
                    (z["tracking"], z["estimation"]), float(z["D_u"])))
    return out


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
                ax.fill_between(Kc, C.min(0), C.max(0), color=c, alpha=0.18, lw=0)   # range across runs
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
    """Left: epoch length per epoch. Right: est_err against time (epoch start step)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for mode, lst in runs.items():
        ep = lst[0][1][:-1]                     # seed 0; drop the epoch truncated by the horizon
        if not ep:
            continue
        c = COLORS.get(mode, "#52514e")
        axes[0].plot([x["e"] for x in ep], [x["steps"] for x in ep],
                     color=c, lw=2, marker="o", ms=4, label=mode)
        axes[1].plot([max(x["k_start"], 1) for x in ep], [x["est_err"] for x in ep],
                     color=c, lw=2, marker="o", ms=4, label=mode)
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch e")
    axes[0].set_ylabel("epoch length (steps per agent)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("time step K (epoch start)")
    axes[1].set_ylabel("‖ρ̄ₑ − ρ*‖₁ (est_err)")
    for ax in axes:
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_compute(runs, path):
    """Left: planning time per epoch (GP refit + target + MH + cap). Right: cumulative planning
    time vs time step, i.e. total compute spent on refits. Needs plan_s in the epoch log."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    drew = False
    for mode, lst in runs.items():
        ep = [x for x in lst[0][1] if "plan_s" in x]          # seed 0
        if not ep:
            continue
        drew = True
        c = COLORS.get(mode, "#52514e")
        ms = [1e3 * x["plan_s"] for x in ep]
        axes[0].plot([x["e"] for x in ep], ms, color=c, lw=1.6, marker="o", ms=3, label=mode)
        axes[1].step([max(x["k_start"], 1) for x in ep], np.cumsum(ms) / 1e3, where="post",
                     color=c, lw=2, label=f"{mode} ({len(ep)} refits)")
    if not drew:
        plt.close(fig); return
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch e"); axes[0].set_ylabel("planning time per epoch (ms)")
    axes[1].set_xscale("log"); axes[1].set_xlabel("time step K"); axes[1].set_ylabel("cumulative planning time (s)")
    for ax in axes:
        _style(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_decomposition(Kc, curves, dec, path):
    """One panel per mode: regret and its two terms (mean over runs)."""
    modes = list(curves)
    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.8), facecolor=BG,
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        c = COLORS.get(mode, "#52514e")
        tr, es = dec[mode]
        ax.plot(Kc, curves[mode].mean(0), color=c, lw=2.2, label="regret")
        ax.plot(Kc, tr.mean(0), color=c, lw=1.6, ls="--", label="tracking  ‖ρ̂ − ρ̄‖₁")
        ax.plot(Kc, es.mean(0), color=c, lw=1.6, ls=":", label="estimation  ‖ρ̄ − ρ*‖₁")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(mode, fontsize=10)
        ax.set_xlabel("time step K (per agent)"); _style(ax)
    axes[0][0].set_ylabel("time average (1/K) Σ")
    axes[0][0].legend(frameon=False, fontsize=8)
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
    entries = []
    for f in folders:
        for ent in (load_hers(f) if is_hers(f) else [load_one(f)]):
            entries.append(ent)
            print(f"{ent[0]:>11}: {os.path.basename(f)}  T={int(ent[1][-1])}  epochs={len(ent[3])}")
    D_u = float(np.mean([ent[6] for ent in entries]))   # one map: its D_u; several maps: the average
    T = min(int(ent[1][-1]) for ent in entries)
    Kc = checkpoints(T)
    on_kc = lambda K, v: np.interp(Kc, K, v)             # common grid (exact for per-step runs)
    runs = {}
    for mode, K, reg, ep, bel, (trk, est), _ in entries:
        runs.setdefault(mode, []).append((on_kc(K, reg), ep, bel, (on_kc(K, trk), on_kc(K, est))))
    order = [m for m in COLORS if m in runs] + sorted(m for m in runs if m not in COLORS)
    runs = {m: runs[m] for m in order}

    eps_list = [0.5 * D_u, 0.1 * D_u]
    os.makedirs(a.out, exist_ok=True)

    summary, curves, dec = dict(D_u=D_u, eps=eps_list, modes={}), {}, {}
    for mode, lst in runs.items():
        C = np.array([r[0] for r in lst])
        curves[mode] = C
        dec[mode] = (np.array([r[3][0] for r in lst]), np.array([r[3][1] for r in lst]))
        m = C.mean(0)
        summary["modes"][mode] = dict(
            final_regret=float(m[-1]), n_seeds=len(lst),
            final_tracking=float(dec[mode][0].mean(0)[-1]),
            final_estimation=float(dec[mode][1].mean(0)[-1]),
            K_eps={f"{e:.4g}": last_passage(Kc, m, e) for e in eps_list},
            epochs_seed0=lst[0][1])
        np.savez(os.path.join(a.out, f"regret_{mode}.npz"), K=Kc, regret=C)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_regret(Kc, curves, runs, D_u, eps_list, os.path.join(a.out, "regret.png"))
    plot_epochs(runs, os.path.join(a.out, "epoch_lengths.png"))
    plot_belief(runs, os.path.join(a.out, "belief_error.png"))
    plot_decomposition(Kc, curves, dec, os.path.join(a.out, "decomposition.png"))
    plot_compute(runs, os.path.join(a.out, "compute.png"))
    print(json.dumps({m: {k: v for k, v in d.items() if k != "epochs_seed0"}
                      for m, d in summary["modes"].items()}, indent=2))
    print(f"D_u (uniform baseline) = {D_u:.4f}\nsaved to {a.out}")


if __name__ == "__main__":
    main()