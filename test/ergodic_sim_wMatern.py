"""Multi-agent ergodic coverage simulator: GP-UCB belief + Metropolis-Hastings epochs.

Grid with no-fly zones, one "mountain" (values near +3) and one "valley" (near -3),
M agents running the degree-based MH chain targeting the epoch belief

    rho_bar_e = (1 - gamma) * softmax(eta * (mu_{e-1} + beta_e * sigma_{e-1})) + gamma / R,

with an exact full GP posterior computed from per-cell counts and sums.
Regret is the full-horizon time average (1/K) sum_k ||rho_hat_k - rho*||_1, where
rho_hat_k is the team-average visitation over steps 1..k.

Epoch-length modes
  fixed    : n_e = --fixed-len for every epoch (outside the theory)
  computed : n_e = ceil(2 M_e / rho_min + Gamma * ell / (2 rho_min^2)) from the
             actual min of rho_bar_e, the exact Gamma, and ell_e
  early    : same cap, but the epoch ends once every cell has M_e pooled visits

Usage
  python ergodic_sim.py --steps 200000 --seeds 5 --modes computed early --out results
  python ergodic_sim.py --steps 100000000 --seeds 20 --workers 8 --out results_1e8

for st in "0,4;0,4;0,4" "4,4;4,4;4,4"; do
  python test/ergodic_sim.py --config configs/simple5x5_static_v0.json --agents 3 --starts "$st" \
    --sigma 0.5 --steps 100000 --seeds 1 --modes computed early \
    --out "results/v0_M3_${st//[,;]/}"
done
python test/ergodic_sim.py --config configs/simple10x10_static_v3.json --agents 1 --starts 9,0 \
  --sigma 0.5 --steps 1000000000  --seeds 1 --modes computed early --out results/v3_10x10_M1

for L in 1000 10000; do
  python test/ergodic_sim.py --config configs/simple10x10_static_v3.json --agents 1 --starts 9,0 \
    --sigma 0.5 --steps 1000000000 --seeds 1 --modes fixed --fixed-len $L --out results/v3_10x10_M1_fixed$L
done




Each run keeps only a running sum of the per-step error and records the regret at
about --checkpoints log-spaced values of K, so memory stays small at any horizon.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------- environment
def build_grid(n, nofly):
    """Cells of an n x n grid minus no-fly cells; 4-neighbor adjacency."""
    nofly = set(map(tuple, nofly))
    cells = [(i, j) for i in range(n) for j in range(n) if (i, j) not in nofly]
    idx = {c: k for k, c in enumerate(cells)}
    nbrs = []
    for (i, j) in cells:
        nb = [idx[(i + di, j + dj)] for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if (i + di, j + dj) in idx]
        nbrs.append(nb)
    # connectivity check
    seen, stack = {0}, [0]
    while stack:
        for t in nbrs[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    if len(seen) != len(cells):
        raise ValueError("no-fly zones disconnect the grid")
    return np.array(cells, float), nbrs


def mountain_valley(coords, mountain, valley, width=1.0, vmax=3.0):
    """Smooth field with one peak and one pit, rescaled to exactly [-vmax, vmax]."""
    d_m = ((coords - np.array(mountain, float)) ** 2).sum(1)
    d_v = ((coords - np.array(valley, float)) ** 2).sum(1)
    g = np.exp(-d_m / (2 * width ** 2)) - np.exp(-d_v / (2 * width ** 2))
    return vmax * (2 * (g - g.min()) / (g.max() - g.min()) - 1)

def field_from_json(path):
    """Grid + phi* from a MAES static config (same rules as env/grid.py):
    base 1.0, each ROI rect [r0:r1, c0:c1] adds its weight, obstacles = no-fly."""
    cfg = json.load(open(path))
    H, W = cfg["grid"]["height"], cfg["grid"]["width"]
    assert H == W, "build_grid assumes a square grid"
    phi = np.ones((H, W))
    for roi in cfg.get("rois", []):
        r0, c0, r1, c1 = roi["shape"]["rect"]
        phi[r0:r1, c0:c1] += roi.get("weight", 1.0)
    nofly = [tuple(c) for c in cfg.get("obstacles", {}).get("cells", [])]
    return H, nofly, phi


def se_kernel(coords, lengthscale, variance):
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    return variance * np.exp(-d2 / (2 * lengthscale ** 2))


def matern32_kernel(coords, lengthscale, variance):
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    s = math.sqrt(3.0) * d / lengthscale
    return variance * (1.0 + s) * np.exp(-s)


# ---------------------------------------------------------------- GP (exact)
class CellGP:
    """Exact GP posterior from per-cell counts and sums (identical to the full-data GP)."""

    def __init__(self, K, sigma):
        self.K, self.s2 = K, sigma ** 2
        self.count = np.zeros(len(K))
        self.total = np.zeros(len(K))

    def observe(self, cells, ys):
        np.add.at(self.count, cells, 1)
        np.add.at(self.total, cells, ys)

    def posterior(self):
        V = np.flatnonzero(self.count)
        if V.size == 0:
            return np.zeros(len(self.K)), np.sqrt(np.diag(self.K))
        Kvv = self.K[np.ix_(V, V)] + np.diag(self.s2 / self.count[V])
        Ksv = self.K[:, V]
        c = cho_factor(Kvv, lower=True)
        mu = Ksv @ cho_solve(c, self.total[V] / self.count[V])
        var = np.diag(self.K) - np.einsum("ij,ji->i", Ksv, cho_solve(c, Ksv.T))
        return mu, np.sqrt(np.maximum(var, 0.0))


# ---------------------------------------------------------------- chain
def mh_kernel(pi, nbrs):
    """Degree-based Metropolis-Hastings kernel targeting pi."""
    R = len(pi)
    P = np.zeros((R, R))
    for r in range(R):
        for t in nbrs[r]:
            P[r, t] = min(1 / len(nbrs[r]), pi[t] / (pi[r] * len(nbrs[t])))
        P[r, r] = 1 - P[r].sum()
    return P


def gamma_constant(P, pi):
    """Gamma = (1 + max(0, lambda_r)) / (1 - max(0, lambda_r)), lambda_r second-largest eigenvalue."""
    d = np.sqrt(pi)
    S = d[:, None] * P / d[None, :]
    lam = np.sort(np.linalg.eigvalsh((S + S.T) / 2))[-2]
    m = max(0.0, lam)
    return (1 + m) / (1 - m)


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()



# ---------------------------------------------------------------- compiled inner loop
@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _run_epoch(x, cum, f, sigma, visits, counts, totals, rho_star, rho_bar, cap, steps_left,
               k, cum_err, Kc, reg, err_k, trk_k, ci, early, Me):
    """Advance all agents through one epoch. Returns (k, steps taken, cum_err, ci)."""
    R = cum.shape[0]
    M = x.shape[0]
    ep = np.zeros(R)
    t = 0
    while t < cap and t < steps_left:
        for a in range(M):
            u = np.random.random()
            row = cum[x[a]]
            j = 0
            while j < R - 1 and row[j] < u:      # first j with cum[j] >= u
                j += 1
            x[a] = j
            counts[j] += 1.0
            totals[j] += f[j] + sigma * np.random.standard_normal()
            visits[j] += 1.0
            ep[j] += 1.0
        k += 1
        t += 1
        inv = 1.0 / (M * k)
        s = 0.0
        for r in range(R):
            s += abs(visits[r] * inv - rho_star[r])
        cum_err += s
        if ci < Kc.shape[0] and k == Kc[ci]:
            reg[ci] = cum_err / k
            err_k[ci] = s                          # ||rho_hat_k - rho*||_1 (instantaneous)
            s2 = 0.0
            for r in range(R):
                s2 += abs(visits[r] * inv - rho_bar[r])
            trk_k[ci] = s2                         # ||rho_hat_k - rho_bar_e||_1 (tracking)
            ci += 1
        if early:
            mn = ep[0]
            for r in range(1, R):
                if ep[r] < mn:
                    mn = ep[r]
            if mn >= Me:
                break
    return k, t, cum_err, ci


# ---------------------------------------------------------------- run
def run(cfg, mode, seed):
    rng = np.random.default_rng(seed)
    _seed(seed)                                   # seeds the compiled loop's generator
    # coords, nbrs = build_grid(cfg["n"], cfg["nofly"])
    if cfg.get("config"):                                   # MAES map
        n, nofly, phi = field_from_json(cfg["config"])
        coords, nbrs = build_grid(n, nofly)
        f = np.array([phi[int(i), int(j)] for i, j in coords])
    else:                                                   # original mountain/valley
        coords, nbrs = build_grid(cfg["n"], cfg["nofly"])
        f = mountain_valley(coords, cfg["mountain"], cfg["valley"], cfg["width"])

    R = len(coords)
    # f = mountain_valley(coords, cfg["mountain"], cfg["valley"], cfg["width"])
    eta = math.log(cfg["q"]) / (f.max() - f.min())
    gam = cfg["gamma"]
    rho_star = (1 - gam) * softmax(eta * f) + gam / R
    kfun = matern32_kernel if cfg.get("kernel") == "matern32" else se_kernel
    gp = CellGP(kfun(coords, cfg["lengthscale"], cfg["kvar"]), cfg["sigma"])
    fixed_len = int(mode[5:]) if mode.startswith("fixed") and len(mode) > 5 else cfg["fixed_len"]
    M, steps, alpha = cfg["agents"], cfg["steps"], cfg["alpha"]
    d1, d2 = cfg["delta1"], cfg["delta2"]

    # x = rng.integers(0, R, M).astype(np.int64)   # agent positions
    if cfg.get("starts"):                                   # same starts as MAP_STARTS
        idx = {tuple(map(int, c)): k for k, c in enumerate(coords)}
        st = [idx[tuple(map(int, s.split(",")))] for s in cfg["starts"].split(";")]
        x = np.array([st[a % len(st)] for a in range(M)], dtype=np.int64)
    else:
        x = rng.integers(0, R, M).astype(np.int64)          # original: random starts
    visits = np.zeros(R)                         # team visits, steps 1..k
    Kc = checkpoints(steps, cfg["checkpoints"])  # K values at which regret is recorded
    reg = np.empty(len(Kc))
    err_k = np.full(len(Kc), np.nan)
    trk_k = np.full(len(Kc), np.nan)
    # warm-up: compile / load the cached loop so epoch 1's run_s is not compile time
    _run_epoch(np.zeros(1, np.int64), np.ones((1, 1)), np.zeros(1), 0.0, np.zeros(1), np.zeros(1),
               np.zeros(1), np.zeros(1), np.zeros(1), 0, 0, 0, 0.0, Kc, np.zeros(len(Kc)),
               np.zeros(len(Kc)), np.zeros(len(Kc)), 0, False, 1.0)
    ci, cum_err = 0, 0.0                         # running sum of ||rho_hat_k - rho*||_1
    epochs = []
    k, e = 0, 0
    while k < steps:
        e += 1
        t0 = time.perf_counter()
        mu, sd = gp.posterior()
        beta = math.sqrt(2 * math.log(R * math.pi ** 2 * e * e / (6 * d1)))
        rho_bar = (1 - gam) * softmax(eta * (mu + beta * sd)) + gam / R
        P = mh_kernel(rho_bar, nbrs)
        cum = np.cumsum(P, 1)
        rmin = rho_bar.min()
        Me = e ** alpha
        if mode.startswith("fixed"):              # constant epoch length, no visitation guarantee
            G, ell = float("nan"), float("nan")
            cap = fixed_len
        else:
            G = gamma_constant(P, rho_bar)
            dm = (6 * d2 / (math.pi ** 2 * e * e)) ** (1 / M)
            ell = math.log(R / rmin / dm)
            cap = math.ceil(2 * Me / rmin + G * ell / (2 * rmin ** 2))
        t1 = time.perf_counter()
        k0, before = k, gp.count.copy()
        k, t, cum_err, ci = _run_epoch(x, cum, f, cfg["sigma"], visits, gp.count, gp.total, rho_star,
                                        rho_bar, int(cap), steps - k, k, cum_err, Kc, reg, err_k, trk_k,
                                        ci, mode == "early", float(Me))
        t2 = time.perf_counter()
        epochs.append(dict(e=e, k_start=k0, steps=t, cap=int(cap), Me=float(Me),
                           min_visits=float((gp.count - before).min()),
                           rho_min=float(rmin), Gamma=float(G), ell=float(ell),
                           est_err=float(np.abs(rho_bar - rho_star).sum()),
                           rmse=float(np.sqrt(np.mean((mu - f) ** 2))),
                           plan_s=t1 - t0, run_s=t2 - t1))
    D_u = float(np.abs(1 / R - rho_star).sum())
    return dict(regret=reg, err=err_k, trk=trk_k), epochs, D_u


def checkpoints(steps, n):
    """About n log-spaced integer step counts in [1, steps], always including steps."""
    return np.unique(np.r_[np.round(np.logspace(0, math.log10(steps), n)).astype(np.int64), steps])


def _task(args):
    cfg, mode, seed = args
    t0 = time.time()
    reg, ep, D_u = run(cfg, mode, seed)
    return mode, seed, reg, ep, D_u, time.time() - t0


def last_passage(Kc, regret, eps):
    """First checkpoint K after which regret stays <= eps through the end of the run (None if never).
    Resolution is the checkpoint spacing (about 0.4% of K with 5000 checkpoints over 8 decades)."""
    above = np.flatnonzero(regret > eps)
    if above.size == 0:
        return int(Kc[0])
    return int(Kc[above[-1] + 1]) if above[-1] + 1 < len(regret) else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--nofly", type=str, default="[[1,3],[2,3],[3,1]]")
    ap.add_argument("--mountain", type=str, default="[0,0]")
    ap.add_argument("--valley", type=str, default="[4,4]")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--q", type=float, default=3.0, help="target best/worst visit ratio")
    ap.add_argument("--gamma", type=float, default=0.0, help="uniform mixing weight")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lengthscale", type=float, default=1.5)
    ap.add_argument("--kvar", type=float, default=4.0)
    ap.add_argument("--kernel", choices=["se", "matern32"], default="se")
    ap.add_argument("--delta1", type=float, default=0.05)
    ap.add_argument("--delta2", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200000, help="time steps per agent")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["computed", "early"],
                    help="any of: computed, early, fixed, fixed<L> (e.g. fixed100 fixed1000)")
    ap.add_argument("--fixed-len", dest="fixed_len", type=int, default=1000,
                    help="epoch length (steps per agent) for --modes fixed")
    ap.add_argument("--eps", type=float, nargs="+", default=None,
                    help="error targets; default 0.1 * D_u and 0.5 * D_u")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one run each)")
    ap.add_argument("--checkpoints", type=int, default=5000, help="log-spaced K values stored per run")
    ap.add_argument("--config", type=str, default=None,
                    help="MAES static JSON; overrides --n/--nofly/--mountain/--valley")
    ap.add_argument("--starts", type=str, default=None,
                    help='start cells "r,c;r,c" (default: random)')   
    a = ap.parse_args()
    cfg = dict(vars(a))
    cfg["nofly"] = json.loads(a.nofly)
    cfg["mountain"] = json.loads(a.mountain)
    cfg["valley"] = json.loads(a.valley)
    os.makedirs(a.out, exist_ok=True)

    # each finished run is saved under <out>/runs so an interrupted job resumes where it stopped
    run_dir = os.path.join(a.out, "runs")
    os.makedirs(run_dir, exist_ok=True)

    def run_path(mode, sd):
        return os.path.join(run_dir, f"{mode}_{sd}.npz")

    out = {}
    tasks = []
    for mode in a.modes:
        for sd in range(a.seeds):
            p = run_path(mode, sd)
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                curves = {c: (z[c] if c in z.files else np.full(len(z["regret"]), np.nan))
                          for c in ("regret", "err", "trk")}
                out[(mode, sd)] = (curves, json.loads(str(z["epochs"])))
                D_u = float(z["D_u"])
            else:
                tasks.append((cfg, mode, sd))
    if out:
        print(f"resuming: {len(out)} runs already saved, {len(tasks)} to go", flush=True)

    def record(mode, sd, curves, ep, D, dt):
        out[(mode, sd)] = (curves, ep)
        np.savez(run_path(mode, sd), **curves, epochs=json.dumps(ep), D_u=D, wall_s=dt)
        print(f"{mode} seed {sd}: {len(ep)} epochs, final regret {curves['regret'][-1]:.4f} "
              f"({dt / 60:.1f} min; {len(out)}/{len(a.modes) * a.seeds} done, "
              f"{(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)

    t_start = time.time()
    if a.workers > 1 and tasks:
        with Pool(a.workers) as pool:
            for mode, sd, reg, ep, D_u, dt in pool.imap_unordered(_task, tasks):
                record(mode, sd, reg, ep, D_u, dt)
    else:
        for task in tasks:
            mode, sd, reg, ep, D_u, dt = _task(task)
            record(mode, sd, reg, ep, D_u, dt)
    Kc = checkpoints(a.steps, a.checkpoints)
    results = {m: (np.array([out[(m, sd)][0]["regret"] for sd in range(a.seeds)]),
                   [out[(m, sd)][1] for sd in range(a.seeds)]) for m in a.modes}
    eps_list = a.eps or [0.5 * D_u, 0.1 * D_u]

    summary = dict(D_u=D_u, eps=eps_list, modes={})
    for mode, (C, logs) in results.items():
        mean = C.mean(0)
        summary["modes"][mode] = dict(
            final_regret=float(mean[-1]),
            K_eps={f"{eps:.4g}": last_passage(Kc, mean, eps) for eps in eps_list},
            epochs_seed0=logs[0])
        np.savez(os.path.join(a.out, f"regret_{mode}.npz"), K=Kc, regret=C,
                 err=np.array([out[(mode, sd)][0]["err"] for sd in range(a.seeds)]),
                 trk=np.array([out[(mode, sd)][0]["trk"] for sd in range(a.seeds)]))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({m: {k: v for k, v in d.items() if k != "epochs_seed0"}
                      for m, d in summary["modes"].items()}, indent=2))
    print(f"D_u (uniform baseline) = {D_u:.4f}")
    plot(Kc, results, D_u, eps_list, os.path.join(a.out, "regret.png"))


def plot(Kc, results, D_u, eps_list, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t as student_t
    colors = {"computed": "#2a78d6", "early": "#eb6834", "fixed": "#1baf7a",
              "fixed100": "#7a3ec2", "fixed1000": "#1baf7a", "fixed10000": "#8a8a85"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for mode, (C, logs) in results.items():
        S = C.shape[0]
        m = C.mean(0)
        sd = C.std(0, ddof=1) if S > 1 else np.zeros_like(m)
        half = student_t.ppf(0.975, max(S - 1, 1)) * sd / math.sqrt(S)   # 95% CI of the mean
        c = colors.get(mode, "#1baf7a")
        label = f"fixed epochs ({logs[0][0]['cap']} steps)" if mode.startswith("fixed") else f"{mode} epochs"
        ax.plot(Kc, m, color=c, lw=2, label=label, zorder=2)
        ax.fill_between(Kc, np.maximum(m - half, 1e-6), m + half, color=c, alpha=0.18, lw=0, zorder=1)
        # epoch switches (seed 0): K at which each new epoch begins, placed on the mean curve
        starts = np.cumsum([d["steps"] for d in logs[0]])[:-1] + 1
        starts = starts[starts <= Kc[-1]]
        if len(starts) > 200:                     # thin to about 200 log-spaced markers
            keep = np.unique(np.searchsorted(starts, np.logspace(0, math.log10(starts[-1]), 200)))
            starts = starts[keep[keep < len(starts)]]
        ys = np.exp(np.interp(np.log(starts), np.log(Kc), np.log(m)))
        ax.scatter(starts, ys, s=18, color=c, linewidth=0, zorder=3)
    ax.axhline(D_u, color="#52514e", lw=1, ls="--")
    ax.text(1.2, D_u * 1.08, "uniform coverage", color="#52514e", fontsize=8)
    for eps in eps_list:
        ax.axhline(eps, color="#9a9994", lw=0.8, ls=":")
        ax.text(1.2, eps * 1.08, f"ε = {eps:.3g}", color="#52514e", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time step K (per agent)", color="#0b0b0b")
    ax.set_ylabel("regret  (1/K) Σ ‖ρ̂ₖ − ρ*‖₁", color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()