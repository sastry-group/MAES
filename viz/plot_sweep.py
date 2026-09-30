#!/usr/bin/env python3
# viz/plot_sweep.py
"""
Compare runs in one folder. plot_run.py is one run; this is the comparison.

    # ergodic vs mac_dt at one noise level, averaged over the start configs
    python3 viz/plot_sweep.py --parent results/test \
        --fix noise_rel=0.05 --compare planner --average start

    python3 viz/plot_sweep.py --parent results/gp_study_A \
        --fix noise_rel=0.05 --compare init --average start

    python3 viz/plot_sweep.py --parent results/gp_study_A \
     --fix gpns=1 --compare noise_rel --average start


    python3 viz/plot_sweep.py --parent results/gp_studies_B \
     --fix map=simple5x5_static_v0 --compare prior --average start

         python3 viz/plot_sweep.py --parent results/gp_study_B \
      --compare prior --average start
    python3 viz/plot_sweep.py --parent results/gp_study_B \
        --compare prior amp --average start map --refresh

    python3 viz/plot_sweep.py --parent results/gp_studies_C \
        --fix map=simple5x5_static_v0 --compare beta_u beta_sched planner --average start --refresh
    python3 viz/plot_sweep.py --parent results/gp_studies_C \
        --fix map=simple10x10_static_v1 --compare beta_u beta_sched planner --average start

            python3 viz/plot_sweep.py --parent results/gp_study_C \
       --compare beta_u beta_sched planner --average start map --refresh

    python3 viz/plot_sweep.py --parent results/gp_studies_D_10 \
     --fix noise_rel=0.05 --compare kernel --average start --refresh
    python3 viz/plot_sweep.py --parent results/gp_studies_D_10 \
  --fix noise_rel=0.2  --compare kernel --average start --refresh
\    python3 viz/plot_sweep.py --parent results/gp_studies_D \
         --fix noise_rel=0.05 --compare kernel --average start --refresh
    python3 viz/plot_sweep.py --parent results/gp_studies_D \
         --fix noise_rel=0.2  --compare kernel --average start --refresh
    python3 viz/plot_sweep.py --parent results/gp_study_D \
         --fix noise_rel=0.05 --compare kernel --average start map --refresh
    python3 viz/plot_sweep.py --parent results/gp_study_D \
         --fix noise_rel=0.2  --compare kernel --average start map --refresh
         

    python3 viz/plot_sweep.py --parent results/gp_studies_E \
        --fix map=simple5x5_static_v0 --compare kernel learn --average start --refresh        

    python3 viz/plot_sweep.py --parent results/gp_studies_E \
        --fix map=simple10x10_static_v1 --compare kernel learn --average start --refresh     

    python3 viz/plot_sweep.py --parent results/gp_study_E \
        --compare kernel learn --average start map --refresh 

        python3 viz/plot_sweep.py --parent results/gp_study_G \
        --compare tau --average start map --refresh  

A. Method, no forgetting (exact vs. averaging vs. SGPR vs. SGPR-FITC):
python3 viz/plot_sweep.py --parent results/gp_study_H \
  --fix gp_obs_hist=100000 --compare gp_method gp_M gp_bin_dt gp_fitc \
  --average start map --name H_A_method --refresh

  
B. FIFO alone:
python3 viz/plot_sweep.py --parent results/gp_study_H \
  --fix gp_method=exact gp_bin_dt=0 --compare gp_obs_hist \
  --average start map --name H_B_fifo

Combos at a fixed B:
  python3 viz/plot_sweep.py --parent results/gp_study_H \
  --fix gp_obs_hist=500 --compare gp_method gp_M gp_bin_dt \
  --average start map --name H_C_combo

    python viz/plot_sweep.py --parent results/test --list      # see the parsed keys


    python3 viz/plot_sweep.py --parent results/gp_study_H_20 \
  --compare gp_method gp_M gp_bin_dt gp_obs_hist --average start map --name H_20x20
python3 viz/plot_gp_timing.py --parent results/gp_study_H_20

Every run is described by keys parsed from its folder name and h5 attrs:
    planner, noise_rel, gpns, start, n_agents, T
You then say what each key is for:
    --fix KEY=VAL ...   keep only runs with this value (dropped from the plot)
    --compare KEY ...   one curve per value
    --average KEY ...   averaged into a mean +/- 1 std band
Any key that still varies and was not named is reported and left as an extra
curve dimension, so nothing is silently averaged together.

Panels (--panels), all on one figure, all vs time:
    erg_err   ||rho_hat - rho*||_1     ergodic error
    E_k       cumulative mean of erg_err   regret
    emp_bel   ||rho_hat - rho_bar||_1
    L_belief  ||rho_bar - rho*||_1     belief error (Eq. (7))
    rmse      GP posterior mean vs true field
    msll      standardized log loss

Metrics come from the same modules plot_run.py uses, so a run's numbers here
match its own plots.
"""
import os
import re
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from viz.metrics import convergence_errors_from_run
from viz import run_io
from viz import belief_errors as be

KEYS = ["planner", "noise_rel", "gpns", "start", "n_agents", "T",
        "map", "prior", "amp", "kernel", "beta_u", "init", "learn",
        "tau", "gp_method", "gp_M", "gp_bin_dt", "gp_fitc", "gp_obs_hist"]
ALL_PANELS = ["erg_err", "E_k", "emp_bel", "L_belief", "rmse", "msll"]
DEFAULT_PANELS = ["erg_err", "E_k", "L_belief", "rmse", "msll"]
LABELS = {
    "erg_err": r"$\|\hat{\rho}-\rho^\star\|_1$",
    "E_k": r"Regret $E_k$",
    "emp_bel": r"$\|\hat{\rho}-\bar{\rho}\|_1$",
    "L_belief": r"$\|\bar{\rho}-\rho^\star\|_1$",
    "rmse": "RMSE",
    "msll": "MSLL",
    
}


# ── describing a run ─────────────────────────────────────────────────────────
def run_keys(folder, data):
    name = os.path.basename(folder)

    def grab(pat, cast=float):
        m = re.search(pat, name)
        return cast(m.group(1)) if m else None

    planner = run_io.planner_name(data)
    if planner != "mac_dt" and re.search(r"T\d+_mac_dt_", name):
        planner = "mac_dt"
    elif planner != "mac_dt":
        planner = grab(r"T\d+_([a-z_]+?)_(?:Pupdate|\d+x\d+)", str) or planner
    return {
        "planner": planner,
        "noise_rel": (float(data["noise_rel"]) if "noise_rel" in data
                      else grab(r"noiseRel([\d.]+)")),
        "gpns": (float(data["gp_noise_scale"]) if "gp_noise_scale" in data
                 else (grab(r"_gpns([\d.]+)") or 1.0)),
        "start": grab(r"_start([0-9]+x[0-9]+)", str),
        "n_agents": int(data.get("n_agents", 1)),
        "T": int(np.asarray(data["traj_states"]).shape[0]),
        "map": os.path.basename(run_io._str(data.get("env_source", ""))).replace(".json", ""),
        "prior": f'{run_io._str(data.get("gp_prior", "data"))}{float(data.get("gp_prior_mean", 0)):g}',
"       beta_u": float(data.get("ucb_beta_u", 3.0)),
        "init": grab(r"_init([\d.]+)") or 6.0,
        # "learn": run_io._str(data.get("learn_hypers", "none")),
        "learn": (lambda m: m.group(1) if m else "none")(re.search(r"_learn(once|every)", name)),
        "amp": float(data.get("gp_kernel_amp", 10.0)),
        "kernel": run_io._str(data.get("gp_kernel", "matern32")),
        "beta_u": (lambda b: 3.0 if b is None else b)(grab(r"_bu([\d.]+)")),
        "beta_sched": "mac_dt" if "_ucb_betamac_dt" in name else "const",
        "tau": grab(r"_Pupdate(\d+)", int),
        "gp_method": (str(data["gp_method"]) if "gp_method" in data
                      else grab(r"_gpm(exact|sgpr)", str)),
        "gp_M": (int(data["gp_M"]) if "gp_M" in data
                 else grab(r"_M(\d+)_bin", int)),
        "gp_bin_dt": (int(data["gp_bin_dt"]) if "gp_bin_dt" in data
                      else grab(r"_bin(\d+)_fitc", int)),
        "gp_fitc": (int(data["gp_fitc"]) if "gp_fitc" in data
                    else grab(r"_fitc(\d)", int)),
        "gp_obs_hist": grab(r"_GPobs(\d+)", int),

    }


def series_of(data, args):
    T = np.asarray(data["traj_states"]).shape[0]
    env = run_io.env_from_run(data)
    traj = run_io.mask_dead_agents(data["traj_states"], data.get("active_mask", None))
    belief = run_io.belief_dense(data, "belief_phi_map", T)
    true_phi = run_io.true_phi_dense(data, T)
    mu = run_io.belief_dense(data, "belief_mu_map", T)
    sigma = run_io.belief_dense(data, "belief_sigma_map", T)

    emp_bel, bel_true, emp_true = convergence_errors_from_run(
        traj_states=traj, belief_phi_map=run_io.team_mean(belief),
        true_phi_map_time=true_phi, env=env,
        empirical_fn=run_io.StreamingEmpirical(traj, env.nstates, window=args.emp_window),
        empirical_kwargs={}, norm="l1")

    out = {"erg_err": emp_true, "emp_bel": emp_bel, "L_belief": bel_true,
           "E_k": np.cumsum(emp_true) / (np.arange(len(emp_true)) + 1)}
    if mu is not None and sigma is not None:
        out["rmse"] = be.team_average(be.rmse(mu, true_phi))
        out["msll"] = be.team_average(
            be.msll(mu, sigma, true_phi, sigma_eps=args.msll_sigma_eps or 0.0))
    return out


def load_all(runs, args):
    cache_dir = os.path.join(args.parent, "analysis", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    loaded = []
    for i, folder in enumerate(runs, 1):
        name = os.path.basename(folder)
        h5 = (glob.glob(os.path.join(folder, "*.h5")) + glob.glob(os.path.join(folder, "*.npz")))[0]
        cache = os.path.join(cache_dir, f"{name}__w{args.emp_window}.npz")
        try:
            if (not args.refresh and os.path.exists(cache)
                    and os.path.getmtime(cache) > os.path.getmtime(h5)):
                z = np.load(cache, allow_pickle=True)
                loaded.append((z["keys"].item(),
                               {k: z[k] for k in ALL_PANELS if k in z.files}))
                print(f"[{i}/{len(runs)}] cached  {name}")
                continue
            data, _ = run_io.load_run(folder)
            keys = run_keys(folder, data)
            ser = series_of(data, args)
            np.savez(cache, keys=np.array(keys, dtype=object), **ser)
            loaded.append((keys, ser))
            print(f"[{i}/{len(runs)}] {name}")
        except Exception as e:
            if args.stop_on_error:
                raise
            print(f"!! FAILED {name}: {type(e).__name__}: {e}")
    return loaded


# ── grouping ─────────────────────────────────────────────────────────────────
def as_str(v):
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def apply_fix(loaded, fixes):
    out = []
    for keys, ser in loaded:
        if all(as_str(keys.get(k)) == v for k, v in fixes.items()):
            out.append((keys, ser))
    return out


def band(runs, metric):
    arrs = [r[metric] for r in runs if metric in r]
    if not arrs:
        return None, None, 0
    n = min(len(a) for a in arrs)
    Y = np.stack([np.asarray(a, dtype=float)[:n] for a in arrs])
    return Y.mean(axis=0), Y.std(axis=0), len(arrs)


def main():
    ap = argparse.ArgumentParser(description="Compare runs in one folder.")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--fix", nargs="+", default=[], metavar="KEY=VAL",
                    help=f"keep only runs with this key value; keys: {KEYS}")
    ap.add_argument("--compare", nargs="+", default=["planner"],
                    help="one curve per value of these keys")
    ap.add_argument("--average", nargs="+", default=["start"],
                    help="average over these keys (mean +/- 1 std band)")
    ap.add_argument("--panels", nargs="+", default=DEFAULT_PANELS, choices=ALL_PANELS)
    ap.add_argument("--match", nargs="+", help="name filter: keep if name contains ALL")
    ap.add_argument("--exclude", nargs="+", help="name filter: drop if name contains ANY")
    ap.add_argument("--list", action="store_true", help="print runs with parsed keys and exit")
    ap.add_argument("--outdir", default=None, help="default <parent>/analysis")
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--emp_window", type=int, default=0,
                    help="rho_hat window, same meaning as plot_run.py; 0 = full history")
    ap.add_argument("--msll_sigma_eps", type=float, default=None)
    ap.add_argument("--logy", nargs="+", default=[], choices=ALL_PANELS,
                    help="panels to draw on a log y axis")
    ap.add_argument("--ncols", type=int, default=0, help="panel columns (0 = auto)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    runs = [os.path.normpath(d) for d in sorted(glob.glob(os.path.join(args.parent, "*")))
            if os.path.isdir(d) and (glob.glob(os.path.join(d, "*.h5"))
                                     or glob.glob(os.path.join(d, "*.npz")))]
    if args.match:
        runs = [r for r in runs if all(m in os.path.basename(r) for m in args.match)]
    if args.exclude:
        runs = [r for r in runs if not any(x in os.path.basename(r) for x in args.exclude)]
    if not runs:
        raise SystemExit("no runs matched")

    if args.list:
        for r in runs:
            data, _ = run_io.load_run(r)
            k = run_keys(r, data)
            print(f"{os.path.basename(r)}\n    " +
                  "  ".join(f"{n}={as_str(k[n])}" for n in KEYS))
        raise SystemExit(0)

    fixes = {}
    for f in args.fix:
        if "=" not in f:
            raise SystemExit(f"--fix wants KEY=VAL, got {f!r}")
        k, v = f.split("=", 1)
        if k not in KEYS:
            raise SystemExit(f"unknown key {k!r}; choose from {KEYS}")
        fixes[k] = v

    loaded = load_all(runs, args)
    loaded = apply_fix(loaded, fixes)
    if not loaded:
        raise SystemExit(f"no runs left after --fix {fixes}")

    # any key that still varies and was not assigned a role
    assigned = set(fixes) | set(args.compare) | set(args.average)
    leftover = [k for k in KEYS if k not in assigned
                and len({as_str(kk.get(k)) for kk, _ in loaded}) > 1]
    if leftover:
        print(f"\n!! these keys still vary and were not named in --compare/--average/--fix: "
              f"{leftover}\n   adding them to the curve labels so nothing is averaged blindly.")
    curve_keys = list(args.compare) + leftover

    grouped = defaultdict(list)
    for keys, ser in loaded:
        grouped[tuple(as_str(keys.get(k)) for k in curve_keys)].append((keys, ser))

    print(f"\n{len(loaded)} runs -> {len(grouped)} curves")
    for g, members in sorted(grouped.items()):
        lbl = ", ".join(f"{k}={v}" for k, v in zip(curve_keys, g))
        avg_vals = {k: sorted({as_str(kk.get(k)) for kk, _ in members}) for k in args.average}
        print(f"  {lbl}: n={len(members)}  averaged over " +
              "; ".join(f"{k}={v}" for k, v in avg_vals.items()))

    # ── figure ───────────────────────────────────────────────────────────────
    panels = [p for p in args.panels
              if any(p in s for _, members in grouped.items() for _, s in members)]
    dropped = [p for p in args.panels if p not in panels]
    if dropped:
        print(f"(no data for {dropped} in any run -- panels omitted)")
    ncols = args.ncols or min(len(panels), 3)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, metric in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        for ci, (g, members) in enumerate(sorted(grouped.items())):
            m, sd, n = band([s for _, s in members], metric)
            if m is None:
                continue
            t = np.arange(len(m))
            lbl = ", ".join(f"{k}={v}" for k, v in zip(curve_keys, g)) + f" (n={n})"
            c = colors[ci % 10]
            ax.plot(t, m, color=c, lw=1.7, label=lbl)
            if n > 1:
                ax.fill_between(t, m - sd, m + sd, color=c, alpha=0.15, lw=0)
        ax.set_xlabel("Time step"); ax.set_ylabel(LABELS[metric])
        ax.grid(True, alpha=0.25)
        if metric in args.logy:
            ax.set_yscale("log")
        if metric == "msll":
            ax.axhline(0, color="k", lw=0.8)
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    # axes[0][0].legend(fontsize=7)
        handles, labels = axes[0][0].get_legend_handles_labels()
        n_empty = nrows * ncols - len(panels)
        if n_empty > 0:
            # put the legend in the first unused (blank) panel
            j = len(panels)
            axes[j // ncols][j % ncols].legend(handles, labels, loc="center", fontsize=8, frameon=False)
        else:
            # no blank panel: put it below the figure
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3),
                    fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    title = "  ".join(f"{k}={v}" for k, v in fixes.items())
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    outdir = args.outdir or os.path.join(args.parent, "analysis")
    os.makedirs(outdir, exist_ok=True)
    stem = args.name or ("compare_" + "_".join(args.compare) +
                         ("__" + "_".join(f"{k}{v}" for k, v in fixes.items()) if fixes else "") +
                         (f"__w{args.emp_window}" if args.emp_window else ""))
    out = os.path.join(outdir, stem + ".png")
    fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()