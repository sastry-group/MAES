#!/usr/bin/env python3
# viz/plot_run.py
"""
Single entry point for per-run output. Replaces plot_from_npz.py.

One run, a list of runs, or every run under a parent folder:

    python viz/plot_run.py --folder results/gp_study_D/T2000_ergodic_mh_Pupdate5_GPupdate5_10x10_dynamic_v02_gpucb_dynamic_3_start904595x3_commModelocal_comm_radius1.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_amp9_priorfixed0_noiseRel0.05 --emp_window 2000
    python viz/plot_run.py --parent results/test --match ergodic_mh --emp_window 4000 --do frames,conv
    python viz/plot_run.py --parent results/test --emp_window 0


    python viz/plot_run.py --do csv --csv_at gp --folder results/test/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.2
    python viz/plot_run.py --folder results/test/<runA> results/test/<runB>
    python viz/plot_run.py --parent results/test                  # all runs under it
    python viz/plot_run.py --parent results/test --match noiseRel0.2
    python viz/plot_run.py --parent results/test --list           # just show what matches

Each run is processed independently and writes to its own <run>/plots/.
Comparison across runs is plot_sweep.py, not this script.

Pieces (--do, comma separated):
    grid      true map
    belief    per-agent final belief (grid + one per agent)
    frames    empirical vs normalized belief at snapshot steps
    coverage  coverage_summary.txt
    conv      L1 convergence + regret
    kl        three-panel KL figure
    errors    RMSE / MSLL vs time (needs belief_mu_map, belief_sigma_map)
    csv       one CSV per agent: true / observed / GP mean / GP var / phi
    anim      mp4
    all       everything
Default: grid,belief,frames,coverage,errors
 
Nothing is computed here. Loading/alignment: run_io. Ergodic + coverage
quantities: metrics. GP accuracy: belief_errors.
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
 
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
 
from viz.plotting import plot_grid, plot_distributions, plot_belief_per_agent
from viz.metrics import (full_map_cover_time, roi_first_visit_per_cfg,
                         convergence_errors_from_run)
from viz import run_io
from viz import belief_errors as be
 
# DEFAULT_DO = "grid,belief,frames,conv,coverage,errors"
DEFAULT_DO = "belief,coverage,errors,conv, anim"
# DEFAULT_DO = "conv"
ALL_DO = ["grid", "belief", "frames", "coverage", "conv", "kl", "errors", "csv", "anim"]
 
 
# ── pieces ───────────────────────────────────────────────────────────────────
def piece_grid(R):
    plot_grid(R.env, filename=R.out(f"{R.stem}__grid_true.png"), title="Grid (true info map)")
 
 
def piece_belief(R):
    if R.belief.shape[1] == 1:
        return
    final = R.belief[-1]
    plot_belief_per_agent(R.env, belief_per_agent=final, mode="grid",
                          filename=R.out(f"{R.stem}__final_belief_per_agent_sharedv.png"))
    plot_belief_per_agent(R.env, belief_per_agent=final, mode="separate",
                          filename_template=R.out(f"{R.stem}__final_belief_sharedv_agent{{agent}}.png"))
 
 
def piece_frames(R):
    emp_final = R.empirical([R.T - 1])[0]
    plot_distributions(R.env, emp_final, R.data["pi_vec_final"],
                       filename=R.out(f"{R.stem}__emp_vs_pi_final.png"))
    emp = R.empirical(R.frames)
    for i, k in enumerate(R.frames):
        plot_distributions(R.env, emp[i], R.phi_norm[k],
                           filename=R.out(f"{R.stem}__t{k}_emp_vs_phi_norm.png"))
    gp_steps = np.asarray(R.data.get("gp_update_steps", []), dtype=int)
    if gp_steps.size:
        with open(R.out(f"{R.stem}__updates.txt"), "w") as f:
            f.write("GP update steps:\n" + ", ".join(map(str, gp_steps.tolist())) + "\n")
 
 
def piece_coverage(R):
    path = R.out("coverage_summary.txt")
    print("Writing coverage summary to", path)
    traj, nstates = R.traj, int(R.env.nstates)
    T, n_agents = traj.shape
    starts = run_io.cfg_starts(R.data, T)
    cover = full_map_cover_time(traj_states=traj, nstates=nstates)
 
    roi_sets = {}
    for roi in getattr(R.env, "roi_cells", []):
        r0, c0, r1, c1 = roi["rect"]
        roi_sets[roi["name"]] = {(int(r), int(c)) for r in range(r0, r1) for c in range(c0, c1)
                                 if (r, c) in R.env._state_index_map}
    roi_times = {n: roi_first_visit_per_cfg(traj_rc=R.traj_rc, roi_cellset=s, cfg_starts=starts)
                 for n, s in roi_sets.items()}
 
    with open(path, "w") as f:
        f.write("Coverage summary\n")
        f.write(f"T={T}, n_agents={n_agents}, nstates={nstates}, env_dyn={R.env_dyn}\n")
        if R.env_dyn == "dynamic":
            f.write(f"segment starts={starts}\n")
        f.write("\n")
        f.write(f"Global full-map coverage time: t={int(cover)}\n\n" if not np.isnan(cover)
                else "Global full-map coverage time: NOT achieved\n\n")
        for cfg, t_start in enumerate(starts):
            t_end = starts[cfg + 1] if cfg + 1 < len(starts) else T
            f.write(f"Config {cfg} segment: t = [{t_start}, {t_end})\n")
            seg = traj[t_start:t_end, :].reshape(-1)
            seg = seg[seg >= 0]                       # dead agents are -1
            visited = np.bincount(seg, minlength=nstates) > 0
            f.write(f"  Segment coverage: {int(visited.sum())}/{nstates} states visited\n")
            for name in roi_sets:
                t_rel = roi_times[name][cfg]
                f.write(f"  ROI '{name}': first visited at t={t_start + int(t_rel)} "
                        f"(segment offset={int(t_rel)})\n" if not np.isnan(t_rel)
                        else f"  ROI '{name}': NOT visited in this segment\n")
            if not roi_sets:
                f.write("  ROI: none defined\n")
            f.write("\n")
 
 
def piece_conv(R):
    err_emp_bel, err_bel_true, err_emp_true = convergence_errors_from_run(
        traj_states=R.traj, belief_phi_map=R.belief_avg, true_phi_map_time=R.true_phi,
        env=R.env, empirical_fn=R.emp_fn(), empirical_kwargs={}, norm="l1")
    regret = np.cumsum(err_emp_true) / (np.arange(len(err_emp_true)) + 1)
    x = np.arange(len(err_emp_true))[::R.stride]
    print(f"The empirical-true error at the final time step is: {err_emp_true[-1]:.4f}")
    print(f"The average regret at the final time step is: {regret[-1]:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(x, err_emp_bel[::R.stride],  label=r"$\|\hat{\rho} - \bar{\rho}\|_1$")
    plt.plot(x, err_bel_true[::R.stride], label=r"$\|\bar{\rho} - \rho^\star\|_1$")
    plt.plot(x, err_emp_true[::R.stride], label=r"$\|\hat{\rho} - \rho^\star\|_1$")
    plt.plot(x, regret[::R.stride], linewidth=2.5, label="Regret")
    plt.xlabel("Time step"); plt.ylabel("Error (L1)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(R.out(f"{R.stem}__convergence_multi.png"), dpi=200); plt.close()
 
 
def piece_kl(R):
    if run_io.is_event_driven(R.data):
        print("KL not set up for MAC-DT / doubling (empirical and belief cadence differ); skipping")
        return
    from viz.metrics import (kl_series_emp_vs_true, kl_series_belief_vs_true,
                             kl_inter_agent_series)
    per_agent = R.belief if R.belief.shape[1] > 1 else None
    n = 3 if per_agent is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    s, kl = kl_series_emp_vs_true(R.traj, R.true_phi, R.env, empirical_fn=R.emp_fn(),
                                  empirical_kwargs={}, stride=R.stride)
    axes[0].plot(s, kl); axes[0].set_ylabel("KL divergence")
    axes[0].set_title(r"$D_{\mathrm{KL}}(\hat{\rho} \| \rho^\star)$")
    s, kl = kl_series_belief_vs_true(R.belief_avg, R.true_phi, stride=R.stride)
    axes[1].plot(s, kl, color="C1")
    axes[1].set_title(r"$D_{\mathrm{KL}}(\bar{\rho} \| \rho^\star)$")
    if per_agent is not None:
        s, m, sd = kl_inter_agent_series(per_agent, stride=R.stride)
        axes[2].plot(s, m, color="C2")
        axes[2].fill_between(s, m - sd, m + sd, alpha=0.2, color="C2")
        axes[2].set_title(r"$D_{\mathrm{KL}}(\rho^m \| \bar{\rho})$ (inter-agent)")
    for ax in axes:
        ax.set_xlabel("Time step"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(R.out(f"{R.stem}__convergence_kl.png"), dpi=200); plt.close()
 
 
def piece_errors(R):
    if R.mu is None or R.sigma is None:
        print("No belief_mu_map / belief_sigma_map in this run; skipping RMSE/MSLL")
        return
    rmse = be.rmse(R.mu, R.true_phi)                                   # (T, N)
    msll = be.msll(R.mu, R.sigma, R.true_phi, sigma_eps=R.sigma_eps)   # (T, N)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, per_agent, name in ((axes[0], rmse, "RMSE"), (axes[1], msll, "MSLL")):
        if per_agent.ndim == 2 and per_agent.shape[1] > 1:
            for a in range(per_agent.shape[1]):
                ax.plot(per_agent[:, a], lw=0.6, alpha=0.35, color="gray")
        ax.plot(be.team_average(per_agent), lw=2.0, color="C0", label="team average")
        ax.set_xlabel("Time step"); ax.set_ylabel(name)
        ax.grid(True, alpha=0.3); ax.legend()
    axes[1].axhline(0, color="k", lw=0.8)      # MSLL = 0 is the trivial predictor
    plt.tight_layout(); plt.savefig(R.out(f"{R.stem}__belief_errors.png"), dpi=200); plt.close()
    np.savez(R.out(f"{R.stem}__belief_errors.npz"),
             rmse=rmse, msll=msll, sigma_eps=R.sigma_eps)
    print(f"RMSE (team, final): {be.team_average(rmse)[-1]:.4f}   "
          f"MSLL (team, final): {be.team_average(msll)[-1]:.4f}")
 
 
def piece_csv(R):
    """
    Raw observations and the GP/belief state, as CSV, under <run>/csv_obs_gp/agent<a>/.
 
        csv_obs_gp/agent0/observations.csv     every observation this agent took
        csv_obs_gp/agent0/belief_t000225.csv   full map, as it stood at that step
 
    Belief snapshots are taken at the step BEFORE each P rebuild (--csv_at p,
    the default), so each file is the state the retarget was computed from.
    --csv_at gp uses the GP refit steps, --csv_at every uses every
    --csv_stride-th step.
 
    obs_y is NaN unless the run logged raw observations; gp_mu / gp_sigma are
    NaN for runs without belief_mu_map / belief_sigma_map.
    """
    import csv as _csv
    nan = float("nan")
    obs = np.asarray(R.data["obs_y"], dtype=float) if "obs_y" in R.data else None
    if obs is None:
        print("  (no obs_y in this run -> observed values will be NaN)")
    rc_of = {i: R.env._index_state_map[i] for i in range(R.env.nstates)}
 
    if R.csv_at == "p":
        steps = run_io.update_steps(R.data, R.T, kind="p")
    elif R.csv_at == "gp":
        steps = run_io.update_steps(R.data, R.T, kind="gp")
    else:
        steps = list(range(0, R.T, max(R.csv_stride, 1)))
    if R.csv_at in ("p", "gp"):
        steps = steps[::max(R.csv_stride, 1)]
    snaps = sorted({max(int(t) - 1, 0) for t in steps} | {R.T - 1})   # right BEFORE the update
    print(f"  {len(snaps)} snapshots (--csv_at {R.csv_at}), {R.env.nstates} cells each")
 
    root = os.path.join(os.path.dirname(R.outdir.rstrip(os.sep)), "csv_obs_gp") \
        if os.path.basename(R.outdir) == "plots" else os.path.join(R.outdir, "csv_obs_gp")
    for a in range(R.n_agents):
        adir = os.path.join(root, f"agent{a}")
        os.makedirs(adir, exist_ok=True)
 
        # every observation this agent took
        with open(os.path.join(adir, "observations.csv"), "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["t", "row", "col", "state", "true_phi", "obs_y", "residual"])
            n_written = 0
            for t in range(R.T):
                st = int(R.traj[t, a])
                if st < 0:
                    continue
                y = obs[t, a] if obs is not None else nan
                if obs is not None and not np.isfinite(y):
                    continue
                r, c = rc_of[st]
                tp = R.true_phi[t, st]
                w.writerow([t, r, c, st, tp, y, (y - tp) if np.isfinite(y) else nan])
                n_written += 1
 
        # belief state at each snapshot
        for t in snaps:
            with open(os.path.join(adir, f"belief_t{t:06d}.csv"), "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["row", "col", "state", "true_phi", "obs_y_here",
                            "gp_mu", "gp_sigma", "gp_var", "phi"])
                here = int(R.traj[t, a])
                for i in range(R.env.nstates):
                    r, c = rc_of[i]
                    w.writerow([
                        r, c, i,
                        R.true_phi[t, i],
                        (obs[t, a] if (obs is not None and i == here) else nan),
                        (R.mu[t, a, i] if R.mu is not None else nan),
                        (R.sigma[t, a, i] if R.sigma is not None else nan),
                        (R.sigma[t, a, i] ** 2 if R.sigma is not None else nan),
                        R.belief[t, a, i],
                    ])
        print(f"  agent{a}: {n_written} observations + {len(snaps)} belief snapshots -> {adir}")
 
 
def piece_anim(R):
    from viz.animation import animate_distributions, animate_distributions_multi
    k = max(R.anim_stride, 1)
    fr = np.arange(0, R.T, k)
    emp = R.empirical(fr)
    path = R.out(f"{R.stem}__emp_vs_phi_norm.mp4")
    common = dict(emp_series=emp, pi_series=R.phi_norm[fr], filename=path,
                  fps=R.fps, stride=1, dpi=150, true_phi_map=R.true_phi[fr])
    if R.n_agents == 1:
        animate_distributions(R.env, traj_rc=R.traj_rc[fr, 0, :], **common)
    else:
        att = R.data.get("attrition_log", None)
        if att is not None:
            att = np.asarray(att, dtype=int).copy()
            att[:, 0] = att[:, 0] // k            # death step -> frame index
        animate_distributions_multi(
            R.env, traj_rc_agents=R.traj_rc[fr], attrition_log=att,
            active_mask=None if R.active_mask is None else np.asarray(R.active_mask)[fr],
            **common)
    print(f"Saved animation to: {path}")
 
 
PIECES = {"grid": piece_grid, "belief": piece_belief, "frames": piece_frames,
          "coverage": piece_coverage, "conv": piece_conv, "kl": piece_kl,
          "errors": piece_errors, "csv": piece_csv, "anim": piece_anim}
 
 
# ── run bundle ───────────────────────────────────────────────────────────────
class Run:
    """Everything the pieces need, loaded and aligned once."""
 
    def __init__(self, folder, args):
        self.folder = folder
        self.data, path = run_io.load_run(folder)
        print(f"Loaded: {path}")
        self.stem = os.path.basename(run_io._str(self.data["stem"]))
        self.outdir = args.outdir or os.path.join(folder, "plots")
        os.makedirs(self.outdir, exist_ok=True)
 
        self.env = run_io.env_from_run(self.data)
        self.env_dyn = run_io._str(self.data.get("env_dyn", "static"))
        self.n_agents = int(self.data.get("n_agents", 1))
        self.active_mask = self.data.get("active_mask", None)
        traj_states = np.asarray(self.data["traj_states"])
        self.T = traj_states.shape[0]
        self.traj = run_io.mask_dead_agents(traj_states, self.active_mask)
        self.traj_rc = np.asarray(self.data["traj_rc_agents_all"])
 
        self.belief = run_io.belief_dense(self.data, "belief_phi_map", self.T)   # (T,N,S)
        self.belief_avg = run_io.team_mean(self.belief)
        self.phi_norm = run_io.normalize_rows(self.belief_avg)
        self.mu = run_io.belief_dense(self.data, "belief_mu_map", self.T)
        self.sigma = run_io.belief_dense(self.data, "belief_sigma_map", self.T)
        self.true_phi = run_io.true_phi_dense(self.data, self.T)
        self.sigma_eps = args.msll_sigma_eps if args.msll_sigma_eps is not None else 0.0
 
        self.stride = args.stride or min(int(self.data.get("gp_update", 100)),
                                         int(self.data.get("p_update", 100)))
        self.frames = run_io.frame_indices(self.data, self.T, stride=args.stride)
        if args.limit is not None:
            self.frames = self.frames[:args.limit]
        self.emp_window = args.emp_window
        print(f"rho_hat window: {self.emp_window if self.emp_window else 'full history'}")
        self.csv_at = args.csv_at
        self.csv_stride = args.csv_stride
        self.anim_stride = args.anim_stride
        self.fps = args.fps
 
    def emp_fn(self):
        """Fresh streaming rho_hat estimator at this run's window, usable as
        metrics' empirical_fn. One per consumer: they each walk t from 0."""
        return run_io.StreamingEmpirical(self.traj, self.env.nstates, window=self.emp_window)
 
    def empirical(self, frames):
        """rho_hat at the given frames, same window as everything else."""
        return run_io.empirical_series(self.traj, self.env.nstates, frames,
                                       window=self.emp_window)
 
    def out(self, name):
        return os.path.join(self.outdir, name)
 
 
def find_runs(args):
    """--folder (one or many) and/or --parent (every subfolder holding a run file)."""
    runs = []
    for f in (args.folder or []):
        runs.append(os.path.normpath(f))
    if args.parent:
        for d in sorted(glob.glob(os.path.join(args.parent, "*"))):
            if os.path.isdir(d) and (glob.glob(os.path.join(d, "*.h5"))
                                     or glob.glob(os.path.join(d, "*.npz"))):
                runs.append(os.path.normpath(d))
    if args.match:
        runs = [r for r in runs if all(m in os.path.basename(r) for m in args.match)]
    if args.exclude:
        runs = [r for r in runs if not any(x in os.path.basename(r) for x in args.exclude)]
    seen, out = set(), []
    for r in runs:                      # de-dup, keep order
        if r not in seen:
            seen.add(r); out.append(r)
    return out
 
 
def main():
    ap = argparse.ArgumentParser(description="Per-run plots and animation.")
    src = ap.add_argument_group("which runs")
    src.add_argument("--folder", nargs="+", help="one or more run folders")
    src.add_argument("--parent", help="folder of run folders; every run under it")
    src.add_argument("--match", nargs="+", help="keep runs whose name contains ALL of these")
    src.add_argument("--exclude", nargs="+", help="drop runs whose name contains ANY of these")
    src.add_argument("--list", action="store_true", help="print matching runs and exit")
    src.add_argument("--skip_done", action="store_true",
                     help="skip runs that already have a non-empty plots/")
    src.add_argument("--stop_on_error", action="store_true",
                     help="default is to log the failure and continue to the next run")
 
    ap.add_argument("--outdir", default=None,
                    help="single run only; with several runs each writes to its own plots/")
    ap.add_argument("--do", default=DEFAULT_DO, help=f"comma separated; {','.join(ALL_DO)}, or 'all'")
    ap.add_argument("--stride", type=int, default=None, help="snapshot/KL stride")
    ap.add_argument("--limit", type=int, default=None, help="max snapshot frames")
    ap.add_argument("--emp_window", type=int, default=0,
                    help="sliding window (steps) for rho_hat, used by frames, conv, kl and "
                         "anim alike. 0 = full history (old empirical_full). "
                         "plot_from_npz used 4000 for the paper figures, 10000 for snapshots.")
    ap.add_argument("--csv_at", choices=["p", "gp", "every"], default="p",
                    help="when to dump a belief snapshot: p = step before each P rebuild "
                         "(default), gp = GP refit steps, every = every --csv_stride steps")
    ap.add_argument("--csv_stride", type=int, default=1,
                    help="keep every k-th snapshot")
    ap.add_argument("--anim_stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--msll_sigma_eps", type=float, default=None,
                    help="add this observation noise to the MSLL predictive variance; "
                         "default 0 (score against the noise-free true field)")
    args = ap.parse_args()
 
    if not args.folder and not args.parent:
        raise SystemExit("give --folder <run> [...] and/or --parent <folder of runs>")
    todo = ALL_DO if args.do.strip() == "all" else [p.strip() for p in args.do.split(",") if p.strip()]
    unknown = [p for p in todo if p not in PIECES]
    if unknown:
        raise SystemExit(f"unknown piece(s) {unknown}; choose from {ALL_DO} or 'all'")
 
    runs = find_runs(args)
    if not runs:
        raise SystemExit("no runs matched")
    if args.list:
        for r in runs:
            print(r)
        raise SystemExit(0)
    if args.outdir and len(runs) > 1:
        raise SystemExit("--outdir only makes sense for a single run")
 
    failed, skipped = [], []
    for i, folder in enumerate(runs, 1):
        name = os.path.basename(folder)
        plots = os.path.join(folder, "plots")
        if args.skip_done and os.path.isdir(plots) and os.listdir(plots):
            print(f"[{i}/{len(runs)}] skip (already done) {name}")
            skipped.append(name)
            continue
        print(f"\n[{i}/{len(runs)}] {name}")
        try:
            R = Run(folder, args)
            for piece in todo:
                print(f"--- {piece}")
                PIECES[piece](R)
            print(f"Saved to: {R.outdir}")
        except Exception as e:
            if args.stop_on_error:
                raise
            print(f"!! FAILED {name}: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))
 
    print(f"\n=== {len(runs) - len(failed) - len(skipped)} ok, "
          f"{len(skipped)} skipped, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
 
 
if __name__ == "__main__":
    main()