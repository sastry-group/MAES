#!/usr/bin/env python3
import os
import argparse
import numpy as np
import glob
import sys
import h5py

import numpy as np
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.grid import  make_env_from_json              
from viz.plotting import (
    plot_grid,
    plot_trajectory,
    plot_distributions,
    plot_convergence,
)

from viz.metrics import full_map_cover_time, roi_first_visit_per_cfg, empirical_from_traj, empirical_full, convergence_errors_from_run


from viz.animation import (
    animate_distributions,
    animate_distributions_multi,
)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}  # load all keys
    z.close()
    return data

def load_h5(path):
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
        for key, val in f.attrs.items():
            data[key] = val
    return data


def plot_convergence_multi(
    env,
    traj_states,             # (T, n_agents)
    belief_phi_map,          # (T, nstates) or (T, N, nstates)
    true_phi_map_time,       # (T, nstates)
    filename,
    stride=1,
    empirical_fn=empirical_full,
    empirical_kwargs=None,
    norm="l1",
    cover_time=None,
):
    """
    Plots:
        || rho_hat - rho_bar ||_1
        || rho_bar - rho_star ||_1
        || rho_hat - rho_star ||_1
        Regret (average cumulative emp-true error)
    """

    if empirical_kwargs is None:
        empirical_kwargs = {}

    # ---- Handle per-agent belief if needed ----
    if belief_phi_map.ndim == 3:
        belief_phi_map = belief_phi_map.mean(axis=1)

    # ---- Compute errors using shared engine ----
    err_emp_bel, err_bel_true, err_emp_true = convergence_errors_from_run(
        traj_states=traj_states,
        belief_phi_map=belief_phi_map,
        true_phi_map_time=true_phi_map_time,
        env=env,
        empirical_fn=empirical_fn,
        empirical_kwargs=empirical_kwargs,
        norm=norm,
    )

    # ---- Regret (average cumulative emp-true error) ----
    regret = np.cumsum(err_emp_true) / (np.arange(len(err_emp_true)) + 1)

    # ---- Apply stride ----
    x = np.arange(len(err_emp_true))
    x = x[::stride]

    err_emp_bel  = err_emp_bel[::stride]
    err_bel_true = err_bel_true[::stride]
    err_emp_true = err_emp_true[::stride]
    regret       = regret[::stride]
    print(f"The empirical-true error at the final time step is: {err_emp_true[-1]:.4f}")
    print(f"The average regret at the final time step is: {regret[-1]:.4f}")


    # ---- Plot ----
    plt.figure(figsize=(10, 6))

    plt.plot(x, err_emp_bel,  label=r"$\|\hat{\rho} - \bar{\rho}\|_1$")
    plt.plot(x, err_bel_true, label=r"$\|\bar{\rho} - \rho^\star\|_1$")
    plt.plot(x, err_emp_true, label=r"$\|\hat{\rho} - \rho^\star\|_1$")
    plt.plot(x, regret,       linewidth=2.5, label="Regret")

    if cover_time is not None and cover_time > 0:
        plt.axvline(
            cover_time,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Full coverage"
        )

    plt.xlabel("Time step")
    plt.ylabel("Error (L1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def write_coverage_summary(
    outdir,
    env,
    traj_states,          # (T, n_agents)
    traj_rc_agents_all,   # (T, n_agents, 2) – (r,c)
    steps_per_cfg,        # int
    env_dyn="static",     # "static" or "dynamic"
):
    """
    Writes:
      - ROI discovery time per config segment (first visit within that segment)
      - Segment coverage (#unique visited within segment)
      - Global full-map coverage time (first time all states have ever been visited)
    """
    summary_path = os.path.join(outdir, "coverage_summary.txt")
    print("Writing coverage summary to", summary_path)

    traj_states = np.asarray(traj_states)
    traj_rc = np.asarray(traj_rc_agents_all)
    T, n_agents = traj_states.shape
    nstates = int(env.nstates)


    full_cover_time = full_map_cover_time(
        traj_states=traj_states,
        nstates=env.nstates,
    )


    # Decide number of configs
    if env_dyn == "dynamic":
        n_cfg = int(np.ceil(T / max(int(steps_per_cfg), 1)))
        cfg_starts = [i * steps_per_cfg for i in range(n_cfg)]
    else:
        n_cfg = 1  # static
        cfg_starts = [0]

    roi_sets = {}
    for roi in getattr(env, "roi_cells", []):
        name = roi["name"]
        r0, c0, r1, c1 = roi["rect"]
        roi_sets[name] = {
            (int(r), int(c))
            for r in range(r0, r1)
            for c in range(c0, c1)
            if (r, c) in env._state_index_map
        }

    roi_times = {}
    for name, cellset in roi_sets.items():
        roi_times[name] = roi_first_visit_per_cfg(
            traj_rc=traj_rc,
            roi_cellset=cellset,
            cfg_starts=cfg_starts,
        )

    with open(summary_path, "w") as f:
        f.write("Coverage summary\n")
        f.write(f"T={T}, n_agents={n_agents}, nstates={nstates}, env_dyn={env_dyn}\n")
        if env_dyn == "dynamic":
            f.write(f"steps_per_cfg={steps_per_cfg}, n_cfg={n_cfg}\n")
        f.write("\n")

        # ---- Global coverage ----
        if not np.isnan(full_cover_time):
            f.write(f"Global full-map coverage time: t={int(full_cover_time)}\n\n")
        else:
            f.write("Global full-map coverage time: NOT achieved\n\n")

        # ---- Per-config stats ----
        for cfg in range(n_cfg):
            t_start = cfg_starts[cfg]
            t_end = min(cfg_starts[cfg + 1], T) if cfg + 1 < n_cfg else T

            f.write(f"Config {cfg} segment: t = [{t_start}, {t_end})\n")

            # Segment coverage (state space, segment-local)
            seg_visits = traj_states[t_start:t_end, :].reshape(-1)
            seg_counts = np.zeros(nstates, dtype=int)
            for s in seg_visits:
                seg_counts[int(s)] += 1
            seg_visited = seg_counts > 0

            f.write(
                f"  Segment coverage: "
                f"{int(seg_visited.sum())}/{nstates} states visited\n"
            )

            # ROI discovery
            if roi_sets:
                for roi_name in roi_sets:
                    t_rel = roi_times[roi_name][cfg]
                    if not np.isnan(t_rel):
                        f.write(
                            f"  ROI '{roi_name}': first visited at "
                            f"t={t_start + int(t_rel)} "
                            f"(segment offset={int(t_rel)})\n"
                        )
                    else:
                        f.write(
                            f"  ROI '{roi_name}': NOT visited in this segment\n"
                        )
            else:
                f.write("  ROI: none defined\n")

            f.write("\n")
   

def plot_convergence_kl(
    env,
    traj_states,           # (T, n_agents)
    belief_phi_map,        # (T, nstates) or (T, N, nstates)
    true_phi_map_time,     # (T, nstates)
    filename,
    phi_agents_time=None,  # (T, N, nstates) – needed for inter-agent KL
    empirical_fn=empirical_full,
    empirical_kwargs=None,
    stride=1,
    cover_time=None,
):
    """
    Three-panel KL convergence figure:
        (a) KL(rho_hat || rho*)   – empirical vs true
        (b) KL(rho_bar || rho*)   – belief vs true
        (c) KL(rho^m  || rho_bar) – inter-agent belief divergence (if available)
    """
    from viz.metrics import (
        kl_series_emp_vs_true,
        kl_series_belief_vs_true,
        kl_inter_agent_series,
    )

    n_panels = 3 if phi_agents_time is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=False)

    # (a) empirical vs true
    steps, kl_et = kl_series_emp_vs_true(
        traj_states, true_phi_map_time, env,
        empirical_fn=empirical_fn,
        empirical_kwargs=empirical_kwargs or {},
        stride=stride,
    )
    axes[0].plot(steps, kl_et)
    axes[0].set_title(r"$D_{\mathrm{KL}}(\hat{\rho} \| \rho^\star)$")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("KL divergence")

    # (b) belief vs true
    steps_b, kl_bt = kl_series_belief_vs_true(
        belief_phi_map, true_phi_map_time, stride=stride
    )
    axes[1].plot(steps_b, kl_bt, color="C1")
    axes[1].set_title(r"$D_{\mathrm{KL}}(\bar{\rho} \| \rho^\star)$")
    axes[1].set_xlabel("Time step")

    # (c) inter-agent (optional)
    if phi_agents_time is not None:
        steps_ia, kl_m, kl_s = kl_inter_agent_series(
            phi_agents_time, stride=stride
        )
        axes[2].plot(steps_ia, kl_m, color="C2")
        axes[2].fill_between(steps_ia, kl_m - kl_s, kl_m + kl_s,
                             alpha=0.2, color="C2")
        axes[2].set_title(r"$D_{\mathrm{KL}}(\rho^m \| \bar{\rho})$ (inter-agent)")
        axes[2].set_xlabel("Time step")

    for ax in axes:
        if cover_time is not None:
            ax.axvline(cover_time, color="red", ls="--", alpha=0.7,
                       linewidth=1.5, label="Full coverage")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"KL convergence plot saved to {filename}")




def mask_dead_agents(traj_states, active_mask):
    """Replace traj_states entries for dead agents with -1 
    so empirical_from_traj can skip them."""
    masked = traj_states.copy()
    if active_mask is not None:
        masked[~active_mask] = -1
    return masked



def main():
    ap = argparse.ArgumentParser(description="Plot saved ergodic/GP run from NPZ.")
    # ap.add_argument("--npz", required=True, help="Path to results/run_*.npz", default='results/RECM_GPupdate25_noise0p5_Pupdate25')
    ap.add_argument("--folder",  help="Path to folder containing a results/run_*.npz", default='results/T500_ergodic_GPupdate25_remc_Pupdate25_v1_gpucb_static_3_start00x3_commModelocal_comm_radius1.0_concenIter5')
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside the npz)")
    ap.add_argument("--stride", type=int, default=50, help="Frame stride when dumping per-step emp vs pi (default: 50)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of frames to export (default: all with stride)")
    ap.add_argument("--seed", type=int, default=1, help="Env seed to match the run() script (default: 1)")
    ap.add_argument("--neighbors", default="4", help="Override neighbors (else read from npz)")
    args = ap.parse_args()


    files = os.listdir(args.folder)

    npz_files = [f for f in files if f.endswith(".npz")]
    h5_files  = [f for f in files if f.endswith(".h5")]

    if npz_files:
        if len(npz_files) > 1:
            print(f"Multiple .npz files found, using the first: {npz_files[0]}")
        path = os.path.join(args.folder, npz_files[0])
        data = load_npz(path)
        print(f"Loaded NPZ: {path}")

    elif h5_files:
        if len(h5_files) > 1:
            print(f"Multiple .h5 files found, using the first: {h5_files[0]}")
        path = os.path.join(args.folder, h5_files[0])
        data = load_h5(path)
        print(f"Loaded H5: {path}")

    else:
        raise FileNotFoundError(f"No .npz or .h5 file found in {args.folder}")


    # --- Attrition info ---
    has_attrition = data.get("attrition", False)
    active_mask = data.get("active_mask", None)  # (T, N) bool or None
    attrition_log = data.get("attrition_log", None)

    n_agents = int(data.get("n_agents", 1))
    planner = str(data.get("planner", data.get("method", "ergodic")))
    if planner == "mac_dt":
        stride = int(data.get("gp_update"))
    elif planner == "ergodic" and args.method == "remc":
        stride = min(
            int(data.get("gp_update")),
            int(data.get("p_update")),
        )
    else:
        stride = 100
    # print(f"Using stride={stride} based on planner={planner} and updates gp={data.get('gp_update')} p={data.get('p_update')}")
        

    stem = data["stem"]
    stem = os.path.basename(str(data["stem"]))


    # outdir = ensure_dir(args.outdir or os.path.join(args.folder, f"plots_{stem}"))
    outdir = ensure_dir(args.outdir or os.path.join(args.folder, "plots"))
    # print(f"Output directory: {outdir}")
    env_source = str(data["env_source"])

    # env = make_env(neighbors=neighbors, seed=args.seed)
    if data["env_dyn"] == "dynamic":
        # env_source is a folder
        json_files = sorted(glob.glob(os.path.join(env_source, "*.json")))
        env = make_env_from_json(json_files[0])
    elif os.path.isfile(env_source):
        # env_source is a single json
        env = make_env_from_json(env_source)
    else:
        env = make_env_from_json("configs/simple5x5_static.json")
        # env = make_env(neighbors=neighbors, seed=args.seed)

    env._indexing()
    env._precompute_graph()
    traj_states = data["traj_states"]
    T = traj_states.shape[0]
    emp_final = empirical_from_traj(traj_states, env, k=T-1)
    # plotting distributions 
    traj_states = data["traj_states"]
    if "belief_phi_map_avg" in data:
        belief_phi = data["belief_phi_map_avg"]

    elif "belief_phi_map" in data:
        belief_phi = data["belief_phi_map"]

        # If per-agent belief (T, N, nstates), average it
        if belief_phi.ndim == 3:
            belief_phi = belief_phi.mean(axis=1)

    else:
        raise KeyError("No belief dataset found.")

    phi_norm_series = belief_phi / np.maximum(
        belief_phi.sum(axis=1, keepdims=True),
        1e-12
    )


    plot_grid(env, filename=os.path.join(outdir, f"{stem}__grid_true.png"),
              title="Grid (true info map)")

    # --- Trajectory ---
    if n_agents == 1:
        plot_trajectory(
            env,
            data["traj_rc_agents_all"][:, 0, :],
            filename=os.path.join(outdir, f"{stem}__trajectory.png"),
            title="Trajectory",
        )
    else:
        for a in range(n_agents):
            traj_a = data["traj_rc_agents_all"][:, a, :]

            if active_mask is not None:
                alive = active_mask[:, a]
                traj_a = traj_a[alive]
                if len(traj_a) == 0:
                    print(f"Agent {a}: no alive steps, skipping trajectory plot")
                    continue
                death_step = np.argmin(alive) if not alive.all() else None
                title = f"Trajectory agent {a} (died t={death_step})" \
                        if death_step is not None else f"Trajectory (agent {a})"
            else:
                title = f"Trajectory (agent {a})"

            plot_trajectory(
                env,
                traj_a,
                filename=os.path.join(outdir, f"{stem}__trajectory_agent{a}.png"),
                title=title,
            )

    # --- Final distributions ---
    # plot_distributions(env, emp_final, data["pi_vec_final"],
    #                    filename=os.path.join(outdir, f"{stem}__emp_vs_pi_final.png"))

    traj_for_metrics = mask_dead_agents(data["traj_states"], active_mask)
    # --- Convergence plot ---
    plot_convergence_multi(
        env=env,
        traj_states=traj_for_metrics,
        belief_phi_map=belief_phi,
        true_phi_map_time=data["true_phi_map_time"],
        filename=os.path.join(outdir, f"{stem}_convergence_multi.png"),
        stride=stride,
    )


    frame_indices = list(range(0, T, stride))
    if args.limit is not None:
        frame_indices = frame_indices[:args.limit]

    for k in frame_indices:
        emp_k = empirical_from_traj(traj_for_metrics, env, k)
        plot_distributions(
            env,
            emp_k,
            phi_norm_series[k],
            filename=os.path.join(outdir, f"{stem}__t{k}_emp_vs_phi_norm.png"),
        )

    gp_steps = data.get("gp_update_steps", np.array([], dtype=int))
    p_steps  = data.get("p_update_steps", np.array([], dtype=int))

    if gp_steps.size > 0 or p_steps.size > 0:
        with open(os.path.join(outdir, f"{stem}__updates.txt"), "w") as f:
            if gp_steps.size > 0:
                f.write("GP update steps:\n")
                f.write(", ".join(map(str, gp_steps.tolist())) + "\n\n")
            if p_steps.size > 0:
                f.write("P rebuild steps:\n")
                f.write(", ".join(map(str, p_steps.tolist())) + "\n")

    print(f"Saved plots to: {outdir}")



    write_coverage_summary(
        outdir=outdir,
        env=env,
        traj_states=traj_for_metrics,
        traj_rc_agents_all=data["traj_rc_agents_all"],
        steps_per_cfg=int(data.get("steps_per_cfg", T)),
        env_dyn=str(data.get("env_dyn", "static")),
    )



    # --- KL convergence plot ---
    phi_agents_time = data.get("belief_phi_map", None)
    # Only use it for inter-agent panel if it's actually per-agent (3D)
    if phi_agents_time is not None and phi_agents_time.ndim != 3:
        phi_agents_time = None

    plot_convergence_kl(
        env=env,
        traj_states=traj_for_metrics,
        belief_phi_map=belief_phi,          # already loaded above (averaged)
        true_phi_map_time=data["true_phi_map_time"],
        filename=os.path.join(outdir, f"{stem}_convergence_kl.png"),
        phi_agents_time=phi_agents_time,    # (T, N, nstates) → enables inter-agent panel
        stride=stride,
    )

    # --- Animations ---
    anim_path = os.path.join(outdir, f"{stem}__emp_vs_phi_norm.mp4")
    anim_stride = 1
    

    true_phi_map_time = data.get("true_phi_map_time", None)

    emp_series = [
        empirical_from_traj(traj_states, env, k)
        for k in range(traj_states.shape[0])
    ]
    emp_series = np.asarray(emp_series)


    if n_agents == 1:


        animate_distributions(
            env,
            emp_series=emp_series,
            pi_series=phi_norm_series,
            filename=anim_path,
            fps=12,
            stride=anim_stride,
            dpi=150,
            true_phi_map=true_phi_map_time,
            traj_rc=data["traj_rc_agents_all"][:, 0, :],
        )

    else:
        animate_distributions_multi(
            env,
            emp_series=emp_series,
            pi_series=phi_norm_series,
            filename=anim_path,
            fps=12,
            stride=anim_stride,
            dpi=150,
            true_phi_map=true_phi_map_time,
            traj_rc_agents=data["traj_rc_agents_all"],
            active_mask=active_mask,          # NEW
            attrition_log=attrition_log,      # NEW
        )

    print(f"Saved animation to: {anim_path}")




if __name__ == "__main__":
    main()