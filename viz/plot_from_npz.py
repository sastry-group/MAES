#!/usr/bin/env python3
import os
import argparse
import numpy as np
import glob
import sys

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

from viz.metrics import full_map_cover_time, roi_first_visit_per_cfg, empirical_from_traj


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



def plot_convergence_multi(env,
                           traj_rc_agents_all,   # (T, n_agents, 2)
                           pi_vec,               # (nstates,)
                           filename,
                           stride=1,
                           title="Convergence (multi-agent)",
                           cover_time=None):
    """
    Plots per-agent || r_hat_a(t) - pi ||_2 over time and optionally the team average.

    Uses traj_rc_agents_all to reconstruct per-agent visitation counts.
    """
    traj = np.asarray(traj_rc_agents_all)
    assert traj.ndim == 3 and traj.shape[2] == 2, f"Expected (T,n_agents,2), got {traj.shape}"
    T, n_agents, _ = traj.shape
    nstates = int(env.nstates)

    pi_vec = np.asarray(pi_vec, dtype=float)
    pi_vec = pi_vec / max(pi_vec.sum(), 1e-12)

    # map (r,c) -> state index once
    # env._state_index_map expects a tuple(int,int)
    idx = np.empty((T, n_agents), dtype=int)
    for t in range(T):
        for a in range(n_agents):
            r, c = traj[t, a]
            idx[t, a] = env._state_index_map[(int(r), int(c))]

    # cumulative counts per agent
    counts_a = np.zeros((n_agents, nstates), dtype=float)
    errs_a = [ [] for _ in range(n_agents) ]

    # optional: team curve
    counts_team = np.zeros(nstates, dtype=float)
    errs_team = []

    for t in range(T):
        # update all agents at this environment time t
        for a in range(n_agents):
            s = idx[t, a]
            counts_a[a, s] += 1.0
            counts_team[s] += 1.0

        if (t % stride) == 0:
            # per-agent errors
            for a in range(n_agents):
                r_hat_a = counts_a[a] / max(counts_a[a].sum(), 1e-12)
                errs_a[a].append(np.linalg.norm(r_hat_a - pi_vec))

            # team error (all visits pooled)
            r_hat_team = counts_team / max(counts_team.sum(), 1e-12)
            errs_team.append(np.linalg.norm(r_hat_team - pi_vec))

    x = np.arange(len(errs_team)) * stride

    plt.figure(figsize=(10, 5))
    for a in range(n_agents):
        plt.plot(x, errs_a[a], label=f"agent {a}")

    plt.plot(x, errs_team, linewidth=3, label="team (pooled)")
    if cover_time is not None and cover_time > 0:
        plt.axvline(
            cover_time,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Full coverage"
        )


    plt.xlabel("time step")
    plt.ylabel(r"$\| r_{\mathrm{hat}} - \hat{\phi} \|_2$")

    plt.title(title)
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
   





def main():
    ap = argparse.ArgumentParser(description="Plot saved ergodic/GP run from NPZ.")
    # ap.add_argument("--npz", required=True, help="Path to results/run_*.npz", default='results/RECM_GPupdate25_noise0p5_Pupdate25')
    ap.add_argument("--folder",  help="Path to folder containing a results/run_*.npz", default='results/T300_mac_dt_GPupdate50_v0_gpucb_static_3_start23x3_commModelocal_comm_radius5.0_concenIter5')
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside the npz)")
    ap.add_argument("--stride", type=int, default=50, help="Frame stride when dumping per-step emp vs pi (default: 50)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of frames to export (default: all with stride)")
    ap.add_argument("--seed", type=int, default=1, help="Env seed to match the run() script (default: 1)")
    ap.add_argument("--neighbors", default="4", help="Override neighbors (else read from npz)")
    args = ap.parse_args()


    npz_files = [f for f in os.listdir(args.folder) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {args.folder}")
    if len(npz_files) > 1:
        print(f"Multiple .npz files found, using the first: {npz_files[0]}")
    npz_path = os.path.join(args.folder, npz_files[0])
    data = load_npz(npz_path)


    n_agents = int(data.get("n_agents", 1))
    planner = str(data.get("planner", "ergodic"))
    if planner == "mac_dt":
        stride = int(data.get("gp_update"))
    else:
        stride = min(
            int(data.get("gp_update")),
            int(data.get("p_update")),
        )
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
            plot_trajectory(
                env,
                data["traj_rc_agents_all"][:, a, :],
                filename=os.path.join(outdir, f"{stem}__trajectory_agent{a}.png"),
                title=f"Trajectory (agent {a})",
            )

    # --- Final distributions ---
    plot_distributions(env, emp_final, data["pi_vec_final"],
                       filename=os.path.join(outdir, f"{stem}__emp_vs_pi_final.png"))

    # --- Convergence plot ---
    plot_convergence_multi(
        env,
        data["traj_rc_agents_all"],
        data["pi_vec_final"],  # team target
        filename=os.path.join(outdir, f"{stem}_convergence_multi.png"),
        stride=stride,
        cover_time=data.get("cover_time", -1),
    )


    # plotting distributions 
    traj_states = data["traj_states"]
    belief_phi = data["belief_phi_map"]          # (T, nstates)
    phi_norm_series = belief_phi / belief_phi.sum(axis=1, keepdims=True)
    T = traj_states.shape[0]

    frame_indices = list(range(0, T, stride))
    if args.limit is not None:
        frame_indices = frame_indices[:args.limit]

    for k in frame_indices:
        emp_k = empirical_from_traj(traj_states, env, k)
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
        traj_states=data["traj_states"],
        traj_rc_agents_all=data["traj_rc_agents_all"],
        steps_per_cfg=int(data.get("steps_per_cfg", T)),  # make sure you saved this; else fallback
        env_dyn=str(data.get("env_dyn", "static")),
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
        )

    print(f"Saved animation to: {anim_path}")




if __name__ == "__main__":
    main()