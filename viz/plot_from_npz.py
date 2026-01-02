#!/usr/bin/env python3
import os
import argparse
import numpy as np
import glob
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.grid import make_env, make_env_from_json              
from viz.plotting import (
    plot_grid,
    plot_trajectory,
    plot_distributions,
    plot_convergence,
)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}  # load all keys
    z.close()
    return data

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_multi(env,
                           traj_rc_agents_all,   # (T, n_agents, 2)
                           pi_vec,               # (nstates,)
                           filename,
                           stride=1,
                           title="Convergence (multi-agent)"):
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

    plt.xlabel("time step")
    plt.ylabel(r"$\| r_{\mathrm{hat}} - \pi \|_2$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot saved ergodic/GP run from NPZ.")
    # ap.add_argument("--npz", required=True, help="Path to results/run_*.npz", default='results/RECM_GPupdate25_noise0p5_Pupdate25')
    ap.add_argument("--folder",  help="Path to folder containing a results/run_*.npz", default='results/T300_ergodic_remc_GPupdate25_Pupdate25_gpmean_static_3_commModelocal_comm_radius2.0_concenIter5_old')
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


    stem = data["stem"]
    stem = os.path.basename(str(data["stem"]))
    print(f"stem filename: {stem}")
    neighbors = args.neighbors if args.neighbors is not None else data["neighbors"]
    # outdir = ensure_dir(args.outdir or os.path.join(args.folder, f"plots_{stem}"))
    outdir = ensure_dir(args.outdir or os.path.join(args.folder, "plots"))
    print(f"Output directory: {outdir}")
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


    plot_grid(env, filename=os.path.join(outdir, f"{stem}__grid_true.png"),
              title="Grid (true info map)")

    # --- Trajectory ---
    plot_trajectory(env, data["traj_rc_all"],
                    filename=os.path.join(outdir, f"{stem}__trajectory.png"),
                    title="Trajectory (from saved run)")

    # --- Final distributions ---
    plot_distributions(env, data["emp_final"], data["pi_vec_final"],
                       filename=os.path.join(outdir, f"{stem}__emp_vs_pi_final.png"))

    # --- Convergence plot ---
    if data.get("true_phi_map_time", None) is not None:
        true_phi = data["true_phi_map_time"]
    else:       
        true_phi = data["true_phi_vec"]

    print("env.nstates =", env.nstates)
    # print("traj_rc_agents_all shape =", data["traj_rc_agents_all"].shape)
    # print("min/max r =", data["traj_rc_agents_all"][...,0].min(), data["traj_rc_agents_all"][...,0].max())
    # print("min/max c =", data["traj_rc_agents_all"][...,1].min(), data["traj_rc_agents_all"][...,1].max())
    if n_agents > 1 and "traj_rc_agents_all" in data and data["traj_rc_agents_all"].size > 0:
        print("traj_rc_agents_all shape =", data["traj_rc_agents_all"].shape)
        print("min/max r =", data["traj_rc_agents_all"][..., 0].min(), data["traj_rc_agents_all"][..., 0].max())
        print("min/max c =", data["traj_rc_agents_all"][..., 1].min(), data["traj_rc_agents_all"][..., 1].max())
    else:
        print("Single-agent run (or no traj_rc_agents_all saved). Using traj_rc_all.")
        print("traj_rc_all shape =", data["traj_rc_all"].shape)




    if n_agents == 1:
        plot_convergence(
            data["emp_series"],          # (T, nstates)
            data["pi_vec_final"],
            filename=os.path.join(outdir, f"{stem}_convergence.png"),
            true_phi=true_phi,
            belief_phi_vec=data["belief_phi_map"],
        )
    else:
        assert "traj_rc_agents_all" in data and data["traj_rc_agents_all"].size > 0, \
            "Multi-agent run requires traj_rc_agents_all in NPZ"

        plot_convergence_multi(
            env,
            data["traj_rc_agents_all"],
            data["pi_vec_final"],  # team target
            filename=os.path.join(outdir, f"{stem}_convergence_multi.png"),
            stride=min(int(data["gp_update"]), int(data["p_update"]))
        )


    # # --- Chunked per-step emp vs pi frames ---
    T = data["emp_series"].shape[0]
    # stride = max(1, int(args.stride))
    stride = min(data["gp_update"], data["p_update"])  # ensure we don't skip updates
    frame_indices = list(range(0, T, stride))
    if args.limit is not None:
        frame_indices = frame_indices[: args.limit]

    # Guard: pi_series rows should align with emp_series rows
    emp_series = data["emp_series"]
    phi_norm_series  = data["phi_norm_series"]
    if emp_series.shape[0] != phi_norm_series.shape[0]:
        m = min(emp_series.shape[0], phi_norm_series.shape[0])
        emp_series = emp_series[:m]
        phi_norm_series  = phi_norm_series[:m]
        frame_indices = [i for i in frame_indices if i < m]

    for i in frame_indices:
        plot_distributions(
            env,
            emp_series[i],
            phi_norm_series[i],
            filename=os.path.join(outdir, f"{stem}__t{i}_emp_vs_phi_norm.png"),
        )

    if data["gp_update_steps"].size > 0 or data["p_update_steps"].size > 0:
        with open(os.path.join(outdir, f"{stem}__updates.txt"), "w") as f:
            if data["gp_update_steps"].size > 0:
                f.write("GP update steps:\n")
                f.write(", ".join(map(str, data["gp_update_steps"].tolist())) + "\n\n")
            if data["p_update_steps"].size > 0:
                f.write("P rebuild steps:\n")
                f.write(", ".join(map(str, data["p_update_steps"].tolist())) + "\n")

    print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    main()