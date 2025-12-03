#!/usr/bin/env python3
import os
import argparse
import numpy as np

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.grid import make_env                
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

def main():
    ap = argparse.ArgumentParser(description="Plot saved ergodic/GP run from NPZ.")
    # ap.add_argument("--npz", required=True, help="Path to results/run_*.npz", default='results/RECM_GPupdate25_noise0p5_Pupdate25')
    ap.add_argument("--folder",  help="Path to folder containing a results/run_*.npz", default='MAES/results/T2000_remc_4_modegp_GPupdate25_Pupdate25_gpmean_dynamic/')
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


    stem = data["stem"]
    neighbors = args.neighbors if args.neighbors is not None else data["neighbors"]
    outdir = ensure_dir(args.outdir or os.path.join(args.folder, f"plots_{stem}"))


    env = make_env(neighbors=neighbors, seed=args.seed)

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
    plot_convergence(
        data["emp_series"],          # iterable of vectors
        data["pi_vec_final"],
        filename=os.path.join(outdir, f"{stem}_convergence.png"),
        true_phi=true_phi,
        belief_phi_vec=data["belief_phi_map"],
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
    pi_series  = data["pi_series"]
    if emp_series.shape[0] != pi_series.shape[0]:
        m = min(emp_series.shape[0], pi_series.shape[0])
        emp_series = emp_series[:m]
        pi_series  = pi_series[:m]
        frame_indices = [i for i in frame_indices if i < m]

    for i in frame_indices:
        plot_distributions(
            env,
            emp_series[i],
            pi_series[i],
            filename=os.path.join(outdir, f"{stem}__t{i}_emp_vs_pi.png"),
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