#!/usr/bin/env python3
# run
# python viz/plot_multiple.py --folders results/RECM_GPupdate25_noise0p5_Pupdate_200 results/RECM_GPupdate5_noise0p5_Pupdate_200 results/RECM_GPupdate25_noise0p5_Pupdate_25  --out results/multi_convergence.png
# python viz/plot_multiple.py --folders results/RECM_GPupdate25_noise0p5_Pupdate_25_mean results/RECM_GPupdate25_noise0p5_Pupdate_25_ucb results/RECM_GPupdate25_noise0p5_Pupdate_25_variance --out results/multi_convergence_update_method.png

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_npz_all(path):
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    z.close()
    return data

def find_npz(folder):
    # pick newest .npz in folder
    paths = sorted(glob.glob(os.path.join(folder, "*.npz")),
                   key=os.path.getmtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No .npz file found in: {folder}")
    return paths[0]

def main():
    ap = argparse.ArgumentParser(description="Plot multiple runs' convergence (error_series) on one plot.")
    ap.add_argument("--folders", nargs="+", required=True,
                    help="Each folder should contain a run_*.npz to read.")
    ap.add_argument("--out", default="multi_convergence.png",
                    help="Output PNG path (default: multi_convergence.png).")
    ap.add_argument("--title", default="Convergence across runs",
                    help="Figure title.")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_plotted = False
    def _normalize(v):
        v = np.clip(np.asarray(v, dtype=float), 0.0, None)
        s = v.sum()
        return v / s if s > 0 else v

    for folder in args.folders:
        try:
            npz_path = find_npz(folder)
            data = load_npz_all(npz_path)
        except Exception as e:
            print(f"[skip] {folder}: {e}")
            continue

        
        pi = np.asarray(data["pi_vec_final"], dtype=float)
        true_phi = np.asarray(data["true_phi_vec"], dtype=float)
        norm_true_pi = _normalize(true_phi)
        emp_series = np.asarray(data["emp_series"], dtype=float)
        belief_phi_vec = np.asarray(data["belief_phi_map"], dtype=float)
        errs = [np.abs(emp_series[i] - _normalize(belief_phi_vec[i])).sum() for i in range(len(emp_series))]

        # if "error_series" in data:
        #     errs = np.asarray(data["error_series"], dtype=float).ravel()
        # elif "emp_series" in data and "pi_vec_final" in data:
        #     emp_series = np.asarray(data["emp_series"], dtype=float)
        #     pi = np.asarray(data["pi_vec_final"], dtype=float)
        #     errs = np.array([np.abs(emp - pi).sum() for emp in emp_series], dtype=float)
        # else:
        #     print(f"[skip] {npz_path}: no error_series and cannot recompute")
        #     continue

        if len(errs) == 0:
            print(f"[skip] {npz_path}: empty error_series")
            continue

        # legend label
        p_upd = data.get("p_update", data.get("P_UPDATE_EVERY", "NA"))
        gp_upd = data.get("gp_update", data.get("GP_UPDATE_EVERY", "NA"))
        gp_method = data.get("gp_mode", "NA")
        sigma = data.get("sensor_sigma", "NA")
        label = f"P{p_upd} | GP{gp_upd} | noise={sigma} | {gp_method}"

        # ax.plot(np.arange(len(errs)), errs, lw=1.6, label=label)
        x = np.arange(len(errs))

        (line_rpi,) = ax.plot(x, errs, lw=1.6, label=f"{label}  (Emp - Belief)")
        any_plotted = True
        diff = np.array([np.abs(_normalize(belief_phi_vec[i]) - norm_true_pi).sum()
        for i in range(len(belief_phi_vec))], dtype=float)
        ax.plot(diff, "--", lw=1.2,
                color=line_rpi.get_color(),
                label=f"{label}  (Belief−True)")
        covered_map_time = data.get("cover_time", None)
        ax.scatter(
            covered_map_time, diff[covered_map_time],
            marker="o",
            s=80,
            edgecolors=line_rpi.get_color(),
            facecolors="none",
            linewidths=1.8,
            zorder=5,
        )
        # ax.plot(
        #     covered_map_time, diff[covered_map_time],
        #     marker="o",
        #     markersize=8,
        #     markeredgecolor=line_rpi.get_color(),
        #     markerfacecolor="none",
        #     markeredgewidth=1.8,
        #     zorder=5,
        # )
        print(f"[ok] {folder} -> {os.path.basename(npz_path)}   len={len(errs)}   {label}")



    if not any_plotted:
        raise RuntimeError("No valid runs plotted.")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Error")
    ax.set_title(args.title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()