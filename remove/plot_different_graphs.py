#!/usr/bin/env python3

# python3 viz/plot_different_graphs.py \
#   --folders \
#   results/T2000_remc_4_modegp_GPupdate25_Pupdate25_gpmean_dyn \
#   results/T2000_remc_4_modegp_GPupdate10_Pupdate10_gpmean_dyn \
#   results/T2000_remc_4_modegp_GPupdate25_Pupdate50_gpmean_dyn \
#   --out results/dynamic_multi


import os, sys, glob, argparse, numpy as np
import matplotlib.pyplot as plt

# Import env (we'll inline vec->grid to match your plot_distributions mapping)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.grid import make_env

def load_npz_all(path):
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    z.close()
    return data

def find_npz(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.npz")),
                   key=os.path.getmtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No .npz file found in: {folder}")
    return paths[0]

def normalize(v):
    v = np.asarray(v, dtype=float).copy()
    v[v < 0] = 0.0
    s = v.sum()
    return v / s if s > 0 else v

def extract_meta(d, fallback_idx):
    p_upd = d.get("p_update", d.get("P_UPDATE_EVERY", None))
    gp_upd = d.get("gp_update", d.get("GP_UPDATE_EVERY", None))
    gp_mode = d.get("gp_mode", "NA")
    sigma = d.get("sensor_sigma", "NA")
    label = f"P{p_upd}|GP{gp_upd}|{gp_mode}|σ{sigma}|run{fallback_idx}"
    return p_upd, gp_upd, gp_mode, sigma, label

def to_scalar(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            v = np.ravel(x)[0]
            return v.item() if hasattr(v, "item") else v
        v = np.ravel(x)[0]
        return v.item() if hasattr(v, "item") else v
    return x

def compute_series(d):
    if "true_phi_map_time" in d:
        true_phi = np.asarray(d["true_phi_map_time"])
        belief_phi_vec = np.asarray(d["belief_phi_map"])
        emp_series = np.asarray(d["emp_series"])
        norm_true = normalize(true_phi)
        T = min(len(emp_series), len(belief_phi_vec))
        emp_bel, bel_true = [], []
        for t in range(T):
            emp = np.asarray(emp_series[t], dtype=float)
            bel = np.asarray(belief_phi_vec[t], dtype=float)
            emp_bel.append(np.abs(emp - normalize(bel)).sum())
            bel_true.append(np.abs(normalize(bel) - normalize(true_phi[t])).sum())
        return np.asarray(emp_bel), np.asarray(bel_true)

    else:
        true_phi = np.asarray(d["true_phi_vec"])

        belief_phi_vec = np.asarray(d["belief_phi_map"])
        emp_series = np.asarray(d["emp_series"])
        norm_true = normalize(true_phi)
        T = min(len(emp_series), len(belief_phi_vec))
        emp_bel, bel_true = [], []
        for t in range(T):
            emp = np.asarray(emp_series[t], dtype=float)
            bel = np.asarray(belief_phi_vec[t], dtype=float)
            emp_bel.append(np.abs(emp - normalize(bel)).sum())
            bel_true.append(np.abs(normalize(bel) - norm_true).sum())
        return np.asarray(emp_bel), np.asarray(bel_true)

def vec_to_grid(env, vec):
    """Map flat distribution (over free cells) to HxW grid exactly like your plot_distributions."""
    occ = np.zeros_like(env.true_phi, dtype=float)
    if getattr(env, "_index_state_map", None) is None:
        env._indexing()
    for i, (r, c) in env._index_state_map.items():
        occ[r, c] = float(vec[i])
    return occ

def main():
    ap = argparse.ArgumentParser(description="Convergence small-multiples + combined distributions figure (top half | bottom half).")
    ap.add_argument("--folders", nargs="+", required=True, help="Folders each containing a run_*.npz")
    ap.add_argument("--outdir", default="viz_out", help="Output directory")
    ap.add_argument("--max_small_multiples", type=int, default=36, help="Max panels in convergence grid")
    ap.add_argument("--show_vline", action="store_true", help="Draw a faint vertical line at cover_time")
    ap.add_argument("--seed", type=int, default=1, help="Env seed to match run()")
    ap.add_argument("--neighbors", default="4", help="Override neighbors (else read from npz)")
    ap.add_argument("--dist_fig", default="all_runs_emp_pi_split.png", help="Output image name for combined distributions figure")
    ap.add_argument("--dist_dpi", type=int, default=220, help="DPI for the combined distributions figure")
    ap.add_argument("--tile_w", type=float, default=2.5, help="Width (inches) of each small tile")
    ap.add_argument("--tile_h", type=float, default=2.2, help="Height (inches) of each small tile")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Data containers
    panels = []      # for convergence grid
    dist_rows = []   # each item: dict(label, occ_emp, occ_pi)

    idx_counter = 0

    # ---- Load runs ----
    for folder in args.folders:
        try:
            npz_path = find_npz(folder)
            d = load_npz_all(npz_path)
        except Exception as e:
            print(f"[skip] {folder}: {e}")
            continue

        # Convergence series
        try:
            e_emp_bel, e_bel_true = compute_series(d)
        except Exception as e:
            print(f"[skip] {npz_path}: cannot compute series ({e})")
            continue
        if len(e_emp_bel) == 0:
            print(f"[skip] {npz_path}: empty series")
            continue

        p_upd, gp_upd, gp_mode, sigma, label = extract_meta(d, idx_counter)
        idx_counter += 1
        cover_time = d.get("cover_time", None)
        x = np.arange(len(e_emp_bel))
        panels.append((p_upd, gp_upd, label, x, e_emp_bel, e_bel_true, cover_time, gp_mode, sigma))

        # Final distributions (exactly like your plot_distributions inputs)
        emp_final = np.asarray(d["emp_final"], dtype=float)
        pi_final  = np.asarray(d["pi_vec_final"], dtype=float)
        if emp_final.sum() > 0: emp_final = emp_final / emp_final.sum()
        if pi_final.sum()  > 0: pi_final  = pi_final  / pi_final.sum()

        neighbors = args.neighbors if args.neighbors is not None else d.get("neighbors", "4")
        neighbors = to_scalar(neighbors)

        env = make_env(neighbors=neighbors, seed=args.seed)
        occ_emp = vec_to_grid(env, emp_final)
        occ_pi  = vec_to_grid(env, pi_final)

        row_label = f"P{to_scalar(p_upd)}|GP{to_scalar(gp_upd)}|{gp_mode}|σ{to_scalar(sigma)}"
        dist_rows.append(dict(label=row_label, occ_emp=occ_emp, occ_pi=occ_pi))

        print(f"[ok] {folder} -> {os.path.basename(npz_path)}  len={len(x)}  {label}")

    if not panels:
        raise RuntimeError("No valid runs processed.")

    # ---- Combined Distributions figure: split runs into top/bottom halves ----
    N = len(dist_rows)
    rows = 3
    runs_per_row = int(np.ceil(N / rows))
    cols = runs_per_row * 2  # Emp | Target for each run

    fig_w = args.tile_w * cols
    fig_h = args.tile_h * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    # Place tiles: each run uses two adjacent subplots (Emp | Target)
    for k, row in enumerate(dist_rows):
        r = min(k // runs_per_row, rows - 1)
        idx_in_row = k % runs_per_row
        c_emp = 2 * idx_in_row
        c_pi  = c_emp + 1

        im0 = axes[r, c_emp].imshow(row["occ_emp"], cmap="viridis")  # own autoscale
        axes[r, c_emp].set_title("Empirical distribution", fontsize=9)
        plt.colorbar(im0, ax=axes[r, c_emp])

        im1 = axes[r, c_pi].imshow(row["occ_pi"], cmap="viridis")    # own autoscale
        axes[r, c_pi].set_title("Target distribution", fontsize=9)
        plt.colorbar(im1, ax=axes[r, c_pi])

        axes[r, c_emp].set_ylabel(row["label"], fontsize=8)

    # Hide unused axes in partially filled rows
    for r in range(rows):
        start = r * runs_per_row
        end = min(N, start + runs_per_row)
        active = max(0, end - start)
        for j in range(active, runs_per_row):
            axes[r, 2*j].axis("off")
            axes[r, 2*j + 1].axis("off")

    fig.tight_layout()
    out_dist = os.path.join(args.outdir, args.dist_fig)
    fig.savefig(out_dist, dpi=args.dist_dpi)
    plt.close(fig)
    print(f"[saved] {out_dist}")

    # ---- Convergence small-multiples (same as before) ----
    n = min(args.max_small_multiples, len(panels))
    panels = panels[:n]
    cols = int(np.ceil(np.sqrt(n)))
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2*cols, 2.4*rows_n), squeeze=False)

    for ax, panel in zip(axes.ravel(), panels):
        P, G, label, x, e1, e2, ctime, gp_mode, sigma = panel
        (line,) = ax.plot(x, e1, lw=1.4, label="Emp−Belief")
        ax.plot(x, e2, "--", lw=1.2, color=line.get_color(), label="Belief−True")

        if ctime is not None:
            try:
                c = int(np.asarray(ctime).item())
            except Exception:
                c = int(ctime)
            if 0 <= c < len(e2):
                y = float(e2[c])
                ax.scatter([c], [y], s=40, facecolors="none",
                           edgecolors=line.get_color(), linewidths=1.6, zorder=5)
                ax.annotate(f"t={c}", (c, y),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7, color=line.get_color(),
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec="none", alpha=0.75))
                if args.show_vline:
                    ax.axvline(c, color=line.get_color(), alpha=0.25, lw=1.0, zorder=4)

        P_s = to_scalar(P); G_s = to_scalar(G); sigma_s = to_scalar(sigma)
        ax.set_title(f"P{P_s}|GP{G_s}|{gp_mode}|σ{sigma_s}", fontsize=8)
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    axes[0, 0].legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    out_sm = os.path.join(args.outdir, "small_multiples.png")
    fig.savefig(out_sm, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_sm}")

if __name__ == "__main__":
    main()
