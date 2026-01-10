# viz/plotting.py  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import cm
from typing import Optional
from env.grid import GridMC, EMPTY, OBSTACLE, ROI, START
from core.types import GRID_CMAP

def plot_grid(env: GridMC,
              filename: Optional[str] = None,
              title: Optional[str] = None,
              dpi: int = 150,
              ax: Optional[plt.Axes] = None,
              field: Optional[np.ndarray] = None,
              legend: bool = True) -> plt.Axes:
    """Draw grid; if `field` is provided, overlay it on ROI cells."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    # base
    base = env.grid.copy()
    base[base == START] = EMPTY
    ax.imshow(base, cmap=GRID_CMAP, origin="upper", vmin=0, vmax=3)

    # choose overlay source:
    # - if field is given, use it
    # - else default to env.true_phi (what you wanted for static figure)
    overlay_src = field if field is not None else env.true_phi

    # overlay only on ROI cells so you can see weights there
    roi_mask = (env.grid == ROI)
    if overlay_src is not None and roi_mask.any():
        vals = np.maximum(overlay_src - 1.0, 0.0)  # your “weight above 1” convention
        if np.any(vals[roi_mask] > 0):
            v = vals[roi_mask]
            vmin, vmax = float(np.min(v)), float(np.max(v))
            cmap = cm.get_cmap("YlOrBr")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = np.zeros((env.H, env.W, 4))
            for r in range(env.H):
                for c in range(env.W):
                    if roi_mask[r, c]:
                        color = cmap(norm(vals[r, c]))
                        rgba[r, c] = (color[0], color[1], color[2], 0.4)
            ax.imshow(rgba, origin="upper")

    # grid lines
    ax.set_xticks(np.arange(-0.5, env.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.H, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])

    if legend:
        handles = [mpatches.Patch(color="#777777", label="Obstacle")]
        if overlay_src is not None and roi_mask.any():
            vals = np.maximum(overlay_src - 1.0, 0.0)
            roi_vals = np.unique(vals[roi_mask])
            roi_vals = roi_vals[roi_vals > 0]
            if roi_vals.size:
                cmap = cm.get_cmap("YlOrBr")
                norm = mcolors.Normalize(vmin=float(roi_vals.min()), vmax=float(roi_vals.max()))
                # put just min/max to keep legend compact
                # for w in [float(roi_vals.min()), float(roi_vals.max())]:
                for w in sorted(roi_vals):
                    color = cmap(norm(w))
                    handles.append(mpatches.Patch(color=color, label=f"ROI (weight≈{w:.1f})"))
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

    if title: ax.set_title(title)
    if filename and created_fig:
        fig.tight_layout(rect=[0, 0, 0.85, 1]); fig.savefig(filename, dpi=dpi); plt.close(fig)
    return ax

def plot_trajectory(env: GridMC,
                    traj_rc: np.ndarray,
                    filename: str,
                    title: Optional[str] = None,
                    stride: int = 1,
                    alpha: float = 0.7,
                    dpi: int = 150):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_grid(env, filename=None, title=None, ax=ax)

    ys = traj_rc[::stride, 0]
    xs = traj_rc[::stride, 1]
    ax.plot(xs, ys, linewidth=1.0, alpha=alpha)
    ax.scatter([xs[0]], [ys[0]], marker="o", s=80, edgecolor="black",
               facecolor="none", linewidths=1.2, label="start")
    ax.scatter([xs[-1]], [ys[-1]], marker="x", s=80, linewidths=2.0, label="end")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def plot_distributions(env: GridMC, emp: np.ndarray, pi: np.ndarray, filename: str, dpi: int = 150):
    occ_emp = np.zeros_like(env.true_phi, dtype=float)
    occ_pi  = np.zeros_like(env.true_phi, dtype=float)
    if getattr(env, "_index_state_map", None) is None:
        env._indexing()
    for i, (r, c) in env._index_state_map.items():
        occ_emp[r, c] = emp[i]
        occ_pi[r,  c] = pi[i]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axs[0].imshow(occ_emp, cmap="viridis"); axs[0].set_title("Empirical distribution"); plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(occ_pi,  cmap="viridis"); axs[1].set_title("Target distribution");    plt.colorbar(im1, ax=axs[1])
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)



def plot_convergence(emp_series, pi, filename: str, true_phi=None, belief_phi_vec=None,cover_time=None):


    def _normalize(v):
        v = np.clip(np.asarray(v, dtype=float), 0.0, None)
        s = v.sum()
        return v / s if s > 0 else v

    errs = [np.abs(emp_series[i] - _normalize(belief_phi_vec[i])).sum() for i in range(len(emp_series))]
    if len(true_phi.shape) > 1:
        norm_true_pi = [_normalize(true_phi[i]) for i in range(true_phi.shape[0])]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(errs, label="Empirical - Belief Map error", color="tab:blue", lw=1.6)

        if true_phi is not None and belief_phi_vec is not None:
            diff = np.array([
                np.abs(_normalize(belief_phi_vec[i]) - norm_true_pi[i]).sum()
                for i in range(len(belief_phi_vec))
            ], dtype=float)
            # ensure same length as errs if needed
            m = min(len(errs), len(diff))
            errs, diff = errs[:m], diff[:m]
            ax.plot(diff, "--", label="Belief − True Map error", color="tab:red", lw=1.6)

            # one common y-scale covering both
            y_min = min(min(errs), diff.min())
            y_max = max(max(errs), diff.max())
            pad = 0.05 * (y_max - y_min + 1e-12)
            ax.set_ylim(y_min - pad, y_max + pad)
    else:
        norm_true_pi = _normalize(true_phi)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(errs, label="Empirical - Belief Map error", color="tab:blue", lw=1.6)

        if true_phi is not None and belief_phi_vec is not None:
            diff = np.array([
                np.abs(_normalize(belief_phi_vec[i]) - norm_true_pi).sum()
                for i in range(len(belief_phi_vec))
            ], dtype=float)
            # ensure same length as errs if needed
            m = min(len(errs), len(diff))
            errs, diff = errs[:m], diff[:m]
            ax.plot(diff, "--", label="Belief − True Map error", color="tab:red", lw=1.6)

            # one common y-scale covering both
            y_min = min(min(errs), diff.min())
            y_max = max(max(errs), diff.max())
            pad = 0.05 * (y_max - y_min + 1e-12)
            ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Error")
    ax.set_title("Convergence")
    ax.legend(loc="upper right")

    if cover_time is not None and cover_time > 0:
        ax.axvline(
            cover_time,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Full coverage"
        )
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def vector_to_field(env: GridMC, v: np.ndarray) -> np.ndarray:
    f = np.zeros_like(env.true_phi, dtype=float)
    for i, (r, c) in env._index_state_map.items():
        f[r, c] = v[i]
    return f

def counts_to_field(env: GridMC, counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    emp = counts/total if total > 0 else np.zeros_like(counts, dtype=float)
    return vector_to_field(env, emp)