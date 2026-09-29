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
              legend: bool = True,
              roi_vmin: Optional[float] = None,
              roi_vmax: Optional[float] = None) -> plt.Axes:
            
    """

    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure


    base = env.grid.copy()
    base[base == START] = EMPTY
    ax.imshow(base, cmap=GRID_CMAP, origin="upper", vmin=0, vmax=1)


    overlay_src = field if field is not None else env.true_phi
    roi_mask = (env.grid == ROI)
    obstacle_mask = (env.grid == OBSTACLE)

    def truncate_cmap(cmap, minval=0.25, maxval=0.9, n=256):
        return mcolors.LinearSegmentedColormap.from_list(
            "trunc", cmap(np.linspace(minval, maxval, n))
        )

    cmap = truncate_cmap(cm.get_cmap("YlOrBr"), 0.25, 0.9)
    norm = None
    # roi_min = roi_max = None

    # -------------------------------------------------
    # ROI overlay (build cmap + norm ONCE)
    # -------------------------------------------------
    if overlay_src is not None and roi_mask.any():
        vals = np.maximum(overlay_src - 1.0, 0.0)
        max_roi_weight = max(roi['weight'] for roi in env.roi_cells)
        vals = np.minimum(vals, max_roi_weight)
        roi_vals = vals[roi_mask]
        roi_vals = roi_vals[roi_vals > 0]
        # print("unique overlay_src values:", np.unique(overlay_src))
        # print("unique vals (overlay_src - 1):", np.unique(vals))
        # print("roi_vals unique:", np.unique(roi_vals))
        # print("roi_min, roi_max:", roi_min, roi_max)
        # print("-------------------")

        if roi_vals.size > 0:
            # roi_min = float(roi_vals.min())
            # roi_max = float(roi_vals.max())
            roi_min = float(roi_vals.min()) if roi_vmin is None else roi_vmin
            roi_max = float(roi_vals.max()) if roi_vmax is None else roi_vmax
            norm = mcolors.Normalize(vmin=roi_min, vmax=roi_max)

            rgba = np.zeros((env.H, env.W, 4))
            for r in range(env.H):
                for c in range(env.W):
                    if roi_mask[r, c] and vals[r, c] > 0:
                        color = cmap(norm(vals[r, c]))
                        rgba[r, c] = (color[0], color[1], color[2], 1.0)

            ax.imshow(rgba, origin="upper")

    obs_r, obs_c = np.where(obstacle_mask)
    ax.scatter(
        obs_c, obs_r,
        marker="x",
        color="black",
        s=80,
        linewidths=2,
        label="No-Fly Zone",

    )


    ax.set_xticks(np.arange(-0.5, env.W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.4, alpha=0.2, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks([])


    if legend and norm is not None:
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label("Region of Interest Weight", fontsize=16)

        # Small legend only for obstacles
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            fontsize=16
        )

        cbar.set_ticks([roi_min, roi_max])
        cbar.set_ticklabels(["Low", "High"])

    # if title:
    #     ax.set_title(title)

    if filename and created_fig:
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

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


def plot_belief_per_agent(env: GridMC,
                          belief_per_agent: np.ndarray,   # (N, nstates)
                          filename: Optional[str] = None,           # for grid mode
                          filename_template: Optional[str] = None,  # for per-agent mode (must contain "{agent}")
                          mode: str = "grid",                       # "grid" or "separate"
                          dpi: int = 150,
                          normalize: bool = True):
    """
    Plot final belief distribution per agent.

    mode="grid"      → one figure with N subplots, saved to `filename`.
    mode="separate"  → one figure per agent, saved using `filename_template`
                       (must contain "{agent}", e.g. ".../run__belief_agent{agent}.png").
    """
    if getattr(env, "_index_state_map", None) is None:
        env._indexing()

    N = belief_per_agent.shape[0]

    if normalize:
        beliefs = np.stack([b / max(b.sum(), 1e-12) for b in belief_per_agent], axis=0)
    else:
        beliefs = belief_per_agent

    # Shared color scale across agents so figures are visually comparable
    vmax = float(beliefs.max()) if beliefs.size else 1.0

    def _field_for(a):
        f = np.zeros_like(env.true_phi, dtype=float)
        for i, (r, c) in env._index_state_map.items():
            f[r, c] = beliefs[a, i]
        return f

    if mode == "grid":
        if filename is None:
            raise ValueError("mode='grid' requires `filename`.")
        ncols = int(np.ceil(np.sqrt(N)))
        nrows = int(np.ceil(N / ncols))
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(4 * ncols, 4 * nrows),
                                squeeze=False)
        for a in range(N):
            ax = axs[a // ncols][a % ncols]
            # im = ax.imshow(_field_for(a), cmap="viridis", vmin=0.0, vmax=vmax, origin="upper")
            im = ax.imshow(_field_for(a), cmap="viridis", origin="upper")
            ax.set_title(f"Agent {a}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for a in range(N, nrows * ncols):
            axs[a // ncols][a % ncols].axis("off")
        fig.suptitle("Final belief distribution per agent", fontsize=14)
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    elif mode == "separate":
        if filename_template is None or "{agent}" not in filename_template:
            raise ValueError("mode='separate' requires `filename_template` containing '{agent}'.")
        for a in range(N):
            fig, ax = plt.subplots(figsize=(6, 6))
            # im = ax.imshow(_field_for(a), cmap="viridis", vmin=0.0, vmax=vmax, origin="upper")
            im = ax.imshow(_field_for(a), cmap="viridis", origin="upper")
            ax.set_title(f"Final belief — Agent {a}", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(filename_template.format(agent=a), dpi=dpi)
            plt.close(fig)

    else:
        raise ValueError(f"Unknown mode={mode!r}; use 'grid' or 'separate'.")



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




def plot_grid_with_starts(env: GridMC,
                          starts: list[tuple[int,int]],
                          filename: str,
                          title: Optional[str] = None,
                          dpi: int = 150):
    """Plot grid with agent starting positions marked."""
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_grid(env, filename=None, title=None, ax=ax, legend=True)

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan",
    ]
    markers = ["o", "s", "D", "^", "X", "P", "v", "<", ">", "h"]

    for i, (r, c) in enumerate(starts):
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.scatter(
            c, r,
            marker=marker,
            s=120,
            color=color,
            edgecolors="black",
            linewidths=1.2,
            zorder=10,
            label=f"Agent {i} ({r},{c})",
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(starts), 4),
        frameon=False,
        fontsize=8,
    )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
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