import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib import cm
from typing import Optional
from env.grid import GridMC, EMPTY, OBSTACLE, ROI, START
from core.types import GRID_CMAP
from matplotlib import animation
from viz.plotting import plot_grid


def animate_trajectory(env: GridMC,
                       traj_rc: np.ndarray,
                       filename: str,
                       fps: int = 10,
                       stride: int = 1,
                       dpi: int = 150,
                       show_start_end: bool = True):
    """
    Animate the drone moving on the same grid drawn by plot_grid().
    - env: GridMC
    - traj_rc: ndarray of shape (T, 2) with (r, c) per step
    - filename: output .mp4 or .gif
    """
    # base figure/axes using your exact grid renderer
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_grid(env, filename=None, title=None, ax=ax, legend=False)

    # Precompute path (subsampled if stride > 1)
    ys = traj_rc[::stride, 0]
    xs = traj_rc[::stride, 1]
    T = len(xs)

    # Artists: trail line + moving point
    line, = ax.plot([], [], linewidth=1.2, alpha=0.9, zorder=4)
    point, = ax.plot([], [], marker="o", markersize=7,
                     markerfacecolor="none", markeredgecolor="black",
                     linewidth=1.2, zorder=5)

    # start/end markers (static)
    if show_start_end and T >= 1:
        ax.scatter([xs[0]], [ys[0]], marker="o", s=80, edgecolor="black",
                   facecolor="none", linewidths=1.2, zorder=5, label="start")
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=80, linewidths=2.0,
                   zorder=5, label="end")
        ax.legend(loc="upper right")

    timestamp_text = ax.text(
        0.5, -0.08, "", transform=ax.transAxes,
        ha="center", va="top", fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )


    def init():
        line.set_data([], [])
        point.set_data([], [])
        timestamp_text.set_text("")
        return line, point, timestamp_text

    def update(t):
        # Draw path up to step t and current point
        line.set_data(xs[:t+1], ys[:t+1])
        point.set_data([xs[t]], [ys[t]])
        timestamp_text.set_text(f"t = {t*stride} steps")
        return line, point, timestamp_text

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=T, interval=1000//max(fps,1), blit=False  # blit=False avoids scatter redraw glitches
    )

    # Save
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            fig.tight_layout()
            anim.save(filename, writer=writer, dpi=dpi)
        except Exception:
            # fall back to GIF if ffmpeg not present
            gif = filename.rsplit(".", 1)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)
            fig.tight_layout()
            anim.save(gif, writer=writer, dpi=dpi)
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        fig.tight_layout()
        anim.save(filename, writer=writer, dpi=dpi)
    else:
        raise ValueError("filename must end with .mp4 or .gif")

    plt.close(fig)



def _vec_to_grid(env, vec):

    """Project a state-vector onto the env grid shape."""
    M = np.zeros_like(env.true_phi, dtype=float)
    for i, (r, c) in env._index_state_map.items():
        M[r, c] = vec[i]
    return M

def animate_distributions(env,
                          emp_series,         # list/array of length T, each an (nstates,) empirical distro
                          pi_series,          # list/array of length T, each an (nstates,) target distro
                          filename,
                          fps=12,
                          stride=1,
                          dpi=150,
                          true_phi_map: Optional[np.ndarray] = None):
    """
    Create a side-by-side animation of empirical vs target distributions over time.
    """
    emp_series = emp_series[::stride]
    pi_series  =  pi_series[::stride]
    T = len(emp_series)
    assert T == len(pi_series) and T > 0, "emp_series and pi_series must be same nonzero length"

    if true_phi_map is not None:
        true_series = true_phi_map[::stride]
        # In case original true_series is longer/shorter, truncate to common length
        T = min(T, len(true_series))
        emp_series = emp_series[:T]
        pi_series  = pi_series[:T]
        true_series = true_series[:T]
    else:
        true_series = None

    # Precompute grids for consistent color scale
    emp_grids = [ _vec_to_grid(env, e) for e in emp_series ]
    pi_grids  = [ _vec_to_grid(env, p) for p in pi_series  ]


    # # Fix color scale across frames for visual consistency
    # vmax = max(np.max([g.max() for g in emp_grids]),
    #            np.max([g.max() for g in pi_grids]))
    # vmax = max(vmax, 1e-12)  # avoid zero vmax

    pi_vmin = 0.0
    pi_vmax = max(g.max() for g in pi_grids)
    pi_vmax = max(pi_vmax, 1e-12)  # avoid zero range
    shared_norm = Normalize(vmin=pi_vmin, vmax=pi_vmax)


    # --- Save a base grid *without* ROIs so we can rebuild ROIs from true_phi_map each frame ---
    base_no_roi = env.grid.copy()
    base_no_roi[base_no_roi == ROI] = EMPTY

    # --- Helper: get field grid for time t (still in vector form externally) ---
    def _get_true_field_grid(t_idx: int) -> Optional[np.ndarray]:
        if true_series is None:
            return None
        v = np.asarray(true_series[t_idx])
        if v.ndim == 1:
            v = _vec_to_grid(env, v)   # inverse of build_map_vec
        return v

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax_emp, ax_pi, ax_true = axs
    # im0 = axs[0].imshow(emp_grids[0], cmap="viridis", vmin=0.0, vmax=vmax)
    im0 = ax_emp.imshow(emp_grids[0], cmap="viridis", norm=shared_norm)
    ax_emp.set_title("Empirical distribution")
    plt.colorbar(im0, ax=ax_emp)

    # im1 = axs[1].imshow(pi_grids[0], cmap="viridis", vmin=0.0, vmax=vmax)
    im1 = ax_pi.imshow(pi_grids[0], cmap="viridis",norm=shared_norm)
    ax_pi.set_title("Target distribution")
    plt.colorbar(im1, ax=ax_pi)

    # Helper to get correct-shaped field for plot_grid at time t
    def _get_true_field(t_idx: int):
        if true_series is None:
            return None
        arr = np.asarray(true_series[t_idx])
        # If it's a 1D vector over states, reshape to (H, W)
        if arr.ndim == 1:
            arr = _vec_to_grid(env, arr)
        return arr

    # --- Figure & axes ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax_emp, ax_pi, ax_true = axs

    # Left: empirical heatmap
    im0 = ax_emp.imshow(emp_grids[0], cmap="viridis", norm=shared_norm)
    ax_emp.set_title("Empirical distribution")
    plt.colorbar(im0, ax=ax_emp)

    # Middle: target heatmap
    im1 = ax_pi.imshow(pi_grids[0], cmap="viridis", norm=shared_norm)
    ax_pi.set_title("Target distribution")
    plt.colorbar(im1, ax=ax_pi)

    # Right: initial true grid
    field0_grid = _get_true_field_grid(0)
    if field0_grid is not None:
        # Build ROI layout from true map at t=0
        env.grid = base_no_roi.copy()
        roi_mask0 = (field0_grid > 1.0)  # your “> 1 means ROI” convention
        env.grid[roi_mask0] = ROI

    plot_grid(env,
              filename=None,
              title="True map",
              dpi=dpi,
              ax=ax_true,
              field=field0_grid,
              legend=True)

    # Time text
    txt = fig.text(
        0.5, 0.97,
        f"t = 0 / {T-1}",
        ha="center",
        va="top",
        fontsize=20,
        color="red",
        fontweight="bold"
    )

    def _update(frame_idx):
        # Update left & middle panels
        im0.set_data(emp_grids[frame_idx])
        im1.set_data(pi_grids[frame_idx])

        # Update true grid panel using true_phi_map[frame_idx]
        ax_true.clear()
        field_t_grid = _get_true_field_grid(frame_idx)

        if field_t_grid is not None:
            # Rebuild env.grid ROI layout from true map at this time
            env.grid = base_no_roi.copy()
            roi_mask_t = (field_t_grid > 1.0)
            env.grid[roi_mask_t] = ROI

        plot_grid(env,
                  filename=None,
                  title="True map",
                  dpi=dpi,
                  ax=ax_true,
                  field=field_t_grid,
                  legend=True)

        txt.set_text(f"t = {frame_idx} / {T-1}")
        # Right axis artists are handled internally by plot_grid,
        # so we only return the tracked ones explicitly:
        return (im0, im1, txt)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000/fps, blit=False
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    fig.tight_layout()
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)