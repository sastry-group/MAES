# animation.py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from matplotlib.colors import Normalize, PowerNorm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Optional
from env.grid import GridMC, EMPTY, OBSTACLE, ROI, START
from core.types import GRID_CMAP
from matplotlib import animation
from viz.plotting import plot_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# main_level = os.path.dirname(__file__)
# drone_img = plt.imread(os.path.join(main_level, "figures", "camera-drone.png"))

# imagebox = OffsetImage(drone_img, zoom=0.05)



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
                          true_phi_map: Optional[np.ndarray] = None,
                          traj_rc: Optional[np.ndarray] = None):
    """
    Create a side-by-side animation of empirical vs target distributions over time.
    """


    emp_series = emp_series[::stride]
    pi_series  =  pi_series[::stride]
    T = len(emp_series)
    if traj_rc is not None:
        traj_rc = traj_rc[::stride][:T]
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

    pi_vmin = 0.0
    pi_vmax = max(g.max() for g in pi_grids)
    pi_vmax = max(pi_vmax, 1e-12)  # avoid zero range
    shared_norm = Normalize(vmin=pi_vmin, vmax=pi_vmax)

    # shared_norm = PowerNorm(
    #     gamma=0.6,     # < 1 enhances low values
    #     vmin=0.0,
    #     vmax=pi_vmax
    # )


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
    plt.colorbar(im0, ax=ax_emp, fraction=0.035, pad=0.03)
    drone_scatter = ax_emp.scatter(
        [], [],
        marker="o",
        s=140,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=6
    )

    if traj_rc is not None and len(traj_rc) > 0:
        r0, c0 = traj_rc[0]
        drone_scatter.set_offsets([[c0, r0]])
    # Middle: target heatmap
    im1 = ax_pi.imshow(pi_grids[0], cmap="viridis", norm=shared_norm)
    ax_pi.set_title("Target distribution")
    plt.colorbar(im1, ax=ax_pi, fraction=0.035, pad=0.03)

    # Right: initial true grid
    field0_grid = _get_true_field_grid(0)

    true_im = ax_true.imshow(
        field0_grid,
        cmap="YlOrBr",
        interpolation="nearest"
    )
    ax_true.set_title("True map")
    ax_true.set_xticks([])
    ax_true.set_yticks([])

    # if field0_grid is not None:
    #     # Build ROI layout from true map at t=0
    #     env.grid = base_no_roi.copy()
    #     roi_mask0 = (field0_grid > 1.0)  # your “> 1 means ROI” convention
    #     env.grid[roi_mask0] = ROI

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
        if traj_rc is not None:
            r, c = traj_rc[frame_idx]
            drone_scatter.set_offsets([[c, r]])

        # Update true grid panel using true_phi_map[frame_idx]
        # ax_true.clear()
        field_t_grid = _get_true_field_grid(frame_idx)

        if field_t_grid is not None:
            # Rebuild env.grid ROI layout from true map at this time
            true_im.set_data(field_t_grid)
            # env.grid = base_no_roi.copy()
            # roi_mask_t = (field_t_grid > 1.0)
            # env.grid[roi_mask_t] = ROI

        # plot_grid(env,
        #           filename=None,
        #           title="True map",
        #           dpi=dpi,
        #           ax=ax_true,
        #           field=field_t_grid,
        #           legend=True)

        txt.set_text(f"t = {frame_idx} / {T-1}")
        # Right axis artists are handled internally by plot_grid,
        # so we only return the tracked ones explicitly:
        return (im0, im1, txt, drone_scatter)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000/fps, blit=False
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    fig.tight_layout()
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)



def animate_distributions_multi(env,
                                emp_series,
                                pi_series,
                                filename,
                                fps=12,
                                stride=1,
                                dpi=150,
                                true_phi_map: Optional[np.ndarray] = None,
                                traj_rc_agents: Optional[np.ndarray] = None,
                                active_mask: Optional[np.ndarray] = None,
                                attrition_log: Optional[np.ndarray] = None):
    """
    Multi-agent version of animate_distributions.

    traj_rc_agents: array of shape (T, n_agents, 2) with (r,c) per agent per frame.
    """


    emp_series = emp_series[::stride]
    pi_series  =  pi_series[::stride]
    T = len(emp_series)
    assert T == len(pi_series) and T > 0, "emp_series and pi_series must be same nonzero length"



    # handle multi-agent trajectories
    if traj_rc_agents is not None:
        traj_rc_agents = np.asarray(traj_rc_agents)
        assert traj_rc_agents.ndim == 3 and traj_rc_agents.shape[2] == 2, \
            f"traj_rc_agents must be (T, n_agents, 2); got {traj_rc_agents.shape}"
        traj_rc_agents = traj_rc_agents[::stride]
        T = min(T, traj_rc_agents.shape[0])
        emp_series = emp_series[:T]
        pi_series  = pi_series[:T]
        traj_rc_agents = traj_rc_agents[:T]
        n_agents = traj_rc_agents.shape[1]
    else:
        n_agents = 0

    # handle true map time series
    if true_phi_map is not None:
        true_series = true_phi_map[::stride]
        T = min(T, len(true_series))
        emp_series = emp_series[:T]
        pi_series  = pi_series[:T]
        true_series = true_series[:T]
        if traj_rc_agents is not None:
            traj_rc_agents = traj_rc_agents[:T]
    else:
        true_series = None



    if active_mask is not None:
        active_mask = active_mask[::stride][:T]

    death_frames = {}
    if attrition_log is not None:
        for step, agent_id in attrition_log:
            death_frames[int(agent_id)] = int(step) // stride


    # Precompute grids for consistent color scale
    emp_grids = [_vec_to_grid(env, e) for e in emp_series]
    pi_grids  = [_vec_to_grid(env, p) for p in pi_series]

    pi_vmin = 0.0
    pi_vmax = max(g.max() for g in pi_grids)
    pi_vmax = max(pi_vmax, 1e-12)
    # shared_norm = Normalize(vmin=pi_vmin, vmax=pi_vmax)

    # base grid without ROIs for dynamic rebuild
    base_no_roi = env.grid.copy()
    base_no_roi[base_no_roi == ROI] = EMPTY

    def _get_true_field_grid(t_idx: int) -> Optional[np.ndarray]:
        if true_series is None:
            return None
        v = np.asarray(true_series[t_idx])
        if v.ndim == 1:
            v = _vec_to_grid(env, v)
        return v

    # --- Figure & axes ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(
        top=0.88,
        bottom=0.24,
        left=0.05,
        right=0.96,
        wspace=0.06
    )
    ax_emp, ax_pi, ax_true = axs

    # shared_norm = PowerNorm(
    #     gamma=0.6,    
    #     vmin=0.0,
    #     vmax=pi_vmax
    # )
    shared_norm = Normalize(vmin=0.0, vmax=pi_vmax)

    # Left: empirical heatmap
    im0 = ax_emp.imshow(emp_grids[0], cmap="viridis", norm=shared_norm)
    ax_emp.set_title("Empirical distribution")
    # plt.colorbar(im0, ax=ax_emp, fraction=0.035, pad=0.03)
    divider = make_axes_locatable(ax_pi)
    cax = divider.append_axes("right", size="4%", pad=0.05)


    # Multi-agent scatters (different colors)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    eps = 0.35
    jitter = np.array([
        [0, 0],
        [ eps, 0],
        [-eps, 0],
        [0,  eps],
        [0, -eps],
    ])
    markers = ["o", "s", "D", "^", "X", "P"]

    drone_scatters = []
    drone_centers = []

    if traj_rc_agents is not None:
        for a in range(n_agents):
            # Outer halo marker (legend entry)
            sc = ax_emp.scatter(
                [], [],
                marker=markers[a % len(markers)],
                s=120,
                facecolors="none",
                edgecolors="white",
                linewidths=1.0,
                zorder=7,
                label="_nolegend_"
            )
            drone_scatters.append(sc)


            ctr = ax_emp.scatter(
                [], [],
                s=30,
                c=colors[a % len(colors)],
                zorder=8,
                
                label=f"agent {a}"
            )
            drone_centers.append(ctr)

        # init offsets
        for a, (sc, ctr) in enumerate(zip(drone_scatters, drone_centers)):
            r0, c0 = traj_rc_agents[0, a]
            dx, dy = jitter[a % len(jitter)]
            sc.set_offsets([[c0 + dx, r0 + dy]])
            ctr.set_offsets([[c0 + dx, r0 + dy]])


    # Middle: target heatmap
    im1 = ax_pi.imshow(pi_grids[0], cmap="viridis", norm=shared_norm)
    ax_pi.set_title("Belief Target distribution")
    # plt.colorbar(im1, ax=ax_pi, fraction=0.035, pad=0.03)

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("Probability density", fontsize=10)


    agent_handles = [
        Line2D(
            [0], [0],
            marker=markers[a % len(markers)],
            linestyle="None",
            markerfacecolor=colors[a % len(colors)],
            # markeredgecolor="black",
            markersize=10,
            label=f"agent {a}"
        )
        for a in range(n_agents)
    ]

    ax_emp.legend(
        handles=agent_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # below the axis
        ncol=min(n_agents, 3),
        frameon=False,
        fontsize=9
    )

    for ax in (ax_emp, ax_pi):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    # Right: initial true grid
    field0_grid = _get_true_field_grid(0)
    if field0_grid is not None:
        env.grid = base_no_roi.copy()
        roi_mask0 = (field0_grid > 1.0)
        env.grid[roi_mask0] = ROI


    # if true_series is not None:
    #     # compute global ROI weight range ONCE
    #     roi_min = min(roi["weight"] for roi in env.roi_cells)
    #     roi_max = max(roi["weight"] for roi in env.roi_cells)


    #     base_cmap = cm.get_cmap("YlOrBr")
    #     cmap = mcolors.LinearSegmentedColormap.from_list(
    #         "trunc_YlOrBr",
    #         base_cmap(np.linspace(0.25, 0.9, 256))
    #     )
    #     norm = mcolors.Normalize(vmin=roi_min, vmax=roi_max)

    #     sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    #     sm.set_array([])

    #     true_cbar = fig.colorbar(
    #         sm,
    #         ax=ax_true,
    #         fraction=0.046,
    #         pad=0.04
    #     )
    #     true_cbar.set_label("Region of Interest Weight", fontsize=10)
    #     true_cbar.set_ticks([roi_min, roi_max])
    #     true_cbar.set_ticklabels(["Low", "High"])

    plot_grid(env,
              filename=None,
              title="True map",
              dpi=dpi,
              ax=ax_true,
              field=field0_grid,
              legend=True)

    # Time text
    txt = fig.text(
        0.5, 0.98,
        f"t = 0 / {T-1}",
        ha="center",
        va="top",
        fontsize=16,
        color="black",
        fontweight="semibold"

    )
    death_annotations = []

    def _update(frame_idx: int):
        im0.set_data(emp_grids[frame_idx])
        im1.set_data(pi_grids[frame_idx])

        # update agent markers
        if traj_rc_agents is not None:
            for a, (sc, ctr) in enumerate(zip(drone_scatters, drone_centers)):
                if active_mask is not None and not active_mask[frame_idx, a]:
                    # dead: hide markers
                    sc.set_offsets([[-999, -999]])
                    ctr.set_offsets([[-999, -999]])
                else:
                    r, c = traj_rc_agents[frame_idx, a]
                    dx, dy = jitter[a % len(jitter)]
                    sc.set_offsets([[c + dx, r + dy]])
                    ctr.set_offsets([[c + dx, r + dy]])
                # r, c = traj_rc_agents[frame_idx, a]
                # sc.set_offsets([[c, r]])
                # ctr.set_offsets([[c, r]])

        # show death marker on the frame the agent dies
        DEATH_DISPLAY_FRAMES = 5
        for agent_id, death_frame in death_frames.items():
            if frame_idx == death_frame:
                prev = max(0, frame_idx - 1)
                r, c = traj_rc_agents[prev, agent_id]
                ann = ax_emp.annotate(
                    "X",
                    xy=(c, r),
                    fontsize=14,
                    color="red",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=10,
                )
                death_annotations.append((ann, frame_idx + DEATH_DISPLAY_FRAMES))

                # strikethrough the legend label
                agent_handles[agent_id].set_label(f"agent {agent_id} ✗")
                agent_handles[agent_id].set_markerfacecolor("lightgray")
                ax_emp.legend(
                    handles=agent_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=min(n_agents, 3),
                    frameon=False,
                    fontsize=9,
                )

        # remove expired death markers
        still_active = []
        for ann, expire in death_annotations:
            if frame_idx >= expire:
                ann.remove()
            else:
                still_active.append((ann, expire))
        death_annotations.clear()
        death_annotations.extend(still_active)
        
        # update true grid
        ax_true.clear()
        field_t_grid = _get_true_field_grid(frame_idx)

        if field_t_grid is not None:
            env.grid = base_no_roi.copy()
            roi_mask_t = (field_t_grid > 1.0)
            env.grid[roi_mask_t] = ROI

        plot_grid(env,
                  filename=None,
                  title="True map",
                  dpi=dpi,
                  ax=ax_true,
                  field=field_t_grid,
                  legend=False)

        txt.set_text(f"t = {frame_idx} / {T-1}")
        return (im0, im1, txt, *drone_scatters, *drone_centers)

    anim = animation.FuncAnimation(fig, _update, frames=T, interval=1000/fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    # fig.tight_layout()
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)