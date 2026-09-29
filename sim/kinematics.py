# sim/kinematics.py
"""Simple multi-UAV dynamics + collision avoidance on top of the grid planner.
Pure numpy/scipy, no external simulator.
RUN=results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_init1_noiseRel0.05

# 1) fly the planner's waypoints -> kinematics.npz
python3 -m sim.kinematics --h5 $RUN/run.h5 --out $RUN/kinematics.npz --altitudes 120 135 150

python -m sim.kinematics --h5 results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05/run.h5 --out results/gp_studies_A/T2000_ergodic_mh_Pupdate25_GPupdate25_5x5_static_v0_gpucb_static_3_start44x3_commModelocal_comm_radius2.0_concenIter5_ltime10000_shareHis100_GPobs2000_ucb_betaconst_noiseRel0.05/kinematics.npz --altitudes 55 65 75

Planner stays kinematics-free. This is a post-processor: it reads the cell
sequence the planner produced (h5 'traj_rc_agents_all', shape (T, N, 2)) and
flies every UAV through it in 3D.

Per planner step k (length dt_s), agent m is handed its next region r_{k+1}.
Move slot: fly to the region center (+ small jitter). Stay slot (MH
self-loop): loiter on a circular orbit around the center, all agents
counter-clockwise, so agents sharing a region fall into a holding pattern.
An agent only needs to be INSIDE the region long enough to sample it (dwell_s).

Dynamics: 3D double integrator per agent, p_ddot = u,
    |u| <= a_max, horizontal speed <= v_max, vertical speed <= vz_max.
Control:
    u = sat((v_des - v)/tau_v, a_max),  a_dot = (u - a)/tau_a   (finite jerk)
    v_des = goal velocity (horizontal, |.| <= min(v_max, sqrt(2 a_max d)),
            altitude hold to cruise) + avoidance velocity for neighbors
            closer than d_act: horizontal push-apart, plus a vertical split
            (one climbs, one descends, within +-dz_max of cruise).
            Avoidance has priority: the goal term shrinks as it grows.
No formal separation guarantee: min separation is measured and reported.

Logged per (k, m): arrival time into r_{k+1}, time spent inside r_{k+1},
slack = time_inside - dwell_s (< 0: step infeasible). Per k: min separation.
"""
from dataclasses import dataclass, asdict
import argparse
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class UnitsConfig:
    cell_m: float = 200.0      # region side ~ wide-camera footprint at altitude_m (~202x152 m)
    dt_s: float = 30.0         # planner time step (diag crossing 26 s incl. 3 s dwell)
    altitude_m: float = 120.0  # cruise altitude (all agents)
    altitudes: tuple = ()      # optional cruise bands, agent m flies altitudes[m % len]; overrides altitude_m
    v_max: float = 16.0        # horizontal, m/s (Skydio X10 cruise)
    vz_max: float = 4.0        # vertical, m/s
    a_max: float = 3.0         # m/s^2 (assumed)
    tau_v: float = 0.5         # s, velocity-tracking time constant
    tau_a: float = 0.0         # s, acceleration lag; 0 = pure double integrator
    kz: float = 0.5            # 1/s, altitude-hold gain
    dwell_s: float = 3.0       # required time inside the region to sample it
    goal_margin_m: float = 30.0  # start positions kept this far from region edges
    goal_jitter_m: float = 20.0  # move slot: goal = region center + small jitter
    loiter_r_m: float = 60.0     # stay slot: orbit this circle around the center
    loiter_v: float = 8.0        # m/s along the orbit (counter-clockwise for all); v^2/r = 1.07 m/s^2
    k_orbit: float = 0.3         # 1/s, radial correction onto the orbit
    d_safe: float = 15.0       # separation target: sNMAC, 50 ft horizontal (RTCA 2022)
    d_act: float = None        # [m] None -> d_full + (2 v_max)^2/(4 a_max), rounded up to 10 m (head-on braking)
    k_rep: float = None        # [m^2/s] None -> repulsion reaches v_max at d_full
    dz_max: float = 5.0        # max vertical deviation from cruise used to deconflict
    veer_deg: float = 45.0     # right-hand rule: push-apart rotated this much (breaks head-on deadlock)
    ctrl_hz: float = 10.0       ## for the control step 1 / ctrl_hz
    log_hz: float = 1.0
    seed: int = 0

    def __post_init__(self):
        d_full = self.d_safe + 2.0 * self.v_max * self.tau_v            # closing during one velocity-loop response
        if self.d_act is None:                                           # head-on pair must brake to rest before d_full
            self.d_act = 10.0 * np.ceil((d_full + (2.0 * self.v_max) ** 2 / (4.0 * self.a_max)) / 10.0)
        if self.k_rep is None:
            self.k_rep = self.v_max / (1.0 / d_full - 1.0 / self.d_act)


def dt_required(cfg, diagonal=False):
    """Rest-to-rest time to cross one region edge + dwell (no avoidance)."""
    L = cfg.cell_m * (np.sqrt(2.0) if diagonal else 1.0)
    v, a = cfg.v_max, cfg.a_max
    t = L / v + v / a if L >= v * v / a else 2.0 * np.sqrt(L / a)
    return t + cfg.dwell_s


def cells_to_xy(traj_rc, cfg):
    """(T, N, 2) row/col -> (T, N, 2) metric x,y of region centers (x=col, y=row)."""
    rc = np.asarray(traj_rc, dtype=float)
    return np.stack([(rc[..., 1] + 0.5) * cfg.cell_m, (rc[..., 0] + 0.5) * cfg.cell_m], axis=-1)


def _sat(x, lim):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x * np.minimum(1.0, lim / np.maximum(n, 1e-12))


def _avoid_velocity(p, cfg, tree):
    """Velocity-level avoidance for pairs closer than d_act (3D distance).
    Horizontal: push apart, k_rep*(1/d - 1/d_act)/d along (p_i - p_j),
                rotated by veer_deg so both agents veer right (no head-on deadlock).
    Vertical:   split the pair, higher index up, lower index down (bounded
                by dz_max via the altitude-hold term). Breaks head-on symmetry.
    """
    vr = np.zeros_like(p)
    pairs = tree.query_pairs(cfg.d_act, output_type="ndarray")
    if len(pairs) == 0:
        return vr
    i, j = pairs[:, 0], pairs[:, 1]
    diff = p[i] - p[j]
    d = np.maximum(np.linalg.norm(diff, axis=1), 0.1)
    w = (1.0 / d - 1.0 / cfg.d_act) / d                 # >= 0, grows as d -> 0
    c, sn = np.cos(np.radians(cfg.veer_deg)), np.sin(np.radians(cfg.veer_deg))
    dx, dy = diff[:, 0], diff[:, 1]
    fh = cfg.k_rep * w[:, None] * np.column_stack([c * dx + sn * dy, -sn * dx + c * dy])
    np.add.at(vr[:, :2], i, fh)      # rotating diff and -diff the same way:
    np.add.at(vr[:, :2], j, -fh)     # both agents veer to their own right
    fz = cfg.vz_max * np.clip(w * cfg.d_safe**2, 0.0, 1.0)  # up to vz_max when d <= ~d_safe
    np.add.at(vr[:, 2], np.maximum(i, j), fz)
    np.add.at(vr[:, 2], np.minimum(i, j), -fz)
    return vr


def _separated_starts(c0, spread, d_min, rng, tries=200):
    """Uniform in each start region, rejecting points closer than d_min to earlier agents."""
    p = np.empty_like(c0)
    for m in range(c0.shape[0]):
        for _ in range(tries):
            q = c0[m] + rng.uniform(-spread, spread, 2)
            if m == 0 or np.min(np.linalg.norm(p[:m] - q, axis=1)) >= d_min:
                break
        p[m] = q
    return p


def track(centers, cfg):
    """centers: (K+1, N, 2) region centers in meters. Returns logs (positions are 3D)."""
    rng = np.random.default_rng(cfg.seed)
    K, N = centers.shape[0] - 1, centers.shape[1]
    h = 1.0 / cfg.ctrl_hz
    n_sub = int(round(cfg.dt_s * cfg.ctrl_hz))
    log_every = max(1, int(round(cfg.ctrl_hz / cfg.log_hz)))
    half = cfg.cell_m / 2.0
    spread = half - cfg.goal_margin_m

    xy0 = _separated_starts(centers[0], spread, cfg.d_safe + 1.0, rng)
    z_c = (np.array(cfg.altitudes, float)[np.arange(N) % len(cfg.altitudes)]
           if len(cfg.altitudes) else np.full(N, cfg.altitude_m))   # per-agent cruise altitude
    p = np.column_stack([xy0, z_c])
    v = np.zeros((N, 3))
    acc = np.zeros((N, 3))

    n_log = K * n_sub // log_every + 1
    P = np.empty((n_log, N, 3), dtype=np.float32); P[0] = p
    V = np.empty_like(P); V[0] = v
    arrival = np.full((K, N), np.nan)
    t_inside = np.zeros((K, N))
    min_sep = np.full(K, np.inf)
    li = 1

    for k in range(K):
        c1 = centers[k + 1]
        stay = np.all(centers[k + 1] == centers[k], axis=1)          # MH self-loop
        goal = c1 + rng.uniform(-cfg.goal_jitter_m, cfg.goal_jitter_m, size=(N, 2))
        # goal = c1
        
        for j in range(n_sub):
            d = goal - p[:, :2]
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            # dist = np.maximum(np.linalg.norm(goal - p[:, :2], axis=1, keepdims=True) - cfg.goal_jitter_m, 0.0)
            speed = np.minimum(cfg.v_max, np.sqrt(2 * cfg.a_max * dist))
            v_des = np.zeros((N, 3))
            v_des[:, :2] = d / np.maximum(dist, 1e-6) * speed
            if stay.any():                                          # loiter: orbit the center
                rel = p[stay, :2] - c1[stay]
                r = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-6)
                radial, tang = rel / r, np.column_stack([-rel[:, 1], rel[:, 0]]) / r
                v_des[stay, :2] = _sat(cfg.loiter_v * tang + cfg.k_orbit * (cfg.loiter_r_m - r) * radial,
                                       cfg.v_max)
            v_des[:, 2] = cfg.kz * (z_c - p[:, 2])

            # avoidance has priority: goal velocity shrinks as avoidance grows
            v_av = _avoid_velocity(p, cfg, cKDTree(p))
            share = 1.0 - np.minimum(1.0, np.linalg.norm(v_av[:, :2], axis=1, keepdims=True) / cfg.v_max)
            v_des[:, :2] = _sat(v_av[:, :2] + share * v_des[:, :2], cfg.v_max)
            v_des[:, 2] = v_des[:, 2] + v_av[:, 2]
            z_lo, z_hi = z_c - cfg.dz_max, z_c + cfg.dz_max
            v_des[:, 2] = np.where(p[:, 2] >= z_hi, np.minimum(v_des[:, 2], 0.0), v_des[:, 2])
            v_des[:, 2] = np.where(p[:, 2] <= z_lo, np.maximum(v_des[:, 2], 0.0), v_des[:, 2])
            v_des[:, 2] = np.clip(v_des[:, 2], -cfg.vz_max, cfg.vz_max)

            u = _sat((v_des - v) / cfg.tau_v, cfg.a_max)
            # acc = acc + (h / cfg.tau_a) * (u - acc)          # jerk-limited: accel is continuous
            acc = u if cfg.tau_a <= 0 else acc + (h / cfg.tau_a) * (u - acc)
            v = v + h * acc
            v[:, :2] = _sat(v[:, :2], cfg.v_max)
            v[:, 2] = np.clip(v[:, 2], -cfg.vz_max, cfg.vz_max)
            p = p + h * v

            ins = np.all(np.abs(p[:, :2] - c1) <= half, axis=1)
            t_inside[k] += h * ins
            first = ins & np.isnan(arrival[k])
            arrival[k, first] = (j + 1) * h
            if N > 1:
                dd, _ = cKDTree(p).query(p, k=2)
                min_sep[k] = min(min_sep[k], dd[:, 1].min())
            if (k * n_sub + j + 1) % log_every == 0:
                P[li], V[li] = p, v
                li += 1

    return dict(t=np.arange(li) * log_every * h, p=P[:li], v=V[:li],
                centers=centers, arrival=arrival, t_inside=t_inside,
                slack=t_inside - cfg.dwell_s, min_sep=min_sep)


def summarize(out, cfg):
    s = out["slack"]
    vh = np.linalg.norm(out["v"][..., :2], axis=-1)
    z = out["p"][..., 2]
    late = np.argwhere(s < 0)
    if len(late):
        print("late (step k, agent m):", late.tolist())
    print(f"steps x agents: {s.size}  never arrived: {np.isnan(out['arrival']).sum()}  "
          f"infeasible (slack<0): {(s < 0).sum()}  slack min/median: {s.min():.2f}/{np.median(s):.2f} s")
    print(f"max horiz speed: {vh.max():.2f} m/s   altitude range: {z.min():.1f}-{z.max():.1f} m   "
          f"min separation: {out['min_sep'].min():.2f} m (target >= {cfg.d_safe:.1f})")
    print(f"dt required (rest-to-rest bound): orth {dt_required(cfg):.1f} s, diag {dt_required(cfg, True):.1f} s")


def run_from_h5(h5_path, out_path, args=None, steps=None, agents=None):
    """Fly a finished run. Called from run_gp.py when --kinematics != none."""
    import h5py
    cfg = UnitsConfig()
    for k in ("dt_s", "cell_m", "altitude_m"):
        if args is not None and getattr(args, k, None) is not None:
            setattr(cfg, k, getattr(args, k))
    if args is not None and getattr(args, "altitudes", None):
        cfg.altitudes = tuple(args.altitudes)
    steps = steps or (getattr(args, "kin_steps", None) if args is not None else None)
    with h5py.File(h5_path, "r") as f:
        rc = f["traj_rc_agents_all"][slice(*steps) if steps else slice(None)]
    if agents is not None:
        rc = rc[:, agents]
    out = track(cells_to_xy(rc, cfg), cfg)
    summarize(out, cfg)
    np.savez_compressed(out_path, **out, **{f"cfg_{k}": np.asarray(v) for k, v in asdict(cfg).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--agents", type=int, nargs="+", default=None, help="subset; default all")
    ap.add_argument("--steps", type=int, nargs=2, default=None, help="step window; default all")
    ap.add_argument("--dt_s", type=float, default=UnitsConfig.dt_s)
    ap.add_argument("--cell_m", type=float, default=UnitsConfig.cell_m)
    ap.add_argument("--altitude_m", type=float, default=UnitsConfig.altitude_m)
    ap.add_argument("--altitudes", type=float, nargs="+", default=None,
                    help="cruise bands, e.g. 55 65 75; agent m flies altitudes[m %% len]")
    ap.add_argument("--out", default="kinematics_out.npz")
    a = ap.parse_args()
    run_from_h5(a.h5, a.out, a, steps=a.steps, agents=a.agents)


if __name__ == "__main__":
    main()