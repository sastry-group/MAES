import numpy as np

def empirical_from_traj(traj_states, env, k, window=10000):
    """
    traj_states: (T, n_agents)
    k: time index (inclusive)
    window: int or None (sliding window)
    """
    # if window is None:
    #     visits = traj_states[:k+1, :].reshape(-1)
    # else:
    start = max(0, k - window + 1)
    print(start, k+1)
    visits = traj_states[start:k+1, :].reshape(-1)

    counts = np.zeros(env.nstates, dtype=float)
    for s in visits:
        counts[int(s)] += 1.0

    return counts / max(counts.sum(), 1e-12)



# def first_visit_times(traj_t, cellset, t0=0, t1=None):
#     """
#     Returns time (relative to t0) when ANY cell in cellset is first visited.
#     traj_t: (T_env, n_agents, 2)
#     """
#     if t1 is None:
#         t1 = traj_t.shape[0]

#     for t in range(t0, min(t1, traj_t.shape[0])):
#         for a in range(traj_t.shape[1]):
#             r, c = traj_t[t, a]
#             if (int(r), int(c)) in cellset:
#                 return t - t0
#     return np.nan


def full_map_cover_time(
    traj_states,    # (T, n_agents)
    nstates,
):
    """
    First time when all states have been visited.
    Returns global timestep or np.nan.
    """
    traj_states = np.asarray(traj_states)
    T, n_agents = traj_states.shape

    visited = np.zeros(nstates, dtype=bool)

    for t in range(T):
        for a in range(n_agents):
            visited[int(traj_states[t, a])] = True
        if visited.all():
            return t

    return np.nan


def roi_first_visit_static(
    traj_rc,        # (T, n_agents, 2)
    roi_cellset,    # set[(r,c)]
):
    """
    Returns first timestep when any agent enters ROI.
    """
    T, n_agents, _ = traj_rc.shape

    for t in range(T):
        for a in range(n_agents):
            r, c = traj_rc[t, a]
            if (int(r), int(c)) in roi_cellset:
                return t
    return np.nan

def roi_first_visit_per_cfg(
    traj_rc,        # (T, n_agents, 2)
    roi_cellset,    # set[(r,c)]
    cfg_starts,     # list[int]
):
    """
    Returns list of offsets, one per config.
    """
    T = traj_rc.shape[0]
    times = []

    for i, t0 in enumerate(cfg_starts):
        t1 = cfg_starts[i+1] if i+1 < len(cfg_starts) else T
        found = np.nan

        for t in range(t0, t1):
            for a in range(traj_rc.shape[1]):
                r, c = traj_rc[t, a]
                if (int(r), int(c)) in roi_cellset:
                    found = t - t0
                    break
            if not np.isnan(found):
                break

        times.append(found)

    return times



def convergence_errors_from_run(
    traj_states,
    belief_phi_map,        # (T, nstates), UNnormalized
    true_phi_map_time,     # (T, nstates), UNnormalized or normalized
    env,
    empirical_fn,
    empirical_kwargs=None,
    norm="l1",
):
    """
    Returns:
        err_emp_true[t] = || rho_hat_t - rho*_t ||
        err_emp_bel[t]  = || rho_hat_t - rho_bar_t ||
        err_bel_true[t] = || rho_bar_t - rho*_t ||

    empirical_fn(traj_states, env, k, **kwargs) -> empirical distribution
    """
    if empirical_kwargs is None:
        empirical_kwargs = {}

    T = traj_states.shape[0]
    nstates = env.nstates

    # normalize belief and true maps
    belief_norm = belief_phi_map / belief_phi_map.sum(axis=1, keepdims=True)
    true_norm   = true_phi_map_time / true_phi_map_time.sum(axis=1, keepdims=True)

    err_emp_bel  = np.zeros(T)
    err_bel_true = np.zeros(T)
    err_emp_true = np.zeros(T)

    for t in range(T):
        rho_hat = empirical_fn(
            traj_states=traj_states,
            env=env,
            k=t,
            **empirical_kwargs
        )

        rho_bar = belief_norm[t]
        rho_star = true_norm[t]

        if norm == "l1":
            err_emp_bel[t]  = np.sum(np.abs(rho_hat - rho_bar))
            err_bel_true[t] = np.sum(np.abs(rho_bar - rho_star))
            err_emp_true[t] = np.sum(np.abs(rho_hat - rho_star))
        elif norm == "l2":
            err_emp_bel[t]  = np.linalg.norm(rho_hat - rho_bar)
            err_bel_true[t] = np.linalg.norm(rho_bar - rho_star)
            err_emp_true[t] = np.linalg.norm(rho_hat - rho_star)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    return err_emp_bel, err_bel_true, err_emp_true


def empirical_full(traj_states, env, k, **kwargs):
    visits = traj_states[:k+1, :].reshape(-1)

    counts = np.zeros(env.nstates)
    for s in visits:
        counts[int(s)] += 1

    return counts / max(counts.sum(), 1e-12)


def empirical_window(traj_states, env, k, window, **kwargs):
    start = max(0, k - window + 1)
    visits = traj_states[start:k+1, :].reshape(-1)

    counts = np.zeros(env.nstates)
    for s in visits:
        counts[int(s)] += 1

    return counts / max(counts.sum(), 1e-12)



def empirical_soft_forget(traj_states, env, k, gamma=0.98, **kwargs):
    T = traj_states.shape[0]
    k = min(k, T - 1)

    counts = np.zeros(env.nstates)
    for tau in range(k + 1):
        w = gamma ** (k - tau)
        for s in traj_states[tau]:
            counts[int(s)] += w

    return counts / max(counts.sum(), 1e-12)