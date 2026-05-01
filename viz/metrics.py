#metrics.py
import numpy as np

def empirical_from_traj(traj_states, env, k, window=10000):
    start = max(0, k - window + 1)
    visits = traj_states[start:k+1, :].reshape(-1)
    visits = visits[visits >= 0]  # skip dead agents
    if len(visits) == 0:
        return np.ones(env.nstates) / env.nstates
    counts = np.zeros(env.nstates, dtype=float)
    for s in visits:
        counts[int(s)] += 1.0
    return counts / max(counts.sum(), 1e-12)





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
            s = int(traj_states[t, a])
            if s >= 0:  # skip dead agents
                visited[s] = True
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

def roi_first_visit_per_cfg(traj_rc, roi_cellset, cfg_starts):
    T = traj_rc.shape[0]
    times = []
    for i, t0 in enumerate(cfg_starts):
        t1 = cfg_starts[i+1] if i+1 < len(cfg_starts) else T
        found = np.nan
        for t in range(t0, t1):
            for a in range(traj_rc.shape[1]):
                r, c = traj_rc[t, a]
                if int(r) < 0:  # skip dead agents
                    continue
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
    visits = visits[visits >= 0]
    if len(visits) == 0:
        return np.ones(env.nstates) / env.nstates
    counts = np.zeros(env.nstates)
    for s in visits:
        counts[int(s)] += 1
    return counts / max(counts.sum(), 1e-12)


def empirical_window(traj_states, env, k, window, **kwargs):
    start = max(0, k - window + 1)
    visits = traj_states[start:k+1, :].reshape(-1)
    visits = visits[visits >= 0]
    if len(visits) == 0:
        return np.ones(env.nstates) / env.nstates
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
            if int(s) >= 0:  # skip dead agents
                counts[int(s)] += w
    if counts.sum() == 0:
        return np.ones(env.nstates) / env.nstates
    return counts / max(counts.sum(), 1e-12)




# ── KL Divergence ──────────────────────────────────────────────────────────────

def kl_divergence(p, q, eps=1e-12):
    """KL(p || q).  Both inputs are normalized to sum to 1 internally."""
    p = np.clip(p, eps, None);  p = p / p.sum()
    q = np.clip(q, eps, None);  q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_series_emp_vs_true(
    traj_states,          # (T, n_agents)
    true_phi_map_time,    # (T, nstates)  – unnormalized OK
    env,
    empirical_fn,
    empirical_kwargs=None,
    stride=1,
):
    """
    KL(rho_hat_t || rho*_t)  for every t (with optional stride).

    Returns
    -------
    steps : np.ndarray   shape (n_steps,)
    kl    : np.ndarray   shape (n_steps,)
    """
    if empirical_kwargs is None:
        empirical_kwargs = {}

    T = traj_states.shape[0]
    true_norm = true_phi_map_time / true_phi_map_time.sum(axis=1, keepdims=True)

    steps, kl = [], []
    for t in range(0, T, stride):
        rho_hat  = empirical_fn(traj_states=traj_states, env=env, k=t,
                                **empirical_kwargs)
        rho_star = true_norm[t]
        steps.append(t)
        kl.append(kl_divergence(rho_hat, rho_star))

    return np.array(steps), np.array(kl)


def kl_series_belief_vs_true(
    belief_phi_map,       # (T, nstates) or (T, N, nstates)
    true_phi_map_time,    # (T, nstates)
    stride=1,
):
    """KL(rho_bar_t || rho*_t) – belief vs ground truth."""
    if belief_phi_map.ndim == 3:
        belief_phi_map = belief_phi_map.mean(axis=1)

    true_norm   = true_phi_map_time / true_phi_map_time.sum(axis=1, keepdims=True)
    belief_norm = belief_phi_map    / belief_phi_map.sum(axis=1, keepdims=True)

    T = belief_norm.shape[0]
    steps = np.arange(0, T, stride)
    kl = np.array([kl_divergence(belief_norm[t], true_norm[t]) for t in steps])
    return steps, kl


def kl_inter_agent_series(
    phi_agents_time,   # (T, N, nstates)  – per-agent beliefs
    gp_update_steps=None,   # optional array of timesteps to evaluate at
    stride=1,
):
    """
    Mean KL between each agent's belief and the team-mean belief.

    D_KL(rho^m_t || rho_bar_t),  averaged over agents m.

    Returns
    -------
    steps    : np.ndarray  (n_steps,)
    kl_mean  : np.ndarray  (n_steps,)
    kl_std   : np.ndarray  (n_steps,)
    """
    T, N, _ = phi_agents_time.shape
    phis = phi_agents_time / phi_agents_time.sum(axis=2, keepdims=True)

    eval_steps = gp_update_steps if gp_update_steps is not None \
                 else np.arange(0, T, stride)

    kl_means, kl_stds = [], []
    for t in eval_steps:
        phi_t    = phis[t]                        # (N, nstates)
        phi_mean = phi_t.mean(axis=0)             # (nstates,)
        vals     = [kl_divergence(phi_t[a], phi_mean) for a in range(N)]
        kl_means.append(np.mean(vals))
        kl_stds.append(np.std(vals))

    return np.array(eval_steps), np.array(kl_means), np.array(kl_stds)