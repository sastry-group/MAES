import numpy as np

def empirical_from_traj(traj_states, env, k, window=None):
    """
    traj_states: (T, n_agents)
    k: time index (inclusive)
    window: int or None (sliding window)
    """
    if window is None:
        visits = traj_states[:k+1, :].reshape(-1)
    else:
        start = max(0, k - window + 1)
        visits = traj_states[start:k+1, :].reshape(-1)

    counts = np.zeros(env.nstates, dtype=float)
    for s in visits:
        counts[int(s)] += 1.0

    return counts / max(counts.sum(), 1e-12)



def first_visit_times(traj_states, nstates):
    """
    Returns array first_visit[s] = first timestep state s was visited, or -1.
    """
    T, n_agents = traj_states.shape
    first = -np.ones(nstates, dtype=int)

    for t in range(T):
        for a in range(n_agents):
            s = int(traj_states[t, a])
            if first[s] == -1:
                first[s] = t
    return first