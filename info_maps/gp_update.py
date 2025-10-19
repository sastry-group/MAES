import numpy as np

from env.grid import OBSTACLE, GridMC
from info_maps.gp_model import GPInfoMap


def update_phi_from_gp(env, gp: GPInfoMap, t: int, mode="variance", beta_u=1.0, beta_m=0.0, base=1e-3, cap=None):
    """
    mode:
      - "variance": phi = base + beta_u * Var
      - "ucb":      phi = base + beta_m * max(0, mu) + beta_u * std
    """
    coords = [env._index_state_map[i] for i in range(env.nstates)]
    mu, std = gp.predict_grid_at_time(coords, t)
    var = np.maximum(std**2, 1e-12)

    if mode == "variance":
        phi_vec = base + beta_u * var
    elif mode == "ucb":
        phi_vec = base + beta_m * np.maximum(mu, 0.0) + beta_u * std
    elif mode == "mean":
        phi_vec = base + beta_m * np.maximum(mu, 0.0)
    else:
        phi_vec = base + beta_u * var

    if cap is not None:
        phi_vec = np.minimum(phi_vec, cap)


    # --- Normalize for visibility (this is your snippet) ---
    m, M = phi_vec.min(), phi_vec.max()
    if M > m:
        phi_vec = (phi_vec - m) / (M - m) + 1e-3  # [1e-3, 1+1e-3]
    # -------------------------------------------------------

    # write back per valid cell; keep obstacles at 0
    for i, (r, c) in enumerate(coords):
        env.phi[r, c] = float(phi_vec[i])
    for r in range(env.H):
        for c in range(env.W):
            if env.grid[r, c] == OBSTACLE:
                env.phi[r, c] = 0.0

    # Rebuild P from the new 
    env.build_transition_matrix(phi_vec=phi_vec)

def push_logged_obs_to_gp(env, gp: GPInfoMap, use_noise=False):
    r, c, t, val = env.get_observations()
    if len(val) == 0:
        return False
    # Drop obstacles (val == None) if they exist; convert None → NaN then mask
    val_arr = np.array(val, dtype=float)
    mask = ~np.isnan(val_arr)
    r, c, t, val_arr = r[mask], c[mask], t[mask], val_arr[mask]
    noise = None
    if use_noise and hasattr(env, "obs_noise_var"):
        noise = np.asarray(env.obs_noise_var)[mask]
    gp.add_batch(r, c, t, val_arr, noise_var=noise)
    return True
