# sim/sensing.py
"""Sensing model shared by all planners (ergodic, mac_dt).

Eq. (3):  y = phi*_k(r) + eps,   eps ~ N(0, sigma_eps^2),
iid across agents, regions and time. sigma_eps is one constant for the run.

Observations are NOT clipped: y can be negative near phi* = 0. Clipping would
bias the GP mean upward there. Nonnegativity is enforced on the belief instead
(GPPhi: base + beta_m*[mu]_+ + beta_u*std > 0), which is what gets normalized.
"""
import numpy as np
from env.grid import OBSTACLE


class GaussianSensor:
    def __init__(self, sigma_eps, seed=0, gp_noise_scale=1.0, var_floor=1e-6):
        self.sigma_eps = float(sigma_eps)
        self.rng = np.random.default_rng(seed)
        self.gp_noise_scale = float(gp_noise_scale)  # 1.0 = GP assumes the true sigma_eps
        self.var_floor = float(var_floor)            # numerical jitter only

    @classmethod
    def from_args(cls, args, env):
        """sigma_eps = --noise_sigma if given, else --noise_rel * max phi*_0.
        Fixed from the initial map, so it does not change when the map changes."""
        if args.noise_sigma is not None:
            sigma = args.noise_sigma
        else:
            phi_max = float(np.max(env.true_phi[env.grid != OBSTACLE]))
            sigma = args.noise_rel * phi_max
        return cls(sigma, seed=args.noise_seed, gp_noise_scale=args.gp_noise_scale)

    def observe(self, env, rows, cols):
        """Noisy samples of env.true_phi at (rows, cols) and the per-sample
        noise variance to hand to the GP."""
        rows = np.asarray(rows, dtype=int); cols = np.asarray(cols, dtype=int)
        y = env.true_phi[rows, cols].astype(float) + self.sigma_eps * self.rng.standard_normal(rows.shape)
        noise_var = np.full(rows.shape, max((self.gp_noise_scale * self.sigma_eps) ** 2, self.var_floor))
        return y, noise_var