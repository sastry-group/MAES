# info_maps/info_map_sources.py
import numpy as np

class KnownPhi:
    def phi_vec(self, env, t=None):
        field = env.true_phi if env.true_phi is not None else env.phi_belief
        print(f"[KnownPhi] Using {'true_phi' if env.true_phi is not None else 'phi_belief'} for phi_vec.")
        vec = env.build_map_vec(field)
        return vec

class UniformPhi:
    def phi_vec(self, env, t=None):
        mask = (env.grid != 1).astype(float)
        return env.build_map_vec(mask)

class GPPhi:
    def __init__(self, gp, mode=None, beta_m=1.0, beta_u=1.0, base=1e-3):
        self.gp, self.mode, self.beta_m, self.beta_u, self.base = gp, mode, beta_m, beta_u, base

    def phi_vec(self, env, t):
        coords = [env._index_state_map[i] for i in range(env.nstates)]
        mu, std = self.gp.predict_grid_at_time(coords, t)
        if self.mode == "ucb":
            phi = self.base + self.beta_m*np.maximum(mu, 0.0) + self.beta_u*std
        elif self.mode == "mean":
            phi = self.base + self.beta_m*np.maximum(mu, 0.0)
        else:  # "variance"
            phi = self.base + self.beta_u*(std**2)
        # m, M = phi.min(), phi.max()
        # if M > m: 
        #     phi = (phi - m)/(M - m) + 1e-3
        return phi