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
    def __init__(self, gp, mode=None, beta_m=1.0, beta_u=1.0, base=1e-3,
                 beta_schedule="const", nstates=None, delta=0.05):
        
        self.gp = gp
        self.mode = mode
        self.beta_m = beta_m
        self.beta_u = beta_u          # used when beta_schedule == "const"
        self.base = base
        self.beta_schedule = beta_schedule
        self.nstates = nstates        # required for "mac_dt"
        self.delta = delta            # required for "mac_dt"
        self.last_mu = None           # posterior mean from the most recent phi_vec() call
        self.last_std = None          # posterior std from the most recent phi_vec() call

    def _beta(self, e):
        if self.beta_schedule == "mac_dt":
            # MAC-DT / Srinivas: beta^(e) = sqrt(2 log(|V| pi^2 e^2 / (6 delta)))
            if e is None:
                raise ValueError("GPPhi(beta_schedule='mac_dt') requires episode index e in phi_vec(...)")
            V = self.nstates
            return np.sqrt(2.0 * np.log(V * np.pi**2 * (e**2) / (6.0 * self.delta)))
        return self.beta_u 

    def phi_vec(self, env, t, e=None):
        coords = [env._index_state_map[i] for i in range(env.nstates)]
        mu, std = self.gp.predict_grid_at_time(coords, t)
        self.last_mu = mu      
        self.last_std = std    
        beta_u = self._beta(e)      # const -> beta_u; mac_dt -> episode beta
        # print(f"Beta {beta_u}, for episode {e}")

        if self.mode == "ucb":
            phi = self.base + self.beta_m*np.maximum(mu, 0.0) + beta_u*std
        elif self.mode == "mean":
            phi = self.base + self.beta_m*np.maximum(mu, 0.0)
        else:  # "variance"
            phi = self.base + beta_u*(std**2)
        # m, M = phi.min(), phi.max()
        # if M > m: 
        #     phi = (phi - m)/(M - m) + 1e-3
        return phi