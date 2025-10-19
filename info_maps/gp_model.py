# info_maps/gp_model.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class GPInfoMap:
    """
    Minimal spatio-temporal GP on inputs [r, c, t] -> y (observed ROI-weight-like value).
    l_space = spatial length scale
    l_time: how far in time you need to go before the correlation between measure points decays., 
    right now l_time is large since I am modeling static environments
    Matern: the covariance function
    ConstantKernel: sets the overall variance scale
    WhiteKernel: observation noise
    """
    def __init__(self, l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-3):
        # ARD over [r, c, t]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[l_space, l_space, l_time], nu=nu) 
                #  + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-6, 1e-1))
        self.obs_var = float(noise) 
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
        self.X = None   # (N,3): [r, c, t]
        self.y = None   # (N,)
        self.alpha = None  # per-sample noise variance

    def add_batch(self, r, c, t, y, noise_var=None):
        """Append a batch of observations; arrays must be same length."""
        r = np.asarray(r); c = np.asarray(c); t = np.asarray(t); y = np.asarray(y)
        X_new = np.stack([r.astype(float), c.astype(float), t.astype(float)], axis=1)
        if noise_var is None:
            alpha_new = np.full(len(y), 1e-3, dtype=float)
        else:
            alpha_new = np.asarray(noise_var, dtype=float)

        if self.X is None:
            self.X, self.y, self.alpha = X_new, y.astype(float), alpha_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.concatenate([self.y, y.astype(float)])
            self.alpha = np.concatenate([self.alpha, alpha_new])

    def fit_if_ready(self, min_pts=10):
        # wait to train until we have enough points
        if self.X is None or len(self.y) < min_pts:
            return False
        self.gp.set_params(alpha=self.alpha)
        self.gp.fit(self.X, self.y)
        return True

    def predict_grid_at_time(self, coords_rc, t):
        """coords_rc: list of (r,c) valid states; returns mean, std."""
        if self.X is None or len(self.y) < 2:
            n = len(coords_rc)
            return np.zeros(n), np.ones(n)  # encourage exploration early
        Xq = np.array([[float(r), float(c), float(t)] for (r,c) in coords_rc], dtype=float)
        mu, std = self.gp.predict(Xq, return_std=True)
        return mu, std