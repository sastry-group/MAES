# info_maps/gp_model.py
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from info_maps.sgpr import SGPR, inducing_grid, inducing_pcd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

KERNELS = ("matern12", "matern32", "matern52", "rbf", "prod_matern_exp")
_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}
_OFF = 1e8                   # length scale that switches an input dimension off
_LS_BOUNDS = (1e-2, 1e5)     # only used when hyperparameters are learned
_AMP_BOUNDS = (1e-3, 1e3)



def build_kernel(name, l_space, l_time, amp, learn=False):
    """
    Non-separable:  amp * k([r, c, t])          (matern12/32/52, rbf; legacy = matern32)
    Separable:      amp * k_space(r, c) * k_time(t)   (prod_matern_exp)
                    k_space = Matern 3/2, k_time = exp(-|t-t'|/l_time)  (Matern 1/2).
    k_time is the forgetting kernel of Bogunovic et al. (2016), (1-eps)^{|t-t'|/2}
    with eps = 1 - exp(-2/l_time). A dimension is switched off by giving it
    length scale _OFF, so both factors take the full [r, c, t] input.
    """
    ls_b = _LS_BOUNDS if learn else "fixed"
    amp_b = _AMP_BOUNDS if learn else "fixed"
    ls = [l_space, l_space, l_time]
    if name in _NU:
        base = Matern(length_scale=ls, length_scale_bounds=ls_b, nu=_NU[name])
    elif name == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=ls_b)
    elif name == "prod_matern_exp":
        b_s = [_LS_BOUNDS, _LS_BOUNDS, (_OFF, _OFF)] if learn else "fixed"
        b_t = [(_OFF, _OFF), (_OFF, _OFF), _LS_BOUNDS] if learn else "fixed"
        base = (Matern(length_scale=[l_space, l_space, _OFF], length_scale_bounds=b_s, nu=1.5)
                * Matern(length_scale=[_OFF, _OFF, l_time], length_scale_bounds=b_t, nu=0.5))
    else:
        raise ValueError(f"unknown kernel {name!r}; choose from {KERNELS}")
    return ConstantKernel(amp, amp_b) * base


class GPInfoMap:
    """
    Minimal spatio-temporal GP on inputs [r, c, t] -> y (observed ROI-weight-like value).
    l_space = spatial length scale
    l_time: how far in time you need to go before the correlation between measure points decays.
    kernel: see build_kernel. amp: prior variance (ConstantKernel).
    prior:  "data"  -> normalize_y=True; amp is in standardized-y units (legacy)
            "fixed" -> normalize_y=False, prior mean prior_mean, amp in phi^2 units
    learn:  "none"  -> hyperparameters fixed, one O(B^3) solve per refit (paper setting)
            "once"  -> type-II ML at the first refit with >= learn_min_pts points, then frozen
            "every" -> type-II ML at every refit (cost reference only)
    Observation noise comes in per sample (alpha) from the sensor, Eq. (3).
    """
    def __init__(self, l_space=2.0, l_time=10000.0, nu=1.5, noise=1e-3,
                 kernel="matern32", amp=10.0, prior="data", prior_mean=0.0,
                 learn="none", learn_min_pts=200, method="exact", gp_M=200, 
                bin_dt=0, fitc=False):
        if kernel == "matern32" and nu != 1.5:          
            kernel = {0.5: "matern12", 2.5: "matern52"}[nu]
        self.kernel_name = kernel
        self.l_space, self.l_time, self.amp = float(l_space), float(l_time), float(amp)
        self.prior, self.prior_mean = prior, float(prior_mean)
        self.learn, self.learn_min_pts = learn, int(learn_min_pts)
        self.obs_var = float(noise)
        self.gp = GaussianProcessRegressor(
            kernel=build_kernel(kernel, l_space, l_time, amp, learn=(learn != "none")),
            normalize_y=(prior == "data"),
            optimizer=None, n_restarts_optimizer=0)
        self.X = None   # (N,3): [row, column, time]
        self.y = None   # (N,)
        self.alpha = None  # per-sample noise variance
        # wall-clock of the last refit / last prediction (T4(a))
        self.last_n, self.last_fit_s, self.last_predict_s = 0, 0.0, 0.0
        self.sgpr = None
        self.method = method  # whether to use regular GP r SGPR (induced points)
        self.gp_M = gp_M # number of induceing points
        self.bin_dt = bin_dt         
        self.fitc = fitc # Fully Independent Training Conditional (Snelson & Ghahramani, 2006)  
                        # false is variatoinal, true is not          
 
    @classmethod
    def from_args(cls, args):
        return cls(l_space=args.l_space, l_time=args.l_time,
                    kernel=getattr(args, "kernel", "matern32"),
                    amp=getattr(args, "kernel_amp", 10.0),
                    prior=getattr(args, "gp_prior", "data"),
                    prior_mean=getattr(args, "gp_prior_mean", 0.0),
                    learn=getattr(args, "learn_hypers", "none"),
                    method=getattr(args, "gp_method", "exact"),
                    gp_M=getattr(args, "gp_M", 200), 
                    bin_dt=getattr(args, "gp_bin_dt", 0),
                    fitc=bool(getattr(args, "gp_fitc", 0)),                   
                   )

    def attrs(self):
        """Kernel description for the h5 attrs."""
        nu_s = {"rbf": np.inf, "prod_matern_exp": 1.5}.get(self.kernel_name, _NU.get(self.kernel_name))
        nu_t = 0.5 if self.kernel_name == "prod_matern_exp" else nu_s
        return dict(gp_kernel=self.kernel_name, gp_kernel_amp=self.amp,
                    gp_nu_space=float(nu_s), gp_nu_time=float(nu_t),
                    l_space=self.l_space, l_time=self.l_time,
                    gp_prior=self.prior, gp_prior_mean=self.prior_mean,
                    learn_hypers=self.learn,
                    p_method=self.method, 
                    gp_M=self.gp_M, gp_bin_dt=self.bin_dt, gp_fitc=int(self.fitc)
                    )


    def trim(self, max_pts=500):
        """Keep only the most recent max_pts observations."""
        if self.X is None or len(self.y) <= max_pts:
            return
        self.X = self.X[-max_pts:]
        self.y = self.y[-max_pts:]
        self.alpha = self.alpha[-max_pts:]

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
            self.X = np.vstack([self.X, X_new]) # stacking region, counts and time (stacking new inputs)
            self.y = np.concatenate([self.y, y.astype(float)])
            self.alpha = np.concatenate([self.alpha, alpha_new]) # storing per observation variance


    def _maybe_learn(self):
        """Switch the optimizer on for this fit if the learn mode asks for it."""
        ready = len(self.y) >= self.learn_min_pts
        if self.learn == "every":
            self.gp.set_params(optimizer="fmin_l_bfgs_b" if ready else None)
        elif self.learn == "once" and self.gp.optimizer is None and ready:
            self.gp.set_params(optimizer="fmin_l_bfgs_b")


    def _compress(self):
        """Method 2: merge repeated measurements of the same region in the same time bin
        (inverse-variance mean, noise 1/sum(1/alpha)). bin_dt=0 -> raw data."""
        X, y, a = self.X, self.y, self.alpha
        if not self.bin_dt:
            return X, y, a
        key = np.stack([X[:, 0], X[:, 1], np.floor(X[:, 2] / self.bin_dt)], axis=1)
        _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        w = 1.0 / a # 1/ sigma^2, a here is the variance sigma^2
        sw = np.bincount(inv, weights=w) # sums of the inverse of variances
        Xb = X[idx].copy()
        Xb[:, 2] = np.bincount(inv, weights=w * X[:, 2]) / sw # weighted mean time of the bin y_i / sigma^2
        return Xb, np.bincount(inv, weights=w * y) / sw, 1.0 / sw # inputs, fused mean, fused variance
 
    def fit_if_ready(self, min_pts=10):
        # wait to train until we have enough points
        if self.X is None or len(self.y) < min_pts:
            return False
        t0 = time.perf_counter()
        X, y, a = self._compress() 
        if self.method == "sgpr":
            m, s = ((float(np.mean(y)), float(np.std(y)) or 1.0) if self.prior == "data"
                    else (self.prior_mean, 1.0))
            Z = inducing_pcd(X, self.gp.kernel, self.gp_M)
            self.sgpr = SGPR(self.gp.kernel, fitc=self.fitc).fit(X, y, a, Z, y_mean=m, y_scale=s)
        else:
            if self.prior == "data":
                # normalize_y=True: sklearn adds alpha in standardized-y units -> rescale by Var(y)
                y_var = float(np.var(y))
                self.gp.set_params(alpha=a / (y_var if y_var > 0 else 1.0))
                y_fit = y
            else:
                self.gp.set_params(alpha=a)
                y_fit = y - self.prior_mean
            self._maybe_learn()
            self.gp.fit(X, y_fit)
            if self.learn == "once" and self.gp.optimizer is not None:
                # freeze: learned kernel becomes the fixed kernel for all later refits
                self.gp.set_params(kernel=self.gp.kernel_, optimizer=None)
                self.learn = "done"
        self.last_n, self.last_fit_s = len(y), time.perf_counter() - t0
        return True
    

    def predict_grid_at_time(self, coords_rc, t):
        """coords_rc: list of (r,c) valid states; returns mean, std."""
        if self.X is None or len(self.y) < 2:
            n = len(coords_rc)
            if self.prior == "data":
                return np.zeros(n), np.ones(n)
            return np.full(n, self.prior_mean), np.full(n, np.sqrt(self.amp))
        t0 = time.perf_counter()
        Xq = np.array([[float(r), float(c), float(t)] for (r,c) in coords_rc], dtype=float)
        mu, std = self.gp.predict(Xq, return_std=True)
        if self.method == "sgpr":
            mu, std = self.sgpr.predict(Xq)
        else:
            mu, std = self.gp.predict(Xq, return_std=True)
            if self.prior == "fixed":
                mu = mu + self.prior_mean
        self.last_predict_s = time.perf_counter() - t0
        return mu, std