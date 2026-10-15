# info_maps/hedac_gp.py
"""Time-varying GP of Mantovani et al. (RA-L 2026, Sec. IV-A), which follows
Pratissoli et al. (TRO 2025, Secs. VII-VIII). Baseline-only belief for
HEDACPlanner (--hedac_gp mantovani). Same interface as GPInfoMap.

  k_SE(x, x') = sc^2 exp(-|x - x'|^2 / (2 l^2))                          Eq. (6)
  K~   = K_SE(X, X) o D o T^D                                             Eq. (8)
         D_ij   = exp(-|x_i - x_j|^2 / lam_s)                             Eq. (10)
         T^D_ij = Td_i Td_j (i != j), 1 (i = j);  Td_i = exp(-(t - t_i)/lam_t)   Eq. (9)
  k~*  = K_SE(X*, X) o d^T o Td^T,   d_i = exp(-|x_i - z|^2 / lam_s),  z = robot position
  mu = k~*^T (K~ + sn^2 I)^-1 y,   Sigma = k(X*, X*) - k~*^T (K~ + sn^2 I)^-1 k~*  Eq. (7)
  data management (no FIFO, no binning):
    add new/shared sample o   iff  sigma(x_o) >= sigma_in  = e_in  / z_N    Pratissoli (25)
    remove stored sample s    iff  sigma(x_s) >= sigma_out = e_out / z_N    Pratissoli (26)

Observations are normalized by the running max observation (Pratissoli Sec. VIII),
so sc = 1 and the thresholds are in normalized units; mu and sigma are returned
in raw units, so RMSE / MSLL stay comparable with the shared GP.

Deviations from the paper (state them):
  - kernel hyperparameters fixed (theirs: learned online); l = --l_space.
  - sn^2 from the shared sensor, Eq. (3), unless --mgp_noise > 0 (theirs: 0.01).
  - grid units: 1 region <-> d_fov = 5 m, so lam_s = 100 m^2 -> 4 cells^2;
    lam_t = 100 steps.
The cross term uses the distance to the robot, not to the query, so the joint
covariance is not guaranteed PSD; predictive variance is clipped at 0.
"""
import time
import types
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class HedacGP:

    def __init__(self, l_space=2.0, sigma_c=1.0, lam_s=4.0, lam_t=100.0,
                 e_in=0.9, e_out=0.7, z_N=1.96, noise=0.0, robot_decay=True):
        self.l_space, self.sc2 = float(l_space), float(sigma_c) ** 2
        self.lam_s, self.lam_t = float(lam_s), float(lam_t)
        self.sig_in, self.sig_out = e_in / z_N, e_out / z_N
        self.e_in, self.e_out, self.z_N = e_in, e_out, z_N
        self.noise, self.robot_decay = float(noise), bool(robot_decay)

        self.X = None            # (N, 3): [row, col, t]   stored dataset
        self.y = None            # (N,)   raw observations
        self.alpha = None        # (N,)   raw noise variances
        self.pending = []        # staged (r, c, t, y, a); filtered by Eq. (25) at the next fit
        self.z = None            # robot (r, c); set by the planner before fit / predict
        self.y_max = 0.0         # running max observation (normalization)

        self.last_n, self.last_fit_s, self.last_predict_s = 0, 0.0, 0.0
        self.n_added = self.n_rejected = self.n_removed = 0
        self.gp = types.SimpleNamespace(kernel=(  # read by ErgodicPlanner.finalize
            f"hedacGP(l={self.l_space}, lam_s={self.lam_s}, lam_t={self.lam_t}, "
            f"e_in={e_in}, e_out={e_out}, z_N={z_N})"))

    @classmethod
    def from_args(cls, args):
        return cls(l_space=args.l_space,
                   sigma_c=getattr(args, "mgp_sigma_c", 1.0),
                   lam_s=getattr(args, "mgp_lam_s", 4.0),
                   lam_t=getattr(args, "mgp_lam_t", 100.0),
                   e_in=getattr(args, "mgp_e_in", 0.9),
                   e_out=getattr(args, "mgp_e_out", 0.7),
                   z_N=getattr(args, "mgp_zN", 1.96),
                   noise=getattr(args, "mgp_noise", 0.0),
                   robot_decay=bool(getattr(args, "mgp_robot_decay", 1)))

    def attrs(self):
        return dict(gp_kernel="mantovani_se_decay", l_space=self.l_space,
                    mgp_lam_s=self.lam_s, mgp_lam_t=self.lam_t,
                    mgp_e_in=self.e_in, mgp_e_out=self.e_out, mgp_zN=self.z_N,
                    mgp_noise=self.noise, mgp_robot_decay=int(self.robot_decay),
                    learn_hypers="none")

    # ------------------------------------------------------------ data in
    def add_batch(self, r, c, t, y, noise_var=None):
        r, c, t, y = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (r, c, t, y))
        a = (np.full(len(y), 1e-3) if noise_var is None
             else np.atleast_1d(np.asarray(noise_var, dtype=float)))
        self.pending.extend(zip(r, c, t, y, a))
        if len(y):
            self.y_max = max(self.y_max, float(np.max(y)))

    def trim(self, max_pts=None):
        """No FIFO: the dataset is managed by Eqs. (25)-(26) only."""
        return

    # ------------------------------------------------------------ model
    def _s(self):
        return max(self.y_max, 1e-12)

    def _an(self, alpha):
        if self.noise > 0:
            return np.full(len(alpha), self.noise ** 2)
        return alpha / self._s() ** 2

    def _w(self, X, t):
        """Per-sample factor of the cross term: d_i * Td_i."""
        w = np.exp(-np.maximum(t - X[:, 2], 0.0) / self.lam_t)
        if self.robot_decay and self.z is not None:
            w = w * np.exp(-((X[:, :2] - np.asarray(self.z, float)) ** 2).sum(1) / self.lam_s)
        return w

    def _build(self, X, y, alpha, t):
        """Factor K~ + sn^2 I at time t. Returns the pieces needed by _post."""
        Td = np.exp(-np.maximum(t - X[:, 2], 0.0) / self.lam_t)
        D2 = ((X[:, None, :2] - X[None, :, :2]) ** 2).sum(-1)
        K = self.sc2 * np.exp(-D2 / (2 * self.l_space ** 2)) * np.exp(-D2 / self.lam_s) * np.outer(Td, Td)
        np.fill_diagonal(K, self.sc2)                     # D_ii = T^D_ii = 1
        K[np.diag_indices_from(K)] += self._an(alpha)
        cf = cho_factor(K, lower=True)
        return dict(X=X, cf=cf, b=cho_solve(cf, y / self._s()), w=self._w(X, t))

    def _post(self, m, Xq_rc):
        """Normalized posterior mean and variance at query cells (current time)."""
        D2 = ((Xq_rc[:, None, :] - m["X"][None, :, :2]) ** 2).sum(-1)
        Ks = self.sc2 * np.exp(-D2 / (2 * self.l_space ** 2)) * m["w"][None, :]
        mu = Ks @ m["b"]
        var = self.sc2 - np.einsum("qn,nq->q", Ks, cho_solve(m["cf"], Ks.T))
        return mu, np.maximum(var, 0.0)

    def _sigma_at(self, m, r, c):
        return float(np.sqrt(self._post(m, np.array([[r, c]], float))[1][0]))

    # ------------------------------------------------------------ fit
    def fit_if_ready(self, min_pts=1):
        if self.X is None and not self.pending:
            return False
        t0 = time.perf_counter()
        t_now = max([p[2] for p in self.pending] + ([self.X[:, 2].max()] if self.X is not None else []))

        # Eq. (25): admit staged samples one at a time, against the current dataset
        m = None if self.X is None else self._build(self.X, self.y, self.alpha, t_now)
        for (r, c, t, y, a) in self.pending:
            if m is not None and self._sigma_at(m, r, c) < self.sig_in:
                self.n_rejected += 1
                continue
            row = np.array([[r, c, t]])
            if self.X is None:
                self.X, self.y, self.alpha = row, np.array([y]), np.array([a])
            else:
                self.X = np.vstack([self.X, row])
                self.y = np.append(self.y, y)
                self.alpha = np.append(self.alpha, a)
            self.n_added += 1
            m = self._build(self.X, self.y, self.alpha, t_now)
        self.pending = []

        # Eq. (26): drop stored samples whose own-location sigma reached sigma_out
        if self.X is not None:
            sig = np.sqrt(self._post(m, self.X[:, :2])[1])
            keep = sig < self.sig_out
            self.n_removed += int((~keep).sum())
            if keep.all():
                pass
            elif keep.any():
                self.X, self.y, self.alpha = self.X[keep], self.y[keep], self.alpha[keep]
            else:
                self.X = self.y = self.alpha = None

        self.last_n = 0 if self.X is None else len(self.y)
        self.last_fit_s = time.perf_counter() - t0
        return True

    # ------------------------------------------------------------ predict
    def predict_grid_at_time(self, coords_rc, t):
        """Raw-unit mean / std at time t (decay re-evaluated at t and at the current z)."""
        n = len(coords_rc)
        if self.X is None:
            return np.zeros(n), np.full(n, np.sqrt(self.sc2) * self._s())
        t0 = time.perf_counter()
        m = self._build(self.X, self.y, self.alpha, float(t))
        mu, var = self._post(m, np.asarray(coords_rc, float))
        self.last_predict_s = time.perf_counter() - t0
        s = self._s()
        return mu * s, np.sqrt(var) * s