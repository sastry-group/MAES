# info_maps/sgpr.py
"""
Sparse variational GP regression with inducing points (SGPR; Titsias 2009),
as in POAM Sec. III-D, Eqs. 4 and 7-9. Inducing points by pivoted Cholesky (Burt et al. 2020).

With fixed hyperparameters (learn_hypers="none") the variational posterior over the
inducing values is closed form (POAM Eqs. 7-9), so no stochastic optimization is needed.
Cost per refit: O(N M^2) instead of O(N^3); prediction O(M^2) per query.

Uses the same sklearn kernel object as the exact GP, so kernel choices carry over.
Noise is per sample (alpha), as in Eq. (3).
fitc=False (default): SGPR/VFE predictions, as in POAM.
fitc=True: adds unexplained variance diag(Kff - Qff) to the noise (Snelson & Ghahramani 2006).
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular


def inducing_grid(X, n_t=3, dt=None):
    """Inducing inputs = every visited region (r, c) at n_t recent times:
    t_now, t_now - dt, ..., t_now - (n_t-1) dt.  M = |R_visited| * n_t."""
    rc = np.unique(X[:, :2], axis=0)
    t_now = X[:, 2].max()
    dt = dt if dt else max(1.0, (t_now - X[:, 2].min()) / max(n_t - 1, 1))
    ts = t_now - dt * np.arange(n_t)
    return np.vstack([np.column_stack([rc, np.full(len(rc), t)]) for t in ts])


def inducing_pcd(X, kernel, M, tol=1e-6):
    """Pivoted Cholesky selection (Burt et al. 2019; used by POAM, OVC).
    Greedily picks the data input with the largest remaining prior variance.
    O(N M^2); never forms the full N x N kernel matrix."""
    Xu = np.unique(X, axis=0)                      # repeated inputs add nothing
    n = len(Xu)
    M = min(M, n)
    d = kernel.diag(Xu).astype(float).copy()
    L = np.zeros((M, n))
    piv = []
    for m in range(M):
        i = int(np.argmax(d))
        if d[i] < tol:
            break
        piv.append(i)
        col = kernel(Xu, Xu[i:i + 1]).ravel()
        l = (col - L[:m].T @ L[:m, i]) / np.sqrt(d[i])
        L[m] = l
        d = np.maximum(d - l ** 2, 0.0)
        d[i] = 0.0
    return Xu[piv]


class SGPR:
    def __init__(self, kernel, jitter=1e-6, fitc=False):
        self.kernel = kernel
        self.jitter = jitter
        self.fitc = fitc   # add diag(Kff - Qff) to the noise (Snelson & Ghahramani 2006)
        self.Z = None

    def fit(self, X, y, alpha, Z, y_mean=0.0, y_scale=1.0):
        """y, alpha in data units; standardized internally by (y_mean, y_scale)."""
        self.y_mean, self.y_scale = float(y_mean), float(y_scale)
        yn = (y - self.y_mean) / self.y_scale
        an = alpha / self.y_scale ** 2
        self.Z = Z
        M = len(Z)
        Kuu = self.kernel(Z) + self.jitter * np.eye(M)
        self.Lu = cholesky(Kuu, lower=True)
        W = solve_triangular(self.Lu, self.kernel(Z, X), lower=True)      # M x N
        if self.fitc:
            # data the inducing points cannot represent (e.g. old or far from Z) gets
            # extra noise instead of being forced through u -> no blow-up, calibrated sigma
            an = an + np.maximum(self.kernel.diag(X) - np.sum(W ** 2, 0), 0.0)
        s = 1.0 / np.sqrt(an)
        Wd = W * s                                                           # scale columns
        self.LB = cholesky(np.eye(M) + Wd @ Wd.T, lower=True)
        self.c = solve_triangular(self.LB, Wd @ (yn * s), lower=True)
        return self

    def predict(self, Xq, return_std=True):
        Ws = solve_triangular(self.Lu, self.kernel(self.Z, Xq), lower=True)  # M x Q
        tmp = solve_triangular(self.LB, Ws, lower=True)
        mu = tmp.T @ self.c
        var = self.kernel.diag(Xq) - np.sum(Ws ** 2, 0) + np.sum(tmp ** 2, 0)
        mu = mu * self.y_scale + self.y_mean
        std = np.sqrt(np.maximum(var, 1e-12)) * self.y_scale
        return (mu, std) if return_std else mu