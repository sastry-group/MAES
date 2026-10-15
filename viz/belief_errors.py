# viz/belief_errors.py
"""
GP belief accuracy: RMSE and MSLL of the posterior against the true field.
Separate from metrics.py, which holds the ergodic/coverage quantities
(E_k, L_belief, KL, cover time).

Shapes follow run_io: mu / std are (T, N, nstates) per agent or (T, nstates)
pooled; true_phi is (T, nstates), already dense and aligned.
"""
import numpy as np


def _mask(arr, mask, axis=-1):
    return arr if mask is None else np.take(arr, np.flatnonzero(mask), axis=axis)


def rmse(mu, true_phi, mask=None):
    """(T,) if mu is (T, nstates); (T, N) if mu is (T, N, nstates)."""
    mu = np.asarray(mu, dtype=float)
    true_phi = np.asarray(true_phi, dtype=float)
    if mu.ndim == 2:
        return np.sqrt(_mask((mu - true_phi) ** 2, mask, 1).mean(axis=1))
    if mu.ndim == 3:
        return np.sqrt(_mask((mu - true_phi[:, None, :]) ** 2, mask, 2).mean(axis=2))
    raise ValueError(f"mu must be 2-D or 3-D, got {mu.shape}")


def predictive_nll(mu, std, true_phi, sigma_eps=0.0, mask=None, eps=1e-12):
    """
    Mean negative log predictive density of the true field under N(mu, var).

    sigma_eps=0.0 (default): var = sigma^2. This is the right choice when
    scoring against the NOISE-FREE true_phi, which is what we do.
    sigma_eps>0: var = sigma^2 + sigma_eps^2, for scoring noisy held-out
    observations instead. Pick one and state it in the paper.
    """
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    true_phi = np.asarray(true_phi, dtype=float)
    var = np.clip(std ** 2 + np.asarray(sigma_eps, dtype=float) ** 2, eps, None)
    tgt = true_phi if mu.ndim == 2 else true_phi[:, None, :]
    nll = 0.5 * np.log(2 * np.pi * var) + (tgt - mu) ** 2 / (2 * var)
    return _mask(nll, mask, -1).mean(axis=-1)


def msll(mu, std, true_phi, sigma_eps=0.0, mask=None,
         baseline_mean=None, baseline_var=None, eps=1e-12):
    """
    Mean standardized log loss (Rasmussen & Williams, Eq. 2.34): predictive NLL
    minus the NLL of a trivial N(baseline_mean, baseline_var) predictor.
    Negative means the GP beats the trivial model. Baseline defaults to the
    mean/variance of the field at t=0.
    """
    true_phi = np.asarray(true_phi, dtype=float)
    if baseline_mean is None:
        baseline_mean = float(_mask(true_phi[0], mask).mean())
    if baseline_var is None:
        baseline_var = max(float(_mask(true_phi[0], mask).var()), eps)
    triv = 0.5 * np.log(2 * np.pi * baseline_var) + \
        (true_phi - baseline_mean) ** 2 / (2 * baseline_var)
    triv = _mask(triv, mask, -1).mean(axis=-1)                     # (T,)
    nll = predictive_nll(mu, std, true_phi, sigma_eps, mask=mask, eps=eps)
    return nll - triv if nll.ndim == 1 else nll - triv[:, None]


def team_average(per_agent):
    """(T, N) -> (T,), ignoring dead-agent NaNs. Pass-through for (T,)."""
    per_agent = np.asarray(per_agent, dtype=float)
    return per_agent if per_agent.ndim == 1 else np.nanmean(per_agent, axis=1)