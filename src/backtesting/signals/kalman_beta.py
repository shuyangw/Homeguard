"""Causal time-varying regression beta via a Kalman FILTER (never a smoother).

Dynamic linear regression y_t = alpha_t + beta_t * x_t + e_t, with the state
theta_t = [alpha_t, beta_t]' following a random walk theta_t = theta_{t-1} + w_t.
This is the standard estimator for a hedge ratio that drifts, replacing a static
or rolling-window OLS.

CAUSALITY (the whole point): only the forward filtering pass is implemented.
theta_t is conditioned on observations up to and including t and is NEVER revised
by later data. The RTS / fixed-interval SMOOTHER, which does revise past states
using future observations, is deliberately NOT provided -- using one inside a
return-generating path is a lookahead bias that manufactures fake performance.

Q and R are inputs, not fitted here: the caller must fix them a priori or estimate
them on training data only. Tuning them until a backtest passes is a specification
search.
"""
from __future__ import annotations

import numpy as np


def select_warmup_indices(y: np.ndarray, x: np.ndarray, size: int) -> np.ndarray | None:
    """Indices of the first `size` FINITE, aligned (y, x) observations.

    Robust to leading or interior non-finite rows (e.g. a panel that starts
    before real price history begins, or a handful of nulled data-quality
    bars) -- unlike a literal `[:size]` slice, which is poisoned by a single
    NaN anywhere in the first `size` rows. Returns None if fewer than `size`
    finite, aligned pairs exist anywhere in the input.
    """
    finite = np.isfinite(y) & np.isfinite(x)
    idx = np.flatnonzero(finite)
    return idx[:size] if len(idx) >= size else None


def causal_dynamic_beta(y: np.ndarray, x: np.ndarray, delta: float, r_var: float,
                        warmup: int) -> tuple[np.ndarray, np.ndarray]:
    """Filtered (alpha, beta) paths for y ~ alpha + beta*x.

    Args:
        y, x: observation arrays (same length).
        delta: random-walk process-noise parameter; Q = delta/(1-delta) * I.
            Smaller = a more slowly-varying (smoother) beta.
        r_var: observation-noise variance R (> 0), estimated on training data only.
        warmup: number of leading FINITE, aligned observations used to seed
            theta_0 by OLS (via `select_warmup_indices`, robust to leading/
            interior NaNs). The returned paths are NaN before the warmup
            completes -- i.e. before the last of those `warmup` observations.

    Returns:
        (alpha_path, beta_path), each length len(y), NaN before warmup
        completes. Entry t is conditioned on observations 0..t only.
    """
    n = len(y)
    alpha_path = np.full(n, np.nan)
    beta_path = np.full(n, np.nan)
    if n <= warmup or warmup < 2 or not np.isfinite(r_var) or r_var <= 0:
        return alpha_path, beta_path

    seed_idx = select_warmup_indices(y, x, warmup)
    if seed_idx is None:
        return alpha_path, beta_path
    y0, x0 = y[seed_idx], x[seed_idx]
    slope, intercept = np.polyfit(x0, y0, 1)

    theta = np.array([float(intercept), float(slope)])
    P = np.eye(2) * 1e-3
    Q = np.eye(2) * (delta / (1.0 - delta))
    seed_end = int(seed_idx[-1])
    alpha_path[seed_end] = theta[0]
    beta_path[seed_end] = theta[1]

    for t in range(seed_end + 1, n):
        if not (np.isfinite(y[t]) and np.isfinite(x[t])):
            alpha_path[t], beta_path[t] = theta[0], theta[1]
            continue
        P_pred = P + Q                          # predict (random walk: theta unchanged)
        H = np.array([1.0, float(x[t])])
        innov = float(y[t]) - H @ theta         # innovation, using data up to t only
        S = float(H @ P_pred @ H + r_var)
        if not np.isfinite(S) or S <= 0:
            alpha_path[t], beta_path[t] = theta[0], theta[1]
            continue
        K = (P_pred @ H) / S                    # Kalman gain
        theta = theta + K * innov               # update
        P = P_pred - np.outer(K, H) @ P_pred
        alpha_path[t], beta_path[t] = theta[0], theta[1]

    return alpha_path, beta_path
