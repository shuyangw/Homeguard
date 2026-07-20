"""FxPcaDollarResidual (#39): PCA dollar-factor residual mean-reversion, market-neutral.

Statistical strategy. At each weekly rebalance, extracts a trailing-window
"dollar factor" (PC1 of standardized daily FX returns across the full 22-pair
G10 panel, via SVD -- see `src.data.artifacts.pca_dollar.dollar_factor`) and
trades the CROSS-SECTIONAL RESIDUAL left after removing that factor: pairs
whose 5-day cumulative residual return is far below its own trailing
distribution are bought (expected to mean-revert up relative to the dollar
factor); pairs far above are sold. All 22 pairs feed the factor estimation
("exotics inform the factor") but only major-tier pairs (USD-legged or
XAU/XAG metals -- see `_tier`) are eligible for a nonzero traded forecast
("but aren't traded") -- a liquidity gate on minor cross pairs (AUDNZD,
EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY, ...).

Exit-logic taxonomy (methodology Section 11.1): this strategy combines
`signal_reversal` (mean-reversion exit -- residual z crosses back through
zero), `vol_scaled_stop`-style (3-sigma disaster stop on the residual
z-score), and `conditional` (PC1 regime-break flatten) exits, layered under
a `time_fixed` weekly rebalance that re-evaluates every held leg regardless.

No-lookahead design (the critical correctness property here)
--------------------------------------------------------------
PC1 is NEVER estimated on the full sample. At each weekly rebalance date t
(every `rebalance_every`-th row of the panel -- a simple, deterministic,
causal proxy for weekly cadence since no explicit day-of-week alignment is
required here), the factor is refit via `dollar_factor()` on ONLY the
trailing `pca_window` (250) daily-return window ending at t (inclusive).
Between rebalances, the daily monitoring (mean-reversion exit / disaster
stop) does NOT refit the SVD -- it reuses that cycle's fixed loading vector
`w` and reprojects each day's OWN trailing-250-day standardized-return
window onto it (a single dot product), which is causal: today's return data
is used, but the loading DIRECTION is frozen at the last rebalance, so it
cannot see information that only arrives after that rebalance. This mirrors
a "refit slow, monitor fast" design.

Design-choice documentation (points not fully pinned down by the pre-registration)
------------------------------------------------------------------------------------
1. `dollar_factor()` returns only `(pc1, residuals)`, not the raw SVD loading
   vector `w`, but the daily-monitoring reprojection requires reusing `w`.
   Since `pc1 = Z @ w` EXACTLY by construction (`w` is a unit right-singular
   vector of `Z`, i.e. an eigenvector of `Z^T Z` with eigenvalue `sigma1^2`),
   `w` is recovered algebraically -- not approximated -- via the
   least-squares projection `w = Z^T @ pc1 / (pc1 . pc1)`:
   `Z^T @ Z @ w = sigma1^2 * w`, and `pc1 . pc1 = w^T Z^T Z w = sigma1^2`, so
   the ratio is exactly `w` (up to floating-point precision). See
   `_recover_loading`.
2. Disaster-stop "against the position" (step 9 of the pre-registration):
   for a long leg (entered because z was very negative, expecting reversion
   up), "against the position" means z has moved to an EVEN MORE negative
   extreme (`z <= -disaster_sigma`) -- the position is losing because the
   residual keeps diverging rather than reverting. Symmetric for shorts
   (`z >= +disaster_sigma`).
3. PC1 regime-break flatten (step 10): "jumped by more than 15pts" is
   interpreted as an ABSOLUTE-VALUE change (`|current - previous| >
   threshold`), not only an increase -- a large DROP in PC1's explained
   variance also signals the factor structure broke down week-over-week.
4. Insufficient-candidate tie-break (step 6): when fewer than `2*n_legs`
   major-tier pairs have a valid z-score, take
   `k = min(n_legs, valid_count // 2)` from each extreme of the sorted z
   distribution (bottom-k long / top-k short) so longs and shorts never
   overlap on the same pair; any pair left over (near the median) stays
   flat. E.g. 3 valid pairs -> 1 long, 1 short, 1 flat; 4 valid pairs ->
   2 long, 2 short (identical to the ordinary >=4-candidate case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.artifacts.pca_dollar import dollar_factor

_METALS = {"XAU", "XAG"}


def _tier(pair: str) -> str:
    if pair[:3] in _METALS or pair[3:] in _METALS:
        return "major"
    if "USD" in (pair[:3], pair[3:]):
        return "major"
    return "minor"


def _recover_loading(window_returns: pd.DataFrame, pc1: pd.Series) -> pd.Series:
    """Recover the SVD loading vector w from dollar_factor's (pc1, ...) output.

    Exact (not approximate): pc1 = Z @ w and w is a unit right-singular
    vector of Z, so Z^T @ pc1 == (pc1 . pc1) * w.
    """
    X = window_returns.dropna(how="any")
    Z = (X - X.mean()) / X.std(ddof=0)
    p = pc1.reindex(X.index).values
    w = (Z.values.T @ p) / (p @ p)
    return pd.Series(w, index=X.columns)


def _reproject_residuals(window_returns: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Project a trailing window onto a FIXED (not refit) loading vector w."""
    cols = list(w.index)
    X = window_returns[cols].dropna(how="any")
    Z = (X - X.mean()) / X.std(ddof=0)
    pc1_values = Z.values @ w.values
    proj = np.outer(pc1_values, w.values)
    residuals = pd.DataFrame(Z.values - proj, index=X.index, columns=cols)
    return residuals.reindex(window_returns.index)


def _residual_zscore(residuals: pd.DataFrame, residual_window: int) -> pd.Series:
    resid_nd = residuals.rolling(residual_window).sum()
    return (resid_nd.iloc[-1] - resid_nd.mean()) / resid_nd.std(ddof=0)


def _explained_variance_pts(window_returns: pd.DataFrame, pc1: pd.Series) -> float:
    total = window_returns.apply(lambda c: ((c - c.mean()) / c.std(ddof=0)).var()).sum()
    return float(pc1.var() / total * 100.0)


class FxPcaDollarResidual:
    def __init__(self, universe, pca_window: int = 250, residual_window: int = 5,
                 n_legs: int = 2, disaster_sigma: float = 3.0,
                 pc1_jump_threshold_pts: float = 15.0, rebalance_every: int = 5,
                 forecast_scale: float = 10.0, **params):
        self.universe = list(universe)
        self.pca_window = int(pca_window)
        self.residual_window = int(residual_window)
        self.n_legs = int(n_legs)
        self.disaster_sigma = float(disaster_sigma)
        self.pc1_jump_threshold_pts = float(pc1_jump_threshold_pts)
        self.rebalance_every = int(rebalance_every)
        self.forecast_scale = float(forecast_scale)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in close_panel.columns if c in self.universe]
        major_pairs = [p for p in present if _tier(p) == "major"]
        returns = close_panel[present].pct_change()

        n = len(close_panel)
        col_pos = {c: i for i, c in enumerate(close_panel.columns)}
        values = np.zeros((n, len(close_panel.columns)))

        rebalance_positions = list(range(0, n, self.rebalance_every))
        prev_var_pts = None
        cycle_forecast = {p: 0.0 for p in present}
        cycle_direction: dict = {}
        cycle_w = None

        for ridx, pos in enumerate(rebalance_positions):
            end_pos = (rebalance_positions[ridx + 1] if ridx + 1 < len(rebalance_positions) else n)

            if pos < self.pca_window - 1:
                cycle_forecast = {p: 0.0 for p in present}
                cycle_direction = {}
                cycle_w = None
            else:
                window = returns.iloc[pos - self.pca_window + 1: pos + 1]
                available_cols = [c for c in window.columns if window[c].notna().all()]
                window = window[available_cols]

                if len(available_cols) < 2 * self.n_legs:
                    # Not enough pairs have complete history over this trailing
                    # window to ever fill both legs -- same treatment as the
                    # "not enough history yet" branch above.
                    cycle_forecast = {p: 0.0 for p in present}
                    cycle_direction = {}
                    cycle_w = None
                else:
                    pc1, residuals = dollar_factor(window)
                    var_pts = _explained_variance_pts(window, pc1)
                    w = _recover_loading(window, pc1)

                    jumped = (prev_var_pts is not None
                              and abs(var_pts - prev_var_pts) > self.pc1_jump_threshold_pts)
                    cycle_forecast = {p: 0.0 for p in present}
                    cycle_direction = {}
                    cycle_w = w
                    if not jumped:
                        z = _residual_zscore(residuals, self.residual_window)
                        available_majors = [p for p in major_pairs if p in available_cols]
                        candidates = z[available_majors].dropna().sort_values()
                        k = min(self.n_legs, len(candidates) // 2)
                        if k > 0:
                            longs = candidates.index[:k]
                            shorts = candidates.index[-k:]
                            for p in longs:
                                cycle_forecast[p] = self.forecast_scale
                                cycle_direction[p] = 1
                            for p in shorts:
                                cycle_forecast[p] = -self.forecast_scale
                                cycle_direction[p] = -1
                    prev_var_pts = var_pts

            for d in range(pos, end_pos):
                if d > pos and cycle_w is not None and cycle_direction and d >= self.pca_window - 1:
                    day_window = returns.iloc[d - self.pca_window + 1: d + 1]
                    resid_today = _reproject_residuals(day_window, cycle_w)
                    z_today = _residual_zscore(resid_today, self.residual_window)
                    for p, direction in list(cycle_direction.items()):
                        if cycle_forecast[p] == 0.0:
                            continue
                        zt = z_today.get(p, np.nan)
                        if pd.isna(zt):
                            continue
                        if direction == 1 and (zt >= 0.0 or zt <= -self.disaster_sigma):
                            cycle_forecast[p] = 0.0
                        elif direction == -1 and (zt <= 0.0 or zt >= self.disaster_sigma):
                            cycle_forecast[p] = 0.0
                for p, val in cycle_forecast.items():
                    values[d, col_pos[p]] = val

        forecast = pd.DataFrame(values, index=close_panel.index, columns=close_panel.columns)
        return forecast.fillna(0.0)
