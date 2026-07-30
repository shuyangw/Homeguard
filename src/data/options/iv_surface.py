"""M7 -- the smoothed implied-volatility surface.

Per-expiry, per-session raw-SVI fit on snapshot quote mids. Populates
`iv_smooth` and a smoothed delta so P1 can select strikes below |delta| 0.10,
where the vendor's shipped Black-Scholes greeks are not trustworthy.

Registered choices live in
`docs/strategies/research/options-slate/20260730_m7_prereg.md`, committed
before the first fit was run. The load-bearing ones:

* **Raw SVI**, not a spline. `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`
  in total variance `w = iv^2 * T` against `k = log(K/F)`. SVI's wings are
  linear in `k` by construction -- the correct large-strike asymptotic, and a
  bounded one. A spline's wing is governed by knot placement and can produce
  unbounded curvature artifacts at exactly the 0.05-delta strikes OPT-047 reads.

* **Dividend-awareness comes from the parity-implied forward**, not from a
  realized dividend series. `F` is solved from the session's own call/put
  quotes; it therefore embeds the market's dividend expectation AS PRICED at
  the snapshot minute. Using a realized (later-paid) dividend history to build
  a historical forward would inject information unknowable at the time -- a
  lookahead leak under spec v2 Section 1.3. The discount factor comes from the
  repo's existing FRED reader.

* **Refusal, never a silent number.** Where a fit is impossible or untrustworthy
  the surface emits NULL plus a reason code. P1 must be able to distinguish
  "no surface here" from "surface says 0.05".

INPUT: `options_chain_eod` (SPY 2017-01..2025-12, QQQ 2012-06..2025-12).
Note that table's greek columns are named `implied_vol` / `delta` / `theta` /
`vega` / `underlying_px` -- NOT the `*_shipped` / `spot` names some specs use.
M7 does not consume them: it re-derives IV from quote mids, so the vendor's
`implied_vol == 0.5` solver sentinel cannot contaminate the fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from src.backtesting.vol.atm_iv import black76_iv
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Registered reason codes (see the pre-registration doc, Section 4)
# --------------------------------------------------------------------------

REASON_OK = "OK"
REASON_ZERO_DTE = "ZERO_DTE"
REASON_DTE_OUT_OF_RANGE = "DTE_OUT_OF_RANGE"
REASON_NO_FORWARD = "NO_FORWARD"
REASON_IMPLAUSIBLE_FORWARD = "IMPLAUSIBLE_FORWARD"
REASON_TOO_FEW_STRIKES = "TOO_FEW_STRIKES"
REASON_ONE_SIDED = "ONE_SIDED"
REASON_FIT_FAILED = "FIT_FAILED"
REASON_ARB_VIOLATION = "ARB_VIOLATION"

#: Registered thresholds. Fixed blind; NOT to be tuned to reduce refusal rate.
MIN_PAIRED_STRIKES = 4
MIN_FIT_POINTS = 8
MIN_POINTS_PER_SIDE = 3
MIN_MID = 0.05
MAX_DTE = 400
IV_SPREAD_FLOOR = 0.005
DIV_YIELD_BOUNDS = (-0.10, 0.15)

#: FRED series used to build the zero curve, by maturity in years.
RATE_CURVE_SERIES = (
    (1.0 / 12.0, "DGS1MO"),
    (0.25, "DGS3MO"),
    (0.5, "DGS6MO"),
    (1.0, "DGS1"),
    (2.0, "DGS2"),
)


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self) -> tuple:
        return (self.a, self.b, self.rho, self.m, self.sigma)


@dataclass(frozen=True)
class ForwardFit:
    forward: float
    discount: float
    implied_div_yield: float
    n_pairs: int
    reason: str


@dataclass(frozen=True)
class ExpiryFit:
    params: Optional[SVIParams]
    reason: str
    forward: float
    discount: float
    T: float
    n_points: int
    rmse_vol_points: float
    max_abs_resid_vol_points: float
    k_min: float
    k_max: float


# --------------------------------------------------------------------------
# The parameterization
# --------------------------------------------------------------------------


def svi_total_variance(k, p: SVIParams):
    """Raw-SVI total implied variance w(k) = iv^2 * T."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def smooth_iv(p: SVIParams, k, T: float):
    """Smoothed implied volatility at log-moneyness `k`."""
    w = np.maximum(svi_total_variance(k, p), 1e-12)
    return np.sqrt(w / T)


def svi_is_arbitrage_free(p: SVIParams, T: float) -> bool:
    """The registered static no-arbitrage constraint set (pre-reg Section 5)."""
    if not np.isfinite(p.as_tuple()).all():
        return False
    if p.b < 0 or p.sigma <= 0 or abs(p.rho) >= 1.0:
        return False
    # total variance non-negative everywhere
    if p.a + p.b * p.sigma * np.sqrt(1.0 - p.rho**2) < 0:
        return False
    # Lee's wing-slope bound -- beyond it the density explodes
    if p.b * (1.0 + abs(p.rho)) > 4.0 / T:
        return False
    return True


def _gatheral_g(k, p: SVIParams):
    """Gatheral's g(k). g >= 0 everywhere <=> no butterfly arbitrage."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    root = np.sqrt(x**2 + p.sigma**2)
    w = p.a + p.b * (p.rho * x + root)
    w = np.maximum(w, 1e-12)
    wp = p.b * (p.rho + x / root)
    wpp = p.b * p.sigma**2 / root**3
    term = 1.0 - 0.5 * k * wp / w
    return term**2 - 0.25 * wp**2 * (1.0 / w + 0.25) + 0.5 * wpp


def butterfly_violation_rate(k, p: SVIParams, T: float) -> float:
    """Fraction of the `k` grid where the risk-neutral density is negative.

    Measured and reported. NEVER silently repaired.
    """
    g = _gatheral_g(k, p)
    return float(np.mean(g < 0.0))


def calendar_violation_rate(k, near: SVIParams, far: SVIParams) -> float:
    """Fraction of `k` where total variance FALLS with maturity (calendar arb)."""
    w_near = svi_total_variance(k, near)
    w_far = svi_total_variance(k, far)
    return float(np.mean(w_far < w_near - 1e-12))


# --------------------------------------------------------------------------
# Black-76 (forward measure). IV inversion reuses src/backtesting/vol/atm_iv.
# --------------------------------------------------------------------------


def black76_price(F: float, K: float, T: float, sigma: float, right: str,
                  D: float = 1.0) -> float:
    """Undiscounted-forward Black-76 price, discounted by `D`."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
        return D * intrinsic
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v**2) / v
    d2 = d1 - v
    if right == "C":
        return float(D * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(D * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def _iv_from_price(price: float, F: float, K: float, T: float, D: float,
                   right: str) -> float:
    """Invert a DISCOUNTED price to Black-76 IV. NaN when not invertible."""
    if not np.isfinite(price) or price <= 0 or D <= 0:
        return float("nan")
    r = -np.log(D) / T if T > 0 else 0.0
    return black76_iv(price, F, K, T, r, right)


def smooth_delta(p: SVIParams, k: float, T: float, F: float, D: float,
                 spot: float, right: str) -> float:
    """Smoothed SPOT delta -- the convention P1 selects strikes on.

    dC/dS = exp(-qT) * N(d1), and exp(-qT) = D * F / S falls straight out of
    the parity-implied forward. No dividend series is needed.
    """
    sigma = float(smooth_iv(p, k, T))
    if not np.isfinite(sigma) or sigma <= 0 or T <= 0:
        return float("nan")
    v = sigma * np.sqrt(T)
    d1 = (-k + 0.5 * v**2) / v  # k = log(K/F) so log(F/K) = -k
    carry = D * F / spot
    if right == "C":
        return float(carry * norm.cdf(d1))
    return float(-carry * norm.cdf(-d1))


# --------------------------------------------------------------------------
# Discount curve (repo's existing FRED reader -- no new rate source)
# --------------------------------------------------------------------------


def zero_rate(d: date, T: float) -> float:
    """Continuously-compounded zero rate at maturity `T`, from the FRED curve.

    Linearly interpolated in maturity across DGS1MO..DGS2, flat-extrapolated at
    both ends. Causal: `get_fred_series` forward-fills from observations <= d.
    """
    from src.data.rates.fred_reader import get_fred_series

    mats, rates = [], []
    for mat, series in RATE_CURVE_SERIES:
        try:
            pct = get_fred_series(series, d)
        except (FileNotFoundError, ValueError) as exc:
            logger.debug(f"[!] {series} unavailable at {d}: {exc}")
            continue
        if pct is None or not np.isfinite(pct):
            continue
        mats.append(mat)
        rates.append(np.log1p(float(pct) / 100.0))
    if not mats:
        raise ValueError(f"[-] no FRED curve point available on or before {d}")
    return float(np.interp(T, mats, rates))


# --------------------------------------------------------------------------
# The parity-implied forward
# --------------------------------------------------------------------------


def implied_forward(strikes_c, mids_c, strikes_p, mids_p, D: float,
                    spot: float, T: float = 1.0,
                    spreads_c=None, spreads_p=None) -> ForwardFit:
    """Solve the forward from put-call parity on the session's own quotes.

    For every strike quoted on both sides, `F_k = K + (C - P) / D`. The estimate
    is the median of `F_k` over the paired strikes nearest spot, which is robust
    to a single stale quote. This is what makes the surface dividend-aware.
    """
    empty = ForwardFit(float("nan"), D, float("nan"), 0, REASON_NO_FORWARD)
    strikes_c = np.asarray(strikes_c, dtype=float)
    strikes_p = np.asarray(strikes_p, dtype=float)
    mids_c = np.asarray(mids_c, dtype=float)
    mids_p = np.asarray(mids_p, dtype=float)
    if strikes_c.size == 0 or strikes_p.size == 0:
        return empty

    call_at = dict(zip(strikes_c, mids_c))
    put_at = dict(zip(strikes_p, mids_p))
    shared = sorted(set(call_at) & set(put_at))
    shared = [k for k in shared
              if np.isfinite(call_at[k]) and np.isfinite(put_at[k])]
    if len(shared) < MIN_PAIRED_STRIKES:
        return empty

    # Nearest-to-spot pairs carry the tightest two-sided markets.
    ordered = sorted(shared, key=lambda k: abs(k - spot))
    used = sorted(ordered[: max(MIN_PAIRED_STRIKES, min(len(ordered), 20))])
    f_est = np.array([k + (call_at[k] - put_at[k]) / D for k in used])
    forward = float(np.median(f_est))

    if not np.isfinite(forward) or forward <= 0:
        return empty

    r = -np.log(D) / T if (T > 0 and D > 0) else 0.0
    q = r - np.log(forward / spot) / T if (T > 0 and spot > 0) else float("nan")
    reason = REASON_OK
    if np.isfinite(q) and not (DIV_YIELD_BOUNDS[0] <= q <= DIV_YIELD_BOUNDS[1]):
        reason = REASON_IMPLAUSIBLE_FORWARD
    return ForwardFit(forward, D, float(q), len(used), reason)


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def _refused(reason: str, F: float, D: float, T: float,
             n: int = 0) -> ExpiryFit:
    nan = float("nan")
    return ExpiryFit(None, reason, F, D, T, n, nan, nan, nan, nan)


def fit_expiry(strikes, rights, bids, asks, forward: float, discount: float,
               T: float, dte: int, spot: float) -> ExpiryFit:
    """Fit one (root, session, expiry) slice. Returns a refusal, never a guess."""
    F, D = float(forward), float(discount)
    if dte == 0 or T <= 0:
        return _refused(REASON_ZERO_DTE, F, D, T)
    if dte > MAX_DTE:
        return _refused(REASON_DTE_OUT_OF_RANGE, F, D, T)
    if not np.isfinite(F) or F <= 0:
        return _refused(REASON_NO_FORWARD, F, D, T)

    strikes = np.asarray(strikes, dtype=float)
    rights = np.asarray(rights, dtype=object)
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    mids = 0.5 * (bids + asks)

    # Registered filter 1: OTM only. Filter 2: penny-option floor.
    otm = np.where(strikes >= F, rights == "C", rights == "P")
    keep = otm & np.isfinite(mids) & (mids >= MIN_MID)
    if keep.sum() < MIN_FIT_POINTS:
        return _refused(REASON_TOO_FEW_STRIKES, F, D, T, int(keep.sum()))

    k_all, w_all, weights = [], [], []
    for K, right, bid, ask, mid in zip(
        strikes[keep], rights[keep], bids[keep], asks[keep], mids[keep]
    ):
        iv = _iv_from_price(mid, F, K, T, D, right)
        if not np.isfinite(iv) or not (0.01 < iv < 5.0):
            continue
        iv_b = _iv_from_price(bid, F, K, T, D, right)
        iv_a = _iv_from_price(ask, F, K, T, D, right)
        band = (iv_a - iv_b) if (np.isfinite(iv_a) and np.isfinite(iv_b)) else np.nan
        if not np.isfinite(band) or band <= 0:
            band = IV_SPREAD_FLOOR
        k_all.append(np.log(K / F))
        w_all.append(iv**2 * T)
        weights.append(1.0 / max(band, IV_SPREAD_FLOOR))

    k = np.asarray(k_all)
    w = np.asarray(w_all)
    wt = np.asarray(weights)
    n = k.size
    if n < MIN_FIT_POINTS:
        return _refused(REASON_TOO_FEW_STRIKES, F, D, T, n)
    if (k < 0).sum() < MIN_POINTS_PER_SIDE or (k > 0).sum() < MIN_POINTS_PER_SIDE:
        return _refused(REASON_ONE_SIDED, F, D, T, n)

    wt = wt / wt.mean()
    atm_w = float(np.interp(0.0, k, w)) if n else 0.04 * T

    def residual(theta):
        a, b, rho, m, s = theta
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
        return wt * (model - w)

    # `b` is box-bounded by 2/T, NOT 4/T. Since |rho| < 1, b <= 2/T is a
    # SUFFICIENT condition for the Lee bound b*(1+|rho|) <= 4/T that
    # `svi_is_arbitrage_free` enforces. Bounding at 4/T instead lets the
    # optimizer settle in a region the box permits but the check rejects,
    # which turns a solvable slice into a spurious ARB_VIOLATION refusal.
    slope_cap = max(2.0 / T, 1e-6)
    lower = [-5.0 * atm_w - 1e-6, 0.0, -0.999, k.min() - 1.0, 1e-4]
    upper = [5.0 * atm_w + 1.0, slope_cap, 0.999, k.max() + 1.0, 5.0]
    best = None
    for m0 in (0.0, float(np.median(k))):
        for s0 in (0.10, 0.35):
            for rho0 in (-0.7, -0.2):
                x0 = [max(atm_w * 0.5, 1e-6), min(0.10, slope_cap * 0.5),
                      rho0, m0, s0]
                x0 = list(np.clip(x0, lower, upper))
                try:
                    sol = least_squares(residual, x0, bounds=(lower, upper),
                                        method="trf", max_nfev=2000)
                except Exception as exc:  # numerical failure -> refuse, never guess
                    logger.debug(f"[!] SVI solve raised: {exc}")
                    continue
                if best is None or sol.cost < best.cost:
                    best = sol
    if best is None:
        return _refused(REASON_FIT_FAILED, F, D, T, n)

    params = SVIParams(*[float(v) for v in best.x])
    if not svi_is_arbitrage_free(params, T):
        return _refused(REASON_ARB_VIOLATION, F, D, T, n)

    model_iv = smooth_iv(params, k, T)
    obs_iv = np.sqrt(np.maximum(w, 1e-12) / T)
    resid = model_iv - obs_iv
    return ExpiryFit(
        params=params,
        reason=REASON_OK,
        forward=F,
        discount=D,
        T=T,
        n_points=n,
        rmse_vol_points=float(np.sqrt(np.mean(resid**2))),
        max_abs_resid_vol_points=float(np.max(np.abs(resid))),
        k_min=float(k.min()),
        k_max=float(k.max()),
    )
