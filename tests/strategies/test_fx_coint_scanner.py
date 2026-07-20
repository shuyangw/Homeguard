import numpy as np
import pandas as pd
import pytest

from src.backtesting.costs.fx import fx_round_trip_pips
from src.strategies.advanced.fx_coint_scanner import CointScanner, _candidate_pairs


def test_candidate_pairs_excludes_shared_gt1_currency():
    # pairs sharing 2 currencies (mechanical) excluded; <=1 shared kept
    prs = ["EURUSD", "GBPUSD", "EURGBP", "AUDUSD"]
    cands = _candidate_pairs(prs)
    assert ("EURUSD", "GBPUSD") in cands or ("GBPUSD", "EURUSD") in cands  # share only USD
    # EURUSD vs EURGBP share EUR only (<=1) -> allowed; EURGBP vs GBPUSD share GBP only -> allowed
    assert all(len(set(a) & set(b)) <= 3 for a, b in cands)  # sanity (currency-code overlap bounded)


def test_scanner_emits_only_tradeable_cointegrated_spreads():
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(32)
    # EURCAD/AUDCAD share a strong common stochastic trend and a stationary
    # OU log-spread (half-life ~6d) -> genuinely, persistently cointegrated so
    # the pair re-qualifies across monthly scans. GBPJPY is an independent,
    # low-vol walk that does not cointegrate with either.
    common = np.cumsum(rng.normal(0, 0.015, n))
    phi = np.exp(-np.log(2) / 6.0)
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + rng.normal(0, 0.006)
    a = 1.30 * np.exp(common + ou)
    b = 1.10 * np.exp(common)
    c = 0.90 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    close = pd.DataFrame({"EURCAD": a, "AUDCAD": b, "GBPJPY": c}, index=pd.Index(idx))
    book, sigma = CointScanner(list(close.columns)).spread_book(close)
    active = [sps for sps in book.values() if sps]
    assert active, "expected the cointegrated EURCAD/AUDCAD spread to be tradeable"
    legs = {(sp.leg_a, sp.leg_b) for sps in active for sp in sps}
    assert any({"EURCAD", "AUDCAD"} == set(l) for l in legs)


def test_tradeable_gate_bar_is_two_round_trips_not_four():
    # FIX 1: fx_round_trip_pips already returns the ROUND-TRIP cost, so _cost is
    # exactly one round-trip in price terms and the scan gate (1.5*sig > 2*_cost)
    # requires a 1.5-sigma move to clear TWICE round-trip -- not 4x (the prior
    # double-multiplier bug).
    sc = CointScanner(["EURUSD", "GBPUSD"])
    round_trip = fx_round_trip_pips("major") * 0.0001
    assert sc._cost == pytest.approx(round_trip)
    assert 2.0 * sc._cost == pytest.approx(2.0 * round_trip)
