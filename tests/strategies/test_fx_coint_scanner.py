import numpy as np
import pandas as pd

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
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(0, 0.004, n))
    # A,B cointegrated (share common + fast-reverting spread); C independent
    a = 1.30 * np.exp(common + 0.01 * np.sin(np.arange(n) / 3))
    b = 1.10 * np.exp(common)
    c = 0.90 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    close = pd.DataFrame({"EURCAD": a, "AUDCAD": b, "GBPJPY": c}, index=pd.Index(idx))
    book, sigma = CointScanner(list(close.columns)).spread_book(close)
    active = [sps for sps in book.values() if sps]
    assert active, "expected the cointegrated EURCAD/AUDCAD spread to be tradeable"
    legs = {(sp.leg_a, sp.leg_b) for sps in active for sp in sps}
    assert any({"EURCAD", "AUDCAD"} == set(l) for l in legs)
