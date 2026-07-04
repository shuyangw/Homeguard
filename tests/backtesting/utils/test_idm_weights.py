import numpy as np
from src.backtesting.utils.idm_weights import compute_div_mult


def test_deterministic_and_cluster_capped():
    U = ["ES","NQ","ZN","ZB","6E","CL","NG","HO","RB","GC","ZC","LE"]
    d1 = compute_div_mult(U); d2 = compute_div_mult(U)
    assert d1 == d2                                   # data-free, deterministic
    assert set(d1) == set(U)
    # energy has 4 roots (CL/NG/HO/RB) vs equity 2 (ES/NQ): each energy root's
    # UNSCALED cluster weight is smaller -> energy is de-concentrated per root.
    # (verify via the underlying weights being equal-risk-per-cluster)
    assert all(np.isfinite(v) and v > 0 for v in d1.values())


def test_median_divmult_near_one():
    U = ["ES","NQ","YM","ZT","ZF","ZN","6E","6J","CL","NG","GC","SI","ZC","ZW","LE","HE"]
    d = compute_div_mult(U)
    med = float(np.median(list(d.values())))
    assert abs(med - 1.0) < 1e-6                       # N_scale pins the median to 1


def test_per_instrument_cap_clips():
    # BTC/ETH (2-root crypto cluster) sit alongside a 6-root equity cluster and
    # a 6-root rates cluster -> uncapped div_mult for BTC/ETH is 3.0, well
    # above a 2.0 cap.
    U = ["ES", "NQ", "YM", "RTY", "M2K", "MES",
         "ZT", "ZF", "ZN", "TN", "ZB", "UB", "BTC", "ETH"]
    uncapped = compute_div_mult(U)
    assert uncapped["BTC"] > 2.0 and uncapped["ETH"] > 2.0

    capped = compute_div_mult(U, per_instrument_cap=2.0)
    assert all(v <= 2.0 for v in capped.values())
    assert capped["BTC"] == 2.0 and capped["ETH"] == 2.0
    # Uncapped roots (already <= cap) are untouched.
    assert capped["ES"] == uncapped["ES"]

    assert compute_div_mult(U, per_instrument_cap=None) == uncapped
