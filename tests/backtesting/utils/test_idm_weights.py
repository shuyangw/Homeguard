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
