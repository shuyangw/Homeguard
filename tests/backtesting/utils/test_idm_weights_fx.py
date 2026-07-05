from src.backtesting.utils.idm_weights import compute_div_mult
from src.data.fx.clusters import fx_cluster_for


def test_fx_cluster_assignments():
    assert fx_cluster_for("EURUSD") == "usd_major"
    assert fx_cluster_for("USDJPY") == "usd_major"
    assert fx_cluster_for("EURGBP") == "eur_cross"
    assert fx_cluster_for("XAUUSD") == "metal"


def test_compute_div_mult_accepts_fx_cluster_fn():
    universe = ["EURUSD", "USDJPY", "XAUUSD"]
    dm = compute_div_mult(universe, cluster_fn=fx_cluster_for)
    assert set(dm) == set(universe)
    assert all(v > 0 for v in dm.values())
