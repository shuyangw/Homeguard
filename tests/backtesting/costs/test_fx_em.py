"""EM per-pair spread costs: EM pairs priced in bps (wider), G10 unchanged."""
from src.backtesting.costs.fx import fx_round_trip_usd, _em_half_bps


def test_em_leg_detected():
    assert _em_half_bps("USDMXN") == 3.0
    assert _em_half_bps("USDTRY") == 15.0
    assert _em_half_bps("EURUSD") is None
    # both-EM cross takes the wider leg
    assert _em_half_bps("TRYZAR") == 15.0


def test_em_pair_priced_in_round_trip_bps():
    # USDMXN, 1M base units at price 18, quote_to_usd 1/18 -> $1M notional.
    # 3 bps/side -> 6 bps round trip -> $600.
    cost = fx_round_trip_usd("USDMXN", 1_000_000, 18.0, 1 / 18.0)
    notional = 1_000_000 * 18.0 * (1 / 18.0)
    assert abs(cost - notional * 6.0 / 1e4) < 1e-6
    assert abs(cost - 600.0) < 1e-6


def test_em_wider_than_g10_major():
    # same $1M notional: EM (USDZAR 6bps/side) costs more than a G10 major.
    em = fx_round_trip_usd("USDZAR", 1_000_000, 18.0, 1 / 18.0)
    g10 = fx_round_trip_usd("EURUSD", 1_000_000, 1.0, 1.0)
    assert em > g10


def test_g10_unchanged_uses_pip_path():
    # EURUSD is not EM -> pip path, independent of the EM bps map.
    cost = fx_round_trip_usd("EURUSD", 1_000_000, 1.08, 1.0)
    assert cost > 0
    assert _em_half_bps("EURUSD") is None
