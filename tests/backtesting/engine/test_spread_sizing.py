import math

from src.backtesting.engine.spread_sizing import Spread, spread_leg_targets


def test_beta_weighted_notional_ratio():
    # One spread, beta=1.5: notional_B = 1.5 * notional_A, opposite signs.
    sp = [Spread("AUDUSD", "NZDUSD", 1.5, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.01}  # 1% daily spread vol
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0}  # both USD-quoted -> quote_to_usd=1
    tgt = spread_leg_targets(sp, sigma, close, q, equity=100000.0,
                             vol_target=0.10, idm=1.0)
    notional_a = tgt["AUDUSD"] * close["AUDUSD"] * q["AUDUSD"]
    notional_b = tgt["NZDUSD"] * close["NZDUSD"] * q["NZDUSD"]
    assert notional_a > 0 and notional_b < 0            # long A, short B
    assert math.isclose(abs(notional_b), 1.5 * abs(notional_a), rel_tol=1e-6)


def test_spread_vol_targets_to_vol_target():
    # notional_A chosen so the spread's annualized vol == vol_target.
    sp = [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.008}
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0}
    eq, vt = 100000.0, 0.10
    tgt = spread_leg_targets(sp, sigma, close, q, eq, vt, idm=1.0)
    notional_a = abs(tgt["AUDUSD"] * close["AUDUSD"] * q["AUDUSD"])
    # spread annualized vol = (notional_a/equity) * sigma_s * sqrt(252) == vt
    implied_vol = (notional_a / eq) * sigma[("AUDUSD", "NZDUSD")] * math.sqrt(252)
    assert math.isclose(implied_vol, vt, rel_tol=1e-6)


def test_shared_leg_nets_across_spreads():
    # Two spreads both long AUDUSD -> net AUDUSD units add.
    sp = [Spread("AUDUSD", "NZDUSD", 1.0, 10.0),
          Spread("AUDUSD", "USDCAD", 1.0, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.01, ("AUDUSD", "USDCAD"): 0.01}
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60, "USDCAD": 1.35}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0, "USDCAD": 1.0 / 1.35}
    tgt = spread_leg_targets(sp, sigma, close, q, 100000.0, 0.10, idm=1.0)
    single = spread_leg_targets([sp[0]], {("AUDUSD", "NZDUSD"): 0.01},
                                close, q, 100000.0, 0.10, idm=1.0)
    assert tgt["AUDUSD"] > single["AUDUSD"]   # two long-A spreads add


def test_strength_scales_size_linearly():
    base = spread_leg_targets([Spread("AUDUSD", "NZDUSD", 1.0, 10.0)],
                              {("AUDUSD", "NZDUSD"): 0.01},
                              {"AUDUSD": 0.65, "NZDUSD": 0.60},
                              {"AUDUSD": 1.0, "NZDUSD": 1.0}, 100000.0, 0.10, 1.0)
    half = spread_leg_targets([Spread("AUDUSD", "NZDUSD", 1.0, 5.0)],
                              {("AUDUSD", "NZDUSD"): 0.01},
                              {"AUDUSD": 0.65, "NZDUSD": 0.60},
                              {"AUDUSD": 1.0, "NZDUSD": 1.0}, 100000.0, 0.10, 1.0)
    assert math.isclose(half["AUDUSD"], 0.5 * base["AUDUSD"], rel_tol=1e-9)
