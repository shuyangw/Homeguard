"""
Comprehensive V1 vs V2 Parity Validation.

Tests all features across both frameworks to ensure complete parity.
"""

import pandas as pd
import numpy as np
import sys

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader


def generate_ma_signals(data, short=5, long=20):
    """MA crossover signals."""
    ma_s = data["close"].rolling(short).mean()
    ma_l = data["close"].rolling(long).mean()
    entries = ((ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))).fillna(False)
    exits = ((ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))).fillna(False)
    return entries, exits


def generate_momentum_signals(data, lookback=20, threshold=0.01):
    """Momentum signals."""
    ret = data["close"].pct_change(lookback)
    entries = (ret > threshold).fillna(False)
    exits = (ret < -threshold / 2).fillna(False)
    return entries, exits


def generate_no_trade_signals(data):
    """No trades."""
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    return entries, exits


def generate_single_trade_signals(data):
    """Single entry/exit."""
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    entries.iloc[100] = True
    exits.iloc[-100] = True
    return entries, exits


def generate_rapid_signals(data):
    """Many rapid trades."""
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    for i in range(200, len(data) - 200, 50):
        entries.iloc[i] = True
        exits.iloc[i + 25] = True
    return entries, exits


def compare_portfolios(name, pf_v1, pf_v2):
    """Compare two portfolios and return parity status."""
    s1 = pf_v1.stats()
    s2 = pf_v2.stats()

    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values)
    max_eq_diff = eq_diff.max()

    ret_diff = abs(s1["Total Return [%]"] - s2["Total Return [%]"])
    trades_diff = abs(s1["Total Trades"] - s2["Total Trades"])

    trades_match = len(pf_v1.trades) == len(pf_v2.trades)
    pnl_match = True
    if trades_match and len(pf_v1.trades) > 0:
        for t1, t2 in zip(pf_v1.trades, pf_v2.trades):
            if "pnl" in t1 and "pnl" in t2:
                if abs(t1["pnl"] - t2["pnl"]) > 0.01:
                    pnl_match = False
                    break

    passed = (
        max_eq_diff < 0.01
        and ret_diff < 0.0001
        and trades_diff == 0
        and trades_match
        and pnl_match
    )

    status = "[+] PASS" if passed else "[-] FAIL"
    print(
        f"  {status} | Eq diff: ${max_eq_diff:.4f} | "
        f"Trades: {len(pf_v1.trades)}/{len(pf_v2.trades)} | "
        f"Final: ${pf_v1.equity_curve.iloc[-1]:,.2f}/${pf_v2.equity_curve.iloc[-1]:,.2f}"
    )

    return {
        "test": name,
        "passed": passed,
        "max_equity_diff": max_eq_diff,
        "return_diff": ret_diff,
        "trades_v1": len(pf_v1.trades),
        "trades_v2": len(pf_v2.trades),
        "final_v1": pf_v1.equity_curve.iloc[-1],
        "final_v2": pf_v2.equity_curve.iloc[-1],
    }


def run_parity_tests():
    """Run all parity tests."""
    print("=" * 80)
    print("COMPREHENSIVE V1 vs V2 PARITY VALIDATION")
    print("=" * 80)

    # Load real market data
    loader = StreamingDataLoader(timeframe="1min")
    print("\nLoading test data (SPY 2023-05-01 to 2023-05-15)...")

    market_data = None
    for symbol, df in loader.iter_symbols(["SPY"], "2023-05-01", "2023-05-15"):
        market_data = df.to_pandas().set_index("timestamp")

    if market_data is None:
        print("ERROR: Could not load data")
        return 1

    print(f"Loaded {len(market_data)} bars")

    results = []

    def run_test(name, entries, exits, **kwargs):
        print(f"\n{name}...")

        base_params = dict(
            close=market_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            freq="1min",
            price_data=market_data,
        )
        base_params.update(kwargs)

        try:
            pf_v1 = from_signals(**base_params)
            pf_v2 = from_signals_v2(**base_params)
            result = compare_portfolios(name, pf_v1, pf_v2)
            results.append(result)
        except Exception as e:
            print(f"  [-] ERROR: {e}")
            results.append({"test": name, "passed": False, "error": str(e)})

    # Generate base signals
    entries, exits = generate_ma_signals(market_data)

    # TEST 1: Basic MA Crossover
    print("\n" + "=" * 80)
    print("TEST 1: Basic MA Crossover (Long Only, No Stops)")
    print("=" * 80)
    run_test(
        "Basic MA Crossover",
        entries,
        exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 2: Shorts Enabled
    print("\n" + "=" * 80)
    print("TEST 2: With Short Selling Enabled")
    print("=" * 80)
    run_test(
        "Shorts Enabled",
        entries,
        exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=True,
        use_numba=True,
    )

    # TEST 3: Stop Loss
    print("\n" + "=" * 80)
    print("TEST 3: Percentage Stop Loss")
    print("=" * 80)
    risk_sl = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        stop_loss_type="percentage",
    )
    run_test(
        "Stop Loss 2%",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_sl,
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 4: Profit Target
    print("\n" + "=" * 80)
    print("TEST 4: Profit Target")
    print("=" * 80)
    risk_pt = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        stop_loss_type="profit_target",
    )
    run_test(
        "Profit Target 3%",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_pt,
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 5: Time Stop
    print("\n" + "=" * 80)
    print("TEST 5: Time-Based Stop")
    print("=" * 80)
    risk_time = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_type="time",
        max_holding_bars=100,
    )
    run_test(
        "Time Stop 100 bars",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_time,
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 6: No Market Hours
    print("\n" + "=" * 80)
    print("TEST 6: Market Hours Disabled")
    print("=" * 80)
    run_test(
        "No Market Hours Filter",
        entries,
        exits,
        market_hours_only=False,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 7: Pure Python
    print("\n" + "=" * 80)
    print("TEST 7: Pure Python (Numba Disabled)")
    print("=" * 80)
    run_test(
        "Pure Python",
        entries,
        exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=False,
    )

    # TEST 8: Aggressive Sizing
    print("\n" + "=" * 80)
    print("TEST 8: Aggressive Position Sizing (25%)")
    print("=" * 80)
    risk_agg = RiskConfig(enabled=True, position_size_pct=0.25)
    run_test(
        "25% Position Size",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_agg,
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 9: Conservative Sizing
    print("\n" + "=" * 80)
    print("TEST 9: Conservative Position Sizing (5%)")
    print("=" * 80)
    risk_con = RiskConfig(enabled=True, position_size_pct=0.05)
    run_test(
        "5% Position Size",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_con,
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 10: No Trades
    print("\n" + "=" * 80)
    print("TEST 10: No Trades Scenario")
    print("=" * 80)
    no_entries, no_exits = generate_no_trade_signals(market_data)
    run_test(
        "No Trades",
        no_entries,
        no_exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 11: Single Trade
    print("\n" + "=" * 80)
    print("TEST 11: Single Trade Scenario")
    print("=" * 80)
    single_entries, single_exits = generate_single_trade_signals(market_data)
    run_test(
        "Single Trade",
        single_entries,
        single_exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 12: Rapid Trading
    print("\n" + "=" * 80)
    print("TEST 12: Rapid Trading (Many Trades)")
    print("=" * 80)
    rapid_entries, rapid_exits = generate_rapid_signals(market_data)
    run_test(
        "Rapid Trading",
        rapid_entries,
        rapid_exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 13: Momentum
    print("\n" + "=" * 80)
    print("TEST 13: Momentum Strategy")
    print("=" * 80)
    mom_entries, mom_exits = generate_momentum_signals(market_data)
    run_test(
        "Momentum Strategy",
        mom_entries,
        mom_exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
    )

    # TEST 14: Full Combo
    print("\n" + "=" * 80)
    print("TEST 14: Shorts + Stop Loss + Profit Target")
    print("=" * 80)
    risk_combo = RiskConfig(
        enabled=True,
        position_size_pct=0.15,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_type="profit_target",
    )
    run_test(
        "Full Features Combo",
        entries,
        exits,
        market_hours_only=True,
        risk_config=risk_combo,
        allow_shorts=True,
        use_numba=True,
    )

    # TEST 15: High Costs
    print("\n" + "=" * 80)
    print("TEST 15: High Slippage + High Fees")
    print("=" * 80)
    run_test(
        "High Costs",
        entries,
        exits,
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        allow_shorts=False,
        use_numba=True,
        fees=0.005,
        slippage=0.002,
    )

    # SUMMARY
    print("\n" + "=" * 80)
    print("PARITY VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    print("\nDetailed Results:")
    print("-" * 80)
    for r in results:
        status = "[+] PASS" if r.get("passed", False) else "[-] FAIL"
        test_name = r["test"]
        if "error" in r:
            print(f"{status} {test_name}: ERROR - {r['error']}")
        else:
            eq_diff = r.get("max_equity_diff", 0)
            t1 = r.get("trades_v1", "?")
            t2 = r.get("trades_v2", "?")
            print(f"{status} {test_name}: Eq diff=${eq_diff:.6f}, Trades={t1}/{t2}")

    print("\n" + "=" * 80)
    if failed == 0:
        print("[+] ALL TESTS PASSED - V1 and V2 have COMPLETE PARITY!")
    else:
        print(f"[-] {failed} TEST(S) FAILED - Parity issues detected")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_parity_tests())
