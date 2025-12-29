"""Additional parity tests for V2 framework."""

import pandas as pd
import numpy as np

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig


def run_additional_tests():
    print("=" * 80)
    print("ADDITIONAL PARITY TESTS: Fractional Shares & V2 Features")
    print("=" * 80)

    # Create synthetic data (like crypto)
    print("\nCreating synthetic high-value asset data (like BTC)...")
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=5000, freq="1min")
    prices = 30000 + np.cumsum(np.random.randn(5000) * 50)

    data = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 10,
            "low": prices - 10,
            "close": prices,
            "volume": np.random.randint(100, 1000, 5000),
        },
        index=dates,
    )

    # Generate signals
    ma_s = data["close"].rolling(10).mean()
    ma_l = data["close"].rolling(30).mean()
    entries = ((ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))).fillna(False)
    exits = ((ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))).fillna(False)

    print(f"Data points: {len(data)}")
    print(f"Entry signals: {entries.sum()}, Exit signals: {exits.sum()}")

    # Test 16: Fractional shares
    print("\n" + "=" * 80)
    print("TEST 16: Fractional Shares (Crypto Mode)")
    print("=" * 80)

    params = dict(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        freq="1min",
        market_hours_only=False,
        risk_config=RiskConfig.moderate(),
        price_data=data,
        allow_shorts=False,
        use_numba=True,
        fractional_shares=True,
    )

    pf_v1 = from_signals(**params)
    pf_v2 = from_signals_v2(**params)

    s1 = pf_v1.stats()
    s2 = pf_v2.stats()

    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values).max()
    final_v1 = pf_v1.equity_curve.iloc[-1]
    final_v2 = pf_v2.equity_curve.iloc[-1]

    print(f"V1: Return={s1['Total Return [%]']:.4f}%, Trades={s1['Total Trades']}, Final=${final_v1:,.2f}")
    print(f"V2: Return={s2['Total Return [%]']:.4f}%, Trades={s2['Total Trades']}, Final=${final_v2:,.2f}")
    print(f"Max equity diff: ${eq_diff:.6f}")

    test16_pass = eq_diff < 0.01
    if test16_pass:
        print("[+] PASS - Fractional shares parity confirmed")
    else:
        print("[-] FAIL - Parity issue detected")

    # Test 17: Portfolio state integrity
    print("\n" + "=" * 80)
    print("TEST 17: V2 Portfolio State Integrity")
    print("=" * 80)

    state = pf_v2.portfolio_state
    print(f"State rows: {len(state)}")
    print(f"State columns: {list(state.columns)}")

    print("\nVerifying state consistency...")
    issues = []

    # Check equity matches
    equity_match = np.allclose(state["total_equity"].values, pf_v2.equity_curve.values)
    print(f"  Equity matches equity_curve: {equity_match}")
    if not equity_match:
        issues.append("Equity mismatch")

    # Check cash + position_value = total_equity
    pos_mask = state["position_shares"] > 0
    if pos_mask.sum() > 0:
        expected_equity = state.loc[pos_mask, "cash"] + state.loc[pos_mask, "position_value"]
        actual_equity = state.loc[pos_mask, "total_equity"]
        value_match = np.allclose(expected_equity.values, actual_equity.values, rtol=0.001)
        print(f"  cash + position_value = total_equity: {value_match}")
        if not value_match:
            issues.append("Value calculation mismatch")

        # Check exposure
        expected_exp = state.loc[pos_mask, "position_value"] / state.loc[pos_mask, "total_equity"]
        actual_exp = state.loc[pos_mask, "exposure_pct"]
        exp_match = np.allclose(expected_exp.values, actual_exp.values, rtol=0.001)
        print(f"  exposure_pct calculation correct: {exp_match}")
        if not exp_match:
            issues.append("Exposure calculation mismatch")

    test17_pass = not issues
    if test17_pass:
        print("[+] PASS - Portfolio state integrity verified")
    else:
        print(f"[-] FAIL - Issues: {issues}")

    # Test 18: Trades DataFrame completeness
    print("\n" + "=" * 80)
    print("TEST 18: V2 Trades DataFrame Completeness")
    print("=" * 80)

    trades = pf_v2.trades_df
    print(f"Trades count: {len(trades)}")
    print(f"Columns: {list(trades.columns)}")

    required_cols = ["timestamp", "type", "price", "shares"]
    missing = [c for c in required_cols if c not in trades.columns]

    test18_pass = not missing
    if test18_pass:
        print("[+] PASS - All required columns present")
    else:
        print(f"[-] FAIL - Missing columns: {missing}")

    # Test 19: Numba vs Python parity within V2
    print("\n" + "=" * 80)
    print("TEST 19: V2 Numba vs Python Internal Parity")
    print("=" * 80)

    pf_v2_numba = from_signals_v2(**{**params, "use_numba": True})
    pf_v2_python = from_signals_v2(**{**params, "use_numba": False})

    eq_diff_internal = np.abs(
        pf_v2_numba.equity_curve.values - pf_v2_python.equity_curve.values
    ).max()
    trades_match = len(pf_v2_numba.trades) == len(pf_v2_python.trades)
    state_match = len(pf_v2_numba.portfolio_state) == len(pf_v2_python.portfolio_state)

    print(f"Numba final: ${pf_v2_numba.equity_curve.iloc[-1]:,.2f}")
    print(f"Python final: ${pf_v2_python.equity_curve.iloc[-1]:,.2f}")
    print(f"Max equity diff: ${eq_diff_internal:.6f}")
    print(f"Trades match: {trades_match} ({len(pf_v2_numba.trades)} vs {len(pf_v2_python.trades)})")
    print(f"State rows match: {state_match}")

    test19_pass = eq_diff_internal < 1.0 and trades_match and state_match
    if test19_pass:
        print("[+] PASS - V2 Numba/Python internal parity")
    else:
        print("[-] FAIL - Internal parity issue")

    # Summary
    print("\n" + "=" * 80)
    print("ADDITIONAL TESTS SUMMARY")
    print("=" * 80)

    all_passed = test16_pass and test17_pass and test18_pass and test19_pass

    print(f"\nTest 16 (Fractional Shares): {'PASS' if test16_pass else 'FAIL'}")
    print(f"Test 17 (State Integrity): {'PASS' if test17_pass else 'FAIL'}")
    print(f"Test 18 (Trades Completeness): {'PASS' if test18_pass else 'FAIL'}")
    print(f"Test 19 (Numba/Python Parity): {'PASS' if test19_pass else 'FAIL'}")

    if all_passed:
        print("\n[+] ALL ADDITIONAL TESTS PASSED!")
        return 0
    else:
        print("\n[-] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_additional_tests())
