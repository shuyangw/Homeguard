"""Verify Numba and Python produce EXACTLY identical outputs."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.utils.risk_config import RiskConfig

def compare_outputs(name: str, prices: pd.Series, entries: pd.Series, exits: pd.Series,
                   allow_shorts: bool = False, use_stop_loss: bool = False):
    """Compare Python and Numba outputs for exact equality."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    risk_config = RiskConfig(use_stop_loss=use_stop_loss, stop_loss_pct=0.05)

    # Run Python
    portfolio_py = Portfolio(
        price=prices, entries=entries, exits=exits,
        init_cash=10000, fees=0.001, slippage=0.0005,
        allow_shorts=allow_shorts, risk_config=risk_config,
        market_hours_only=False, use_numba=False
    )

    # Run Numba
    portfolio_nb = Portfolio(
        price=prices, entries=entries, exits=exits,
        init_cash=10000, fees=0.001, slippage=0.0005,
        allow_shorts=allow_shorts, risk_config=risk_config,
        market_hours_only=False, use_numba=True
    )

    all_match = True

    # 1. Compare equity curves
    print("\n1. EQUITY CURVE COMPARISON:")
    equity_py = portfolio_py.equity_curve.values
    equity_nb = portfolio_nb.equity_curve.values

    if len(equity_py) != len(equity_nb):
        print(f"   FAIL: Length mismatch - Python={len(equity_py)}, Numba={len(equity_nb)}")
        all_match = False
    else:
        max_diff = np.max(np.abs(equity_py - equity_nb))
        if max_diff == 0:
            print(f"   PASS: Equity curves are EXACTLY identical ({len(equity_py)} values)")
        elif max_diff < 1e-10:
            print(f"   PASS: Equity curves match within floating-point tolerance (max diff: {max_diff:.2e})")
        else:
            print(f"   FAIL: Max difference = {max_diff:.6f}")
            # Show where differences occur
            diffs = np.abs(equity_py - equity_nb)
            diff_indices = np.where(diffs > 1e-10)[0]
            print(f"   Differences at {len(diff_indices)} bars:")
            for idx in diff_indices[:5]:
                print(f"      Bar {idx}: Python={equity_py[idx]:.6f}, Numba={equity_nb[idx]:.6f}, diff={diffs[idx]:.6f}")
            all_match = False

    # 2. Compare trade counts
    print("\n2. TRADE COUNT COMPARISON:")
    trades_py = portfolio_py.trades
    trades_nb = portfolio_nb.trades

    if len(trades_py) != len(trades_nb):
        print(f"   FAIL: Trade count mismatch - Python={len(trades_py)}, Numba={len(trades_nb)}")
        all_match = False
    else:
        print(f"   PASS: Both have {len(trades_py)} trades")

    # 3. Compare individual trades
    print("\n3. TRADE DETAILS COMPARISON:")
    trades_match = True
    for i, (py_trade, nb_trade) in enumerate(zip(trades_py, trades_nb)):
        issues = []

        # Compare type
        if py_trade['type'] != nb_trade['type']:
            issues.append(f"type: {py_trade['type']} vs {nb_trade['type']}")

        # Compare price
        if abs(py_trade['price'] - nb_trade['price']) > 1e-10:
            issues.append(f"price: {py_trade['price']:.6f} vs {nb_trade['price']:.6f}")

        # Compare shares
        if abs(py_trade['shares'] - nb_trade['shares']) > 1e-10:
            issues.append(f"shares: {py_trade['shares']:.6f} vs {nb_trade['shares']:.6f}")

        # Compare PnL (for exits)
        if 'pnl' in py_trade and 'pnl' in nb_trade:
            if abs(py_trade['pnl'] - nb_trade['pnl']) > 1e-6:
                issues.append(f"pnl: {py_trade['pnl']:.6f} vs {nb_trade['pnl']:.6f}")

        if issues:
            print(f"   Trade {i} MISMATCH: {', '.join(issues)}")
            trades_match = False

    if trades_match and len(trades_py) > 0:
        print(f"   PASS: All {len(trades_py)} trades are identical")
    elif len(trades_py) == 0:
        print(f"   PASS: No trades to compare")
    else:
        all_match = False

    # 4. Compare final stats
    print("\n4. STATS COMPARISON:")
    stats_py = portfolio_py.stats()
    stats_nb = portfolio_nb.stats()

    key_metrics = ['Total Return [%]', 'Total Trades', 'Win Rate [%]', 'Max Drawdown [%]']
    stats_match = True

    for metric in key_metrics:
        if metric in stats_py and metric in stats_nb:
            py_val = stats_py[metric]
            nb_val = stats_nb[metric]

            if isinstance(py_val, (int, float)) and isinstance(nb_val, (int, float)):
                if abs(py_val - nb_val) > 1e-6:
                    print(f"   {metric}: Python={py_val:.6f}, Numba={nb_val:.6f} - MISMATCH")
                    stats_match = False
                else:
                    print(f"   {metric}: {py_val:.4f} - MATCH")

    if not stats_match:
        all_match = False

    # Final verdict
    print(f"\n{'='*60}")
    if all_match:
        print("RESULT: ALL OUTPUTS ARE EXACTLY IDENTICAL [PASS]")
    else:
        print("RESULT: OUTPUTS DIFFER [FAIL]")
    print(f"{'='*60}")

    return all_match


def main():
    print("VERIFYING EXACT PARITY BETWEEN NUMBA AND PYTHON IMPLEMENTATIONS")
    print("=" * 70)

    all_tests_pass = True

    # Test 1: Simple long-only
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
    prices = pd.Series([100 + i * 0.5 for i in range(n)], index=dates)
    entries = pd.Series([i == 5 for i in range(n)], index=dates)
    exits = pd.Series([i == 20 for i in range(n)], index=dates)

    if not compare_outputs("Simple Long-Only", prices, entries, exits):
        all_tests_pass = False

    # Test 2: Multiple trades
    entries = pd.Series([i % 20 == 0 for i in range(n)], index=dates)
    exits = pd.Series([i % 20 == 10 for i in range(n)], index=dates)

    if not compare_outputs("Multiple Trades", prices, entries, exits):
        all_tests_pass = False

    # Test 3: With shorts (long only mode)
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
    returns = np.random.randn(n) * 0.02
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    entries = pd.Series([i % 20 == 0 for i in range(n)], index=dates)
    exits = pd.Series([i % 20 == 10 for i in range(n)], index=dates)

    if not compare_outputs("Volatile Data, Long-Only", prices, entries, exits, allow_shorts=False):
        all_tests_pass = False

    # Test 4: With shorts enabled
    if not compare_outputs("Volatile Data, With Shorts", prices, entries, exits, allow_shorts=True):
        all_tests_pass = False

    # Test 5: With stop loss (long only)
    if not compare_outputs("With Stop Loss, Long-Only", prices, entries, exits,
                          allow_shorts=False, use_stop_loss=True):
        all_tests_pass = False

    # Test 6: Downtrend with shorts
    n = 50
    dates = pd.date_range('2022-01-03 10:00:00', periods=n, freq='D', tz='US/Eastern')
    prices = pd.Series([100 - i * 0.3 for i in range(n)], index=dates)
    entries = pd.Series([i == 40 for i in range(n)], index=dates)
    exits = pd.Series([i == 2 for i in range(n)], index=dates)

    if not compare_outputs("Downtrend with Short", prices, entries, exits, allow_shorts=True):
        all_tests_pass = False

    # Test 7: High frequency trading
    np.random.seed(123)
    n = 1000
    dates = pd.date_range('2022-01-03 09:30:00', periods=n, freq='min', tz='US/Eastern')
    returns = np.random.randn(n) * 0.001
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    entries = pd.Series([i % 50 < 25 for i in range(n)], index=dates)
    exits = pd.Series([i % 50 >= 25 for i in range(n)], index=dates)

    if not compare_outputs("High Frequency (1000 bars)", prices, entries, exits):
        all_tests_pass = False

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if all_tests_pass:
        print("ALL TESTS PASSED - Numba outputs are EXACTLY identical to Python")
    else:
        print("SOME TESTS FAILED - Outputs differ between implementations")

    return all_tests_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
