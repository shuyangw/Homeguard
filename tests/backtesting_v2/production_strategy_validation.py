"""
Production Strategy Validation: OMR and RAMP pattern tests.

Tests complex patterns used in production strategies:
1. OMR-style: Overnight positions, leveraged ETFs, Bayesian-like signals
2. RAMP-style: Multi-symbol momentum, regime-aware signals
"""

import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Tuple, Any

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader


def compare_results(name: str, pf_v1, pf_v2) -> Dict[str, Any]:
    """Compare V1 and V2 portfolio results."""
    s1 = pf_v1.stats()
    s2 = pf_v2.stats()

    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values)
    max_eq_diff = eq_diff.max()

    ret_diff = abs(s1["Total Return [%]"] - s2["Total Return [%]"])
    trades_match = len(pf_v1.trades) == len(pf_v2.trades)

    pnl_match = True
    if trades_match and len(pf_v1.trades) > 0:
        for t1, t2 in zip(pf_v1.trades, pf_v2.trades):
            if "pnl" in t1 and "pnl" in t2:
                if abs(t1["pnl"] - t2["pnl"]) > 0.01:
                    pnl_match = False
                    break

    passed = max_eq_diff < 0.01 and ret_diff < 0.0001 and trades_match and pnl_match

    status = "[+] PASS" if passed else "[-] FAIL"
    print(
        f"  {status} | {name}: Eq diff=${max_eq_diff:.4f}, "
        f"Trades={len(pf_v1.trades)}/{len(pf_v2.trades)}, "
        f"Final=${pf_v1.equity_curve.iloc[-1]:,.2f}/${pf_v2.equity_curve.iloc[-1]:,.2f}"
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


def generate_omr_style_signals(data: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Generate OMR-style signals (overnight mean reversion pattern).

    Entry: Strong intraday down move (potential overnight reversion)
    Exit: Next day open signal
    """
    close = data["close"]
    open_price = data["open"]
    high = data["high"]
    low = data["low"]

    # Calculate intraday return
    intraday_ret = (close - open_price) / open_price

    # Calculate z-score of intraday return
    intraday_mean = intraday_ret.rolling(lookback).mean()
    intraday_std = intraday_ret.rolling(lookback).std()
    z_score = (intraday_ret - intraday_mean) / intraday_std.replace(0, np.nan)

    # Entry on strong down day (z-score < -1.5)
    entries = z_score < -1.5

    # Exit on next bar (overnight hold pattern)
    exits = entries.shift(1).fillna(False)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def generate_momentum_style_signals(data: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Generate RAMP-style momentum signals.

    Entry: Price momentum positive over lookback period
    Exit: Momentum turns negative
    """
    close = data["close"]

    # Calculate momentum
    momentum = close.pct_change(lookback)

    # Entry when momentum crosses above 0
    entries = (momentum > 0) & (momentum.shift(1) <= 0)

    # Exit when momentum crosses below 0
    exits = (momentum < 0) & (momentum.shift(1) >= 0)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def generate_regime_aware_signals(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Generate regime-aware signals (RAMP-style).

    Uses volatility regime to adjust signal generation.
    """
    close = data["close"]

    # Calculate volatility regime
    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_avg = vol.rolling(60).mean()

    # Low volatility regime: momentum works better
    low_vol_regime = vol < vol_avg

    # Momentum signal
    momentum = close.pct_change(10)

    # Entry: momentum positive in low vol, or strong momentum in high vol
    entry_low_vol = low_vol_regime & (momentum > 0.01)
    entry_high_vol = ~low_vol_regime & (momentum > 0.03)
    entries = (entry_low_vol | entry_high_vol) & ~(entry_low_vol.shift(1) | entry_high_vol.shift(1))

    # Exit: momentum turns negative
    exits = (momentum < -0.01) & (momentum.shift(1) >= -0.01)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    return entries, exits


def run_production_validation():
    """Run OMR and RAMP production strategy pattern validation."""
    print("=" * 80)
    print("PRODUCTION STRATEGY VALIDATION: OMR & RAMP PATTERNS")
    print("=" * 80)

    loader = StreamingDataLoader(timeframe="1min")
    results = []

    # =========================================================================
    # SECTION 1: OMR-STYLE PATTERNS (Leveraged ETFs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: OMR-STYLE PATTERNS (Leveraged ETFs)")
    print("=" * 80)

    # Test with leveraged ETF symbols used by OMR
    omr_symbols = ["TQQQ", "SQQQ", "UPRO", "SPXU"]

    for symbol in omr_symbols:
        print(f"\nLoading {symbol} data (2023-01-01 to 2023-06-30)...")
        symbol_data = None
        try:
            for sym, df in loader.iter_symbols([symbol], "2023-01-01", "2023-06-30"):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Warning: Could not load {symbol}: {e}")
            continue

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Warning: Insufficient data for {symbol}, skipping")
            continue

        print(f"  Loaded {len(symbol_data)} bars")

        # OMR-style signals
        entries, exits = generate_omr_style_signals(symbol_data)

        # Test with various risk configs
        for risk_name, risk_config in [
            ("Moderate", RiskConfig.moderate()),
            ("2% Stop Loss", RiskConfig(
                enabled=True, position_size_pct=0.10,
                use_stop_loss=True, stop_loss_pct=0.02, stop_loss_type="percentage"
            )),
            ("3% Profit Target", RiskConfig(
                enabled=True, position_size_pct=0.10,
                use_stop_loss=True, stop_loss_pct=0.02,
                take_profit_pct=0.03, stop_loss_type="profit_target"
            )),
        ]:
            params = dict(
                close=symbol_data["close"],
                entries=entries,
                exits=exits,
                init_cash=100000,
                fees=0.001,
                slippage=0.0005,
                freq="1min",
                market_hours_only=True,
                risk_config=risk_config,
                price_data=symbol_data,
                allow_shorts=False,
                use_numba=True,
            )

            try:
                pf_v1 = from_signals(**params)
                pf_v2 = from_signals_v2(**params)
                results.append(compare_results(f"{symbol} OMR {risk_name}", pf_v1, pf_v2))
            except Exception as e:
                print(f"  [-] ERROR | {symbol} OMR {risk_name}: {e}")
                results.append({"test": f"{symbol} OMR {risk_name}", "passed": False, "error": str(e)})

    # =========================================================================
    # SECTION 2: RAMP-STYLE PATTERNS (Momentum + Regime)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: RAMP-STYLE PATTERNS (Momentum + Regime)")
    print("=" * 80)

    # Test with S&P 500 component symbols
    ramp_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    for symbol in ramp_symbols:
        print(f"\nLoading {symbol} data (2023-01-01 to 2023-06-30)...")
        symbol_data = None
        try:
            for sym, df in loader.iter_symbols([symbol], "2023-01-01", "2023-06-30"):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Warning: Could not load {symbol}: {e}")
            continue

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Warning: Insufficient data for {symbol}, skipping")
            continue

        print(f"  Loaded {len(symbol_data)} bars")

        # Momentum-style signals
        entries, exits = generate_momentum_style_signals(symbol_data)

        params = dict(
            close=symbol_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            freq="1min",
            market_hours_only=True,
            risk_config=RiskConfig.moderate(),
            price_data=symbol_data,
            allow_shorts=False,
            use_numba=True,
        )

        try:
            pf_v1 = from_signals(**params)
            pf_v2 = from_signals_v2(**params)
            results.append(compare_results(f"{symbol} Momentum", pf_v1, pf_v2))
        except Exception as e:
            print(f"  [-] ERROR | {symbol} Momentum: {e}")
            results.append({"test": f"{symbol} Momentum", "passed": False, "error": str(e)})

        # Regime-aware signals
        entries, exits = generate_regime_aware_signals(symbol_data)

        params["entries"] = entries
        params["exits"] = exits

        try:
            pf_v1 = from_signals(**params)
            pf_v2 = from_signals_v2(**params)
            results.append(compare_results(f"{symbol} Regime-Aware", pf_v1, pf_v2))
        except Exception as e:
            print(f"  [-] ERROR | {symbol} Regime-Aware: {e}")
            results.append({"test": f"{symbol} Regime-Aware", "passed": False, "error": str(e)})

    # =========================================================================
    # SECTION 3: COMPLEX COMBINATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: COMPLEX STRATEGY COMBINATIONS")
    print("=" * 80)

    # Load SPY for complex tests
    print("\nLoading SPY data...")
    spy_data = None
    for symbol, df in loader.iter_symbols(["SPY"], "2023-01-01", "2023-06-30"):
        spy_data = df.to_pandas().set_index("timestamp")

    if spy_data is not None and len(spy_data) > 0:
        print(f"  Loaded {len(spy_data)} bars")

        # Test 1: OMR signals with shorts enabled
        print("\n--- OMR Pattern + Shorts ---")
        entries, exits = generate_omr_style_signals(spy_data)
        params = dict(
            close=spy_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            freq="1min",
            market_hours_only=True,
            risk_config=RiskConfig.moderate(),
            price_data=spy_data,
            allow_shorts=True,
            use_numba=True,
        )
        pf_v1 = from_signals(**params)
        pf_v2 = from_signals_v2(**params)
        results.append(compare_results("SPY OMR + Shorts", pf_v1, pf_v2))

        # Test 2: Momentum with aggressive sizing
        print("\n--- Momentum + 25% Position ---")
        entries, exits = generate_momentum_style_signals(spy_data)
        params["entries"] = entries
        params["exits"] = exits
        params["allow_shorts"] = False
        params["risk_config"] = RiskConfig(enabled=True, position_size_pct=0.25)
        pf_v1 = from_signals(**params)
        pf_v2 = from_signals_v2(**params)
        results.append(compare_results("SPY Momentum + 25%", pf_v1, pf_v2))

        # Test 3: Regime-aware with full combo
        print("\n--- Regime-Aware + Full Combo ---")
        entries, exits = generate_regime_aware_signals(spy_data)
        params["entries"] = entries
        params["exits"] = exits
        params["allow_shorts"] = True
        params["risk_config"] = RiskConfig(
            enabled=True,
            position_size_pct=0.15,
            use_stop_loss=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            stop_loss_type="profit_target",
        )
        pf_v1 = from_signals(**params)
        pf_v2 = from_signals_v2(**params)
        results.append(compare_results("SPY Regime + Full Combo", pf_v1, pf_v2))

        # Test 4: Python engine parity
        print("\n--- Python Engine Parity ---")
        params["use_numba"] = False
        params["allow_shorts"] = False
        params["risk_config"] = RiskConfig.moderate()
        entries, exits = generate_omr_style_signals(spy_data)
        params["entries"] = entries
        params["exits"] = exits
        pf_v1 = from_signals(**params)
        pf_v2 = from_signals_v2(**params)
        results.append(compare_results("SPY Python Engine", pf_v1, pf_v2))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PRODUCTION STRATEGY VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed Tests:")
        print("-" * 80)
        for r in results:
            if not r.get("passed", False):
                test_name = r["test"]
                if "error" in r:
                    print(f"  [-] {test_name}: ERROR - {r['error']}")
                else:
                    eq_diff = r.get("max_equity_diff", "?")
                    t1 = r.get("trades_v1", "?")
                    t2 = r.get("trades_v2", "?")
                    print(f"  [-] {test_name}: Eq diff=${eq_diff}, Trades={t1}/{t2}")

    print("\n" + "=" * 80)
    if failed == 0:
        print("[+] ALL PRODUCTION TESTS PASSED - V1 and V2 have COMPLETE PARITY!")
    else:
        print(f"[-] {failed} PRODUCTION TEST(S) FAILED - Parity issues detected")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_production_validation())
