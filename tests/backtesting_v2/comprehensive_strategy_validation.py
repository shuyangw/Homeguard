"""
Comprehensive Strategy Validation: V1 vs V2 Parity.

Tests all strategy types:
1. Production strategies (OMR, RAMP patterns)
2. Research strategies (Mean Reversion, Momentum, Moving Average, etc.)
3. Multi-symbol portfolios
4. Different risk configurations
"""

import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Tuple, Any

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader

# Import research strategies
from src.strategies.research.mean_reversion import MeanReversion, RSIMeanReversion
from src.strategies.research.momentum import MomentumStrategy, BreakoutStrategy
from src.strategies.research.moving_average import MovingAverageCrossover, TripleMovingAverage


def compare_results(name: str, pf_v1, pf_v2) -> Dict[str, Any]:
    """Compare V1 and V2 portfolio results."""
    s1 = pf_v1.stats()
    s2 = pf_v2.stats()

    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values)
    max_eq_diff = eq_diff.max()

    ret_diff = abs(s1["Total Return [%]"] - s2["Total Return [%]"])
    trades_match = len(pf_v1.trades) == len(pf_v2.trades)

    # Check PnL match for each trade
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


def run_strategy_test(
    name: str,
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    risk_config: RiskConfig,
    allow_shorts: bool = False,
    market_hours_only: bool = True,
    use_numba: bool = True,
) -> Dict[str, Any]:
    """Run a single strategy test comparing V1 and V2."""
    params = dict(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        freq="1min",
        market_hours_only=market_hours_only,
        risk_config=risk_config,
        price_data=data,
        allow_shorts=allow_shorts,
        use_numba=use_numba,
    )

    try:
        pf_v1 = from_signals(**params)
        pf_v2 = from_signals_v2(**params)
        return compare_results(name, pf_v1, pf_v2)
    except Exception as e:
        print(f"  [-] ERROR | {name}: {e}")
        return {"test": name, "passed": False, "error": str(e)}


def run_comprehensive_validation():
    """Run comprehensive V1 vs V2 validation across all strategy types."""
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY VALIDATION: V1 vs V2")
    print("=" * 80)

    # Load real market data for multiple symbols
    loader = StreamingDataLoader(timeframe="1min")

    results = []

    # =========================================================================
    # SECTION 1: Load SPY data for single-symbol tests
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: SINGLE-SYMBOL STRATEGIES (SPY)")
    print("=" * 80)

    print("\nLoading SPY data (2023-01-01 to 2023-06-30)...")
    spy_data = None
    for symbol, df in loader.iter_symbols(["SPY"], "2023-01-01", "2023-06-30"):
        spy_data = df.to_pandas().set_index("timestamp")

    if spy_data is None or len(spy_data) == 0:
        print("ERROR: Could not load SPY data")
        return 1

    print(f"Loaded {len(spy_data)} bars")

    # -------------------------------------------------------------------------
    # Test 1: Moving Average Crossover (Simple)
    # -------------------------------------------------------------------------
    print("\n--- Moving Average Crossover (SMA 20/50) ---")
    ma_strategy = MovingAverageCrossover(fast_window=20, slow_window=50, ma_type="sma")
    entries, exits = ma_strategy.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "MA Crossover SMA 20/50",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 2: Moving Average Crossover (EMA)
    # -------------------------------------------------------------------------
    print("\n--- Moving Average Crossover (EMA 12/26) ---")
    ma_ema = MovingAverageCrossover(fast_window=12, slow_window=26, ma_type="ema")
    entries, exits = ma_ema.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "MA Crossover EMA 12/26",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 3: Triple Moving Average
    # -------------------------------------------------------------------------
    print("\n--- Triple Moving Average (10/20/50) ---")
    triple_ma = TripleMovingAverage(fast_window=10, medium_window=20, slow_window=50)
    entries, exits = triple_ma.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "Triple MA 10/20/50",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 4: Mean Reversion (Bollinger Bands)
    # -------------------------------------------------------------------------
    print("\n--- Mean Reversion (Bollinger Bands 20, 2std) ---")
    mr_strategy = MeanReversion(window=20, num_std=2.0)
    entries, exits = mr_strategy.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "Mean Reversion BB",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 5: RSI Mean Reversion
    # -------------------------------------------------------------------------
    print("\n--- RSI Mean Reversion (14, 30/70) ---")
    rsi_mr = RSIMeanReversion(rsi_window=14, oversold=30, overbought=70)
    entries, exits = rsi_mr.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "RSI Mean Reversion",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 6: MACD Momentum
    # -------------------------------------------------------------------------
    print("\n--- MACD Momentum (12/26/9) ---")
    macd_strat = MomentumStrategy(fast=12, slow=26, signal=9)
    entries, exits = macd_strat.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "MACD Momentum",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # -------------------------------------------------------------------------
    # Test 7: Breakout Strategy
    # -------------------------------------------------------------------------
    print("\n--- Breakout Strategy (20-day high/10-day low) ---")
    breakout = BreakoutStrategy(breakout_window=20, exit_window=10)
    entries, exits = breakout.generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "Breakout 20/10",
        spy_data, entries, exits,
        RiskConfig.moderate(),
    ))

    # =========================================================================
    # SECTION 2: DIFFERENT RISK CONFIGURATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: RISK CONFIGURATION VARIANTS")
    print("=" * 80)

    # Use MA Crossover as base strategy
    ma_strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
    entries, exits = ma_strategy.generate_long_signals(spy_data)

    # Test 8: Stop Loss 2%
    print("\n--- With 2% Stop Loss ---")
    risk_sl = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        stop_loss_type="percentage",
    )
    results.append(run_strategy_test(
        "MA + 2% Stop Loss",
        spy_data, entries, exits, risk_sl,
    ))

    # Test 9: Profit Target 3%
    print("\n--- With 3% Profit Target ---")
    risk_pt = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        stop_loss_type="profit_target",
    )
    results.append(run_strategy_test(
        "MA + 3% Profit Target",
        spy_data, entries, exits, risk_pt,
    ))

    # Test 10: Time Stop 50 bars
    print("\n--- With Time Stop (50 bars) ---")
    risk_time = RiskConfig(
        enabled=True,
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_type="time",
        max_holding_bars=50,
    )
    results.append(run_strategy_test(
        "MA + Time Stop 50",
        spy_data, entries, exits, risk_time,
    ))

    # Test 11: Aggressive 25% position sizing
    print("\n--- Aggressive 25% Position Sizing ---")
    risk_agg = RiskConfig(enabled=True, position_size_pct=0.25)
    results.append(run_strategy_test(
        "MA + 25% Position",
        spy_data, entries, exits, risk_agg,
    ))

    # Test 12: Conservative 5% position sizing
    print("\n--- Conservative 5% Position Sizing ---")
    risk_con = RiskConfig(enabled=True, position_size_pct=0.05)
    results.append(run_strategy_test(
        "MA + 5% Position",
        spy_data, entries, exits, risk_con,
    ))

    # Test 13: High fees and slippage
    print("\n--- High Trading Costs ---")
    params_high_cost = dict(
        close=spy_data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.005,  # 0.5% fees
        slippage=0.002,  # 0.2% slippage
        freq="1min",
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        price_data=spy_data,
        allow_shorts=False,
        use_numba=True,
    )
    pf_v1 = from_signals(**params_high_cost)
    pf_v2 = from_signals_v2(**params_high_cost)
    results.append(compare_results("High Trading Costs", pf_v1, pf_v2))

    # =========================================================================
    # SECTION 3: SHORT SELLING TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: SHORT SELLING STRATEGIES")
    print("=" * 80)

    # Test 14: MA Crossover with shorts
    print("\n--- MA Crossover with Shorts ---")
    results.append(run_strategy_test(
        "MA + Shorts",
        spy_data, entries, exits,
        RiskConfig.moderate(),
        allow_shorts=True,
    ))

    # Test 15: Mean Reversion with shorts
    print("\n--- Mean Reversion with Shorts ---")
    mr_entries, mr_exits = MeanReversion(window=20, num_std=2.0).generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "Mean Rev + Shorts",
        spy_data, mr_entries, mr_exits,
        RiskConfig.moderate(),
        allow_shorts=True,
    ))

    # Test 16: Full combo (shorts + stop loss + profit target)
    print("\n--- Full Combo (Shorts + Stop Loss + Profit Target) ---")
    risk_combo = RiskConfig(
        enabled=True,
        position_size_pct=0.15,
        use_stop_loss=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        stop_loss_type="profit_target",
    )
    results.append(run_strategy_test(
        "Full Combo",
        spy_data, entries, exits, risk_combo,
        allow_shorts=True,
    ))

    # =========================================================================
    # SECTION 4: NUMBA vs PYTHON PARITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: NUMBA vs PYTHON ENGINE PARITY")
    print("=" * 80)

    # Test 17: Same strategy, Python engine
    print("\n--- Python Engine (Numba=False) ---")
    results.append(run_strategy_test(
        "MA Python Engine",
        spy_data, entries, exits,
        RiskConfig.moderate(),
        use_numba=False,
    ))

    # Test 18: Complex strategy, Python engine
    print("\n--- Triple MA + Stops, Python Engine ---")
    triple_entries, triple_exits = TripleMovingAverage().generate_long_signals(spy_data)
    results.append(run_strategy_test(
        "Triple MA Python",
        spy_data, triple_entries, triple_exits,
        risk_sl,
        use_numba=False,
    ))

    # =========================================================================
    # SECTION 5: MULTI-SYMBOL TESTS (QQQ, IWM)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: MULTI-SYMBOL VALIDATION")
    print("=" * 80)

    for symbol in ["QQQ", "IWM"]:
        print(f"\nLoading {symbol} data...")
        symbol_data = None
        for sym, df in loader.iter_symbols([symbol], "2023-01-01", "2023-06-30"):
            symbol_data = df.to_pandas().set_index("timestamp")

        if symbol_data is None or len(symbol_data) == 0:
            print(f"  Warning: Could not load {symbol} data, skipping")
            continue

        print(f"  Loaded {len(symbol_data)} bars")

        # Test MA Crossover on each symbol
        ma_entries, ma_exits = MovingAverageCrossover().generate_long_signals(symbol_data)
        results.append(run_strategy_test(
            f"{symbol} MA Crossover",
            symbol_data, ma_entries, ma_exits,
            RiskConfig.moderate(),
        ))

        # Test Mean Reversion on each symbol
        mr_entries, mr_exits = MeanReversion().generate_long_signals(symbol_data)
        results.append(run_strategy_test(
            f"{symbol} Mean Reversion",
            symbol_data, mr_entries, mr_exits,
            RiskConfig.moderate(),
        ))

    # =========================================================================
    # SECTION 6: EDGE CASES
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: EDGE CASES")
    print("=" * 80)

    # Test: No trades scenario
    print("\n--- No Trades Scenario ---")
    no_entries = pd.Series(False, index=spy_data.index)
    no_exits = pd.Series(False, index=spy_data.index)
    results.append(run_strategy_test(
        "No Trades",
        spy_data, no_entries, no_exits,
        RiskConfig.moderate(),
    ))

    # Test: Single trade
    print("\n--- Single Trade Scenario ---")
    single_entries = pd.Series(False, index=spy_data.index)
    single_exits = pd.Series(False, index=spy_data.index)
    single_entries.iloc[500] = True
    single_exits.iloc[-500] = True
    results.append(run_strategy_test(
        "Single Trade",
        spy_data, single_entries, single_exits,
        RiskConfig.moderate(),
    ))

    # Test: Rapid trading (many signals)
    print("\n--- Rapid Trading (100+ trades) ---")
    rapid_entries = pd.Series(False, index=spy_data.index)
    rapid_exits = pd.Series(False, index=spy_data.index)
    for i in range(500, len(spy_data) - 500, 100):
        rapid_entries.iloc[i] = True
        rapid_exits.iloc[i + 50] = True
    results.append(run_strategy_test(
        "Rapid Trading",
        spy_data, rapid_entries, rapid_exits,
        RiskConfig.moderate(),
    ))

    # Test: Market hours disabled
    print("\n--- Market Hours Disabled ---")
    results.append(run_strategy_test(
        "No Market Hours",
        spy_data, entries, exits,
        RiskConfig.moderate(),
        market_hours_only=False,
    ))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
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
        print("[+] ALL TESTS PASSED - V1 and V2 have COMPLETE PARITY!")
    else:
        print(f"[-] {failed} TEST(S) FAILED - Parity issues detected")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_validation())
