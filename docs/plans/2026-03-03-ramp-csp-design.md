# RAMP-CSP: Cash-Secured Puts on Momentum Names -- Design Document

**Date**: 2026-03-03
**Status**: Approved
**Scope**: Phases 1-4 (backtest only, no live adapter)

---

## Summary

RAMP-CSP sells cash-secured puts on high-momentum stocks during STRONG_BULL regime,
combining the volatility risk premium with RAMP's cross-sectional momentum edge.

This design covers the backtesting engine, data loader, and walk-forward validation.
Live trading adapter is out of scope for this iteration.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Universe | Restrict to ~12 available stocks + ETFs in E:\OptionsData | Most liquid names, likely in RAMP top_n frequently |
| Data granularity | Loader supports intraday; CSP uses EOD snapshots | Flexibility for future strategies |
| Module location | `src/strategies/options/csp/` | Clean namespace, room for future options strategies |
| Architecture | Hybrid: custom event-driven engine + shared reporting | Options semantics need custom engine; QuantStats reporting stays consistent |
| Data source | `E:\OptionsData\options_combined\` (Hive-partitioned 1-min parquet) | Existing data with Greeks, deep history (2012-2026) |

---

## Data Layer

### Source Format

Location: `E:\OptionsData\options_combined\root={SYMBOL}\year={YYYY}\month={MM}\data.parquet`

Available symbols (31):
- **Stocks (~12)**: AAPL, AMD, AMZN, AVGO, COIN, FB/META, GOOGL, MSFT, MSTR, NVDA, PLTR, TSLA
- **ETFs (~19)**: DIA, EEM, FXI, GLD, IBIT, IWM, QQQ, SLV, SMH, SPX, SPY, TLT, VIX, XLE, XLF, XLI, XLK, XLV

### Schema Mapping

| Source field | Internal field | Transform |
|-------------|---------------|-----------|
| `timestamp` (str) | `datetime` | pd.to_datetime() |
| `expiration` (str) | `expiry` (date) | pd.to_datetime().date() |
| `right` ("PUT"/"CALL") | `option_type` ("P"/"C") | Map first char |
| `strike` (float) | `strike` (float) | Direct |
| `bid_close`/`ask_close` | `bid`/`ask` | Direct |
| `delta` (float) | `delta` (float) | Direct |
| `gamma_eod` | `gamma` | Direct |
| `theta` (float) | `theta` (float) | Direct |
| `vega` (float) | `vega` (float) | Direct |
| `implied_vol` (float) | `implied_vol` (float) | Direct |
| `open_interest_eod` (int) | `open_interest` (int) | Direct |
| `underlying_px` (float) | `underlying_price` (float) | Direct |
| `volume` (int) | `volume` (int) | Direct |

### OptionsDataLoader

Location: `src/strategies/options/data_loader.py`

Key methods:
- `load_chains(symbol, start_date, end_date)` -> full DataFrame for date range
- `get_chain_at_time(symbol, date, time)` -> chain at specific timestamp
- `get_eod_chain(symbol, date)` -> convenience for 16:00 snapshot
- `get_available_symbols()` -> list root= directories
- `get_date_range(symbol)` -> earliest/latest dates

---

## Contract Selection

Location: `src/strategies/options/csp/contract_selector.py`

`CSPContractSelector` -- pure function, stateless.

Filter pipeline:
1. option_type == 'P' (puts only)
2. min_dte <= days_to_expiry <= max_dte (21-35 default)
3. target_delta_min <= delta <= target_delta_max (-0.35 to -0.25 default)
4. open_interest >= min_oi (100 default)
5. spread_pct <= max_spread (15% default)
6. Rank by mid_price descending (maximize premium)
7. Return top candidate or None

---

## Position Tracking

Location: `src/strategies/options/csp/position.py`

### CSPPosition (open positions)

Fields: symbol, strike, expiry, entry_date, entry_price, num_contracts, collateral
Daily update: current_price, current_delta, current_dte
Properties: premium_collected, unrealized_pnl, pnl_pct_of_premium

### CSPTrade (closed trades)

Fields: symbol, strike, expiry, entry/exit dates/prices, num_contracts,
exit_reason, regime_at_entry/exit, momentum_rank_at_entry
Properties: realized_pnl, holding_days, return_on_collateral

---

## Mark-to-Market

Location: `src/strategies/options/csp/mark_to_market.py`

Matches open positions to current chain by (expiry, strike, type='P').
When contract is missing from chain (data gap), uses Black-Scholes estimate
with last known IV as fallback.

---

## Backtest Engine

Location: `src/strategies/options/csp/engine.py`

Event-driven, day-by-day iteration:

```
For each trading day:
  1. REGIME: MarketRegimeDetector.detect_regime(spy, vix)
     + crash protection check (VIX > 25, SPY DD > 5%)

  2. MANAGE POSITIONS: For each open CSP position:
     - MTM via chain lookup (match expiry/strike/type)
     - Check exits: profit target (50%), loss limit (200%),
       DTE <= 5, regime change, crash protection, stock left top_n
     - If exit: buy to close at ask (conservative fill)

  3. NEW ENTRIES (STRONG_BULL only):
     - Get RAMP top_n momentum ranking
     - Intersect with available options symbols
     - For each candidate (up to max_positions - current):
       * Load EOD chain, run CSPContractSelector
       * If valid: sell put at bid (conservative fill), deduct collateral

  4. DAILY ACCOUNTING:
     equity = cash + sum(collateral) + sum(unrealized_pnl)
     Record snapshot for equity curve
```

### RAMP Integration

- Reuses RAMPSignals.generate_signals() for momentum ranking
- Reuses MarketRegimeDetector.detect_regime() for regime classification
- Intersects RAMP ranked list with available options symbols (~12 stocks)
- Same regime-adaptive top_n as production RAMP

### Cost Model (conservative)

- Entry: sell at bid (not mid)
- Exit: buy at ask (not mid)
- Additional 1% of mid slippage buffer
- $0.02 per contract regulatory fee

### Regime Gate

```
STRONG_BULL + no crash: Open new positions, manage existing
WEAK_BULL / SIDEWAYS: No new entries, manage existing (exits only)
BEAR / UNPREDICTABLE / crash: Emergency exit all positions
```

---

## Strategy Parameters

| Parameter | Default | Search Range (IS only) |
|-----------|---------|----------------------|
| target_delta_min | -0.35 | [-0.40, -0.30, -0.20] |
| target_delta_max | -0.25 | [-0.30, -0.25, -0.15] |
| min_dte | 21 | [14, 21, 28] |
| max_dte | 35 | [28, 35, 45] |
| profit_target_pct | 0.50 | [0.40, 0.50, 0.65, 0.75] |
| loss_limit_multiple | 2.0 | [1.5, 2.0, 3.0] |
| max_positions | 5 | [3, 5, 7] |
| max_csp_allocation | 0.30 | Fixed |
| momentum_rank_cutoff | 20 | [10, 15, 20] |

---

## Walk-Forward Validation

| Period | Date Range | Purpose |
|--------|-----------|---------|
| In-Sample | 2022-01 to 2023-06 | Parameter optimization (grid search) |
| Out-of-Sample | 2023-07 to 2024-12 | Unbiased validation |

Optimization metric: Sharpe ratio (daily returns)
Constraint: Max drawdown < 10%

---

## Outputs

### Compatible with StandardReportGenerator:
- Daily equity curve (pd.Series) -> QuantStats tearsheet
- Comparison benchmarks: SPY, RAMP equity-only

### Options-specific metrics:
- Win rate (% positive P&L trades)
- Average premium collected per trade
- Average return on collateral per trade
- Average holding period (days)
- P&L distribution by exit reason
- P&L distribution by regime at entry
- Capital utilization (% of allocated capital deployed)
- Assignment rate (% reaching expiration ITM)

---

## File Structure

```
src/strategies/options/
    __init__.py
    data_loader.py              # OptionsDataLoader (reads E:\OptionsData)
    csp/
        __init__.py
        contract_selector.py    # CSPContractSelector
        position.py             # CSPPosition, CSPTrade dataclasses
        mark_to_market.py       # CSPMarkToMarket
        engine.py               # CSPBacktestEngine
        metrics.py              # CSP-specific performance metrics

tests/strategies/options/
    __init__.py
    test_data_loader.py
    csp/
        __init__.py
        test_contract_selector.py
        test_mark_to_market.py
        test_engine.py

config/strategies/
    ramp_csp.yaml               # Strategy parameters
```

---

## Success Criteria

1. Out-of-sample Sharpe >= 0.5
2. Out-of-sample max drawdown < 10%
3. Win rate >= 60%
4. Average return on collateral per trade >= 1%
5. Assignment rate < 5%

---

## Risk Considerations

- **Limited universe**: Only ~12 stocks means fewer trade opportunities; may have
  periods with no candidates if none are in RAMP's top_n
- **Regime detector latency**: Crash protection (VIX/SPY DD) provides secondary safety net
- **Data gaps**: Missing chain snapshots handled by Black-Scholes MTM fallback
- **Correlated drawdown**: All CSP positions are momentum stocks that may correlate
  in selloffs; 5-position limit and 30% cap bound exposure
- **Backtest bias risk**: Using bid/ask rather than mid prices mitigates fill assumption bias
