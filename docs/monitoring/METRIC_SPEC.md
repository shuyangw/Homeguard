# Homeguard Metric Specification v1

All metrics use the `hg_` prefix. Units are encoded in the metric name suffix.
Counters end with `_total`. Label values for `strategy` match systemd service
names: `omr`, `ramp`, `mp`, `cscm`.

## Portfolio Metrics (emitted by all strategy processes)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_portfolio_equity_usd` | gauge | `broker` | `broker.get_account()['portfolio_value']` |
| `hg_portfolio_cash_usd` | gauge | `broker` | `broker.get_account()['cash']` |
| `hg_portfolio_buying_power_usd` | gauge | `broker` | `broker.get_account()['buying_power']` |
| `hg_portfolio_drawdown_pct` | gauge | -- | rolling peak equity tracker |
| `hg_portfolio_day_pnl_usd` | gauge | -- | equity minus day-open equity |

Multiple processes emit portfolio metrics from the same broker account.
VictoriaMetrics deduplicates identical samples. Grafana uses `max by (broker)`.

## Strategy Metrics (emitted by owning strategy only)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_strategy_realized_pnl_usd` | gauge | `strategy` | StrategyStateManager |
| `hg_strategy_unrealized_pnl_usd` | gauge | `strategy` | StrategyStateManager |
| `hg_strategy_positions_count` | gauge | `strategy` | StrategyStateManager |
| `hg_strategy_capital_allocated_usd` | gauge | `strategy` | sum of market value of open positions (deployed capital) |
| `hg_strategy_initial_capital_usd` | gauge | `strategy` | starting/budgeted capital, emitted once at process startup |
| `hg_strategy_equity_usd` | gauge | `strategy` | current equity = broker.get_account().portfolio_value (cash + positions); emitted per-tick |
| `hg_strategy_rebalance_errors_total` | counter | `strategy`, `phase` | per-phase order-submission failures during rebalance (phases: buy, sell, close, reconcile, other). Incremented from RAMPLiveAdapter.rebalance() except-blocks. |
| `hg_strategy_last_signal_timestamp` | gauge | `strategy` | adapter run_once hook |
| `hg_strategy_signal_symbols_missing` | gauge | `strategy` | adapter data fetch |

## Position Metrics (per open position, dies on close)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_position_unrealized_pnl_usd` | gauge | `symbol`, `strategy` | per open position |
| `hg_position_qty` | gauge | `symbol`, `strategy` | per open position |

Typical cardinality: 20-40 series during market hours.

## Regime Metrics (RAMP primarily)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_regime_state_code` | gauge | -- | MarketRegimeDetector (0=STRONG_BULL, 1=WEAK_BULL, 2=SIDEWAYS, 3=UNPREDICTABLE, 4=BEAR) |
| `hg_regime_sma_signal` | gauge | `period` (20, 50, 200) | SMA relative position |
| `hg_regime_time_in_state_seconds` | gauge | -- | time since last transition |

## Order Metrics

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_orders_submitted_total` | counter | `strategy`, `side`, `broker` | ExecutionEngine |
| `hg_orders_rejected_total` | counter | `strategy`, `reason`, `broker` | ExecutionEngine |
| `hg_orders_filled_total` | counter | `strategy`, `broker` | ExecutionEngine |
| `hg_fill_slippage_bps` | histogram | `strategy` | fill price vs expected |

## Infrastructure Metrics

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_broker_reconnect_total` | counter | `broker` | broker client hook |
| `hg_broker_last_heartbeat_timestamp` | gauge (unix-seconds) | `broker` | last successful call; consumers compute age as `time() - value` |
| `hg_websocket_connected` | gauge (0/1) | `provider` | LiveDataProvider.is_connected() |
| `hg_websocket_symbols_subscribed` | gauge | `provider` | hub state |
| `hg_market_open` | gauge (0/1) | -- | synthetic, gates alerts |
| `hg_process_rss_bytes` | gauge | `strategy` | os.getpid() RSS |

## RAMP-Specific Metrics

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `hg_ramp_cache_age_seconds` | gauge | -- | os.path.getmtime on pickle |
| `hg_ramp_cache_hit_total` | counter | -- | cache load path |

## Naming Conventions

- Prefix: `hg_`
- Units in name: `_usd`, `_pct`, `_seconds`, `_bps`, `_bytes`
- Counters always end `_total`
- Gauges use units, never `_count` (use `_positions_count` pattern)
- `strategy` label values: `omr`, `ramp`, `mp`, `cscm`
- `broker` label values: `alpaca`, `ibkr`
- `provider` label values: `iex`, `sip`

Total: ~29 distinct metrics, ~100-150 active series during market hours.
