# Strategy-Pluggable Futures Runner - Design (Option B)

**Date:** 2026-07-02 · **Status:** approved, pre-plan · **Depends on:** merged futures harness (`main` @ a4972f0)

## Goal

Replace the hardcoded `CarverMomentumStrategy` in `run_futures_backtest` with a
config-driven strategy lookup, so a new futures signal is a strategy class + a
registry entry + a config `name` -- no runner edit. Unblocks C (carry) and any
later futures signal (Donchian, MOP, etc.) on the plug mechanism.

## Context (verified)

- `src/strategies/registry.py` already provides mature, generic, lazy name->class
  resolution (`get_strategy_class(name)`), display-name aliases, a class cache, and a
  runtime `register_strategy`. It returns `BaseStrategy` subclasses. `CarverMomentumStrategy`
  is NOT currently registered.
- `src/backtesting/engine/futures_backtest.py::run_futures_backtest` hardcodes the strategy
  on one line: `forecasts = CarverMomentumStrategy(universe).forecast_panel(close)`, and
  passes a hardcoded `"CarverMomentum"` label to `StandardReportGenerator`.
- `CarverMomentumStrategy(universe, speeds=None, forecast_cap=20.0, **params)` extends
  `MultiSymbolStrategy` (-> `BaseStrategy`) and exposes
  `forecast_panel(close_panel: pd.DataFrame) -> pd.DataFrame` (per-root forecast in the
  +/- forecast_cap convention that `FuturesPortfolioSimulator.run_sized` already sizes).
- Existing futures configs (`carver_tsmom.yaml`, `carver_tsmom_broad.yaml`) have
  `strategy.universe` but NO `strategy.name`.

## Architecture

Reuse the existing `registry.py` (do NOT build a parallel futures registry). Register
Carver there; teach `run_futures_backtest` to resolve the strategy by config name through
`get_strategy_class`. The strategy INPUT contract stays close-only (per the brainstorming
decision): a registered futures forecast strategy exposes
`forecast_panel(close_panel) -> forecast DataFrame` and constructs as
`__init__(self, universe, **params)`.

## Components

1. **Register Carver** (`src/strategies/registry.py`): add
   `"CarverMomentum": ("src.strategies.advanced.carver_momentum_strategy", "CarverMomentumStrategy")`
   to `_STRATEGY_REGISTRY`, plus display aliases in `_DISPLAY_NAME_MAP`
   (`"Carver"`, `"Carver TSMOM"`, `"Carver Momentum"` -> `"CarverMomentum"`). Additive only.

2. **`SupportsForecastPanel` protocol + validation** (defined INLINE in
   `src/backtesting/engine/futures_backtest.py` -- it is ~5 lines, does not warrant a new
   file): a `typing.Protocol` (with `@runtime_checkable`) declaring
   `forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame`. Used for typing; at
   runtime the harness checks `hasattr(strat, "forecast_panel")` (or
   `isinstance(strat, SupportsForecastPanel)`) and raises a clear `ValueError` naming the
   strategy if absent.

3. **`run_futures_backtest` wiring** (`src/backtesting/engine/futures_backtest.py`): replace
   the hardcoded strategy line with:
   - `name = strategy_cfg.get("name", "CarverMomentum")` (default preserves existing
     name-less configs)
   - `params = strategy_cfg.get("params", {})` (dict passed to the constructor)
   - `cls = get_strategy_class(name)`
   - `strat = cls(universe, **params)`
   - runtime check: `forecast_panel` present, else `ValueError`
   - `forecasts = strat.forecast_panel(close)`
   - report label uses the resolved strategy name instead of the hardcoded `"CarverMomentum"`.

## Data Flow

Unchanged except the strategy is resolved, not hardcoded:
`config -> get_strategy_class(name) -> cls(universe, **params).forecast_panel(close) -> run_sized`.

## Config Schema (additive, backward-compatible)

```yaml
strategy:
  name: CarverMomentum        # optional; defaults to CarverMomentum
  universe: [...]             # required (unchanged)
  params: {}                  # optional dict -> strategy __init__ kwargs (e.g. forecast_cap)
```

Existing configs (no `name`, no `params`) run unchanged via the defaults.

## Error Handling

- Unknown `strategy.name` -> `get_strategy_class` already raises `ValueError` listing
  available strategies.
- Resolved class lacks `forecast_panel` -> `ValueError` naming the strategy (a strategy
  registered for the equity path, e.g. one returning signals not forecasts, must fail loud
  rather than silently mis-size).
- Bad `params` (unexpected kwarg) -> the strategy constructor raises `TypeError`; surfaced,
  not swallowed.

## Testing

- Registry resolves `"CarverMomentum"` (and an alias) to `CarverMomentumStrategy`.
- `run_futures_backtest` with NO `strategy.name` still runs Carver (backward compat).
- `run_futures_backtest` with explicit `name: CarverMomentum` yields the SAME result as the
  pre-B hardcoded path (equivalence) on a small fixed slice.
- Unknown `name` -> `ValueError` (clear message).
- A strategy missing `forecast_panel` -> `ValueError` (clear message).
- **Pluggability proof:** register a runtime STUB forecast strategy (constant forecast
  panel over the universe) and run it through `run_futures_backtest`; assert the harness
  uses the stub (result reflects the constant forecast), not Carver.

## Out of Scope (deferred to C or later)

- Walk-forward pluggability: `run_carver_walkforward.py::_run_window` still reconstructs a
  Carver config; making it strategy-agnostic belongs to C's spec (carry needs walk-forward
  too, and its data contract is pinned there).
- Richer-than-close data contract (carry near/far prices, high/low for Donchian): C's spec.
- No change to the equity/crypto path, the simulator, sizing, loader, or existing configs.

## B/C Tie-In

Once B lands, C (carry) is: a `FuturesCarryStrategy` implementing `forecast_panel` (fed
its carry data per C's own data decision) + a registry entry + a config `name: FuturesCarry`
-- no runner edit. B is the enabler; C is the first real consumer.
