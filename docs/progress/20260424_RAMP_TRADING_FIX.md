# RAMP IBKR Execution Fix + Broker Contract Test + Smoke Test - 2026-04-24

## Summary
RAMP's 3:55 PM ET rebalance today correctly selected 12+ S&P 500 symbols to buy but executed **zero** of them, all failing with `'IBKRBroker' object has no attribute 'place_order'`. Root cause: `ExecutionEngine` calls three deprecated broker shims (`place_order`, `get_position`, `get_positions`) that live on `BrokerInterface` but aren't inherited by `IBKRBroker` -- which extends the newer fragmented interface mixins instead. Fixed by switching `ExecutionEngine` to the non-deprecated names (`place_stock_order` etc.), added a parametrized contract test to prevent this class of regression, added a `hg_strategy_rebalance_errors_total` counter + dashboard panel so silent multi-symbol failures surface immediately, and shipped an end-to-end smoke test that exercises connect→submit→query→cancel against IBKR paper. Smoke test passed 10/10 steps on EC2 with the IBKR paper account in verifiably clean final state.

## Root Cause

`ExecutionEngine` (`src/trading/core/execution_engine.py`) called three methods that only resolve on `AlpacaBroker` via the deprecated `BrokerInterface` shims:

| Call site | Method | AlpacaBroker | IBKRBroker |
|---|---|---|---|
| line 316 | `get_position(symbol)` | ✓ (deprecated shim → `get_stock_position`) | ❌ absent |
| line 347 | `get_positions()` | ✓ (deprecated shim → `get_stock_positions`) | ❌ absent |
| line 393 | `place_order(...)` | ✓ (deprecated shim → `place_stock_order`) | ❌ absent |

`AlpacaBroker(BrokerInterface)` inherits from the deprecated composite; `IBKRBroker(AccountInterface, MarketHoursInterface, MarketDataInterface, StockTradingInterface, OptionsTradingInterface)` inherits from the split modern interfaces (see the "DEPRECATED" markers in `BrokerInterface` itself at `broker_interface.py:70-87`). Every call from the execution engine to IBKR would fail with `AttributeError` the moment it fired.

RAMP has never actually executed a live buy via IBKR since the routing fix -- yesterday (Thu Apr 23) the service came up at 22:14 UTC, after 3:55 PM ET's window (19:55 UTC), so no rebalance attempt happened. Today (Fri Apr 24) was the first real attempt, which then failed on every symbol.

## Changes Made

- **`src/trading/core/execution_engine.py`**: three one-line method-name fixes (`place_order → place_stock_order`, `get_position → get_stock_position`, `get_positions → get_stock_positions`). Signatures verified identical on both brokers.
- **`tests/trading/brokers/test_broker_contract.py` (new)**: parametrized static contract test. 24 cases (6 methods × (Alpaca, IBKR) × 2 assertion families). Uses `inspect.signature`, no broker instantiation, no credentials. Catches "broker missing required method" regressions at unit-test time.
- **`src/monitoring/registry.py` + `hooks.py`**: new `hg_strategy_rebalance_errors_total{strategy, phase}` counter (phases: `buy`, `sell`, `close`, `reconcile`, `other`). Methods `inc_rebalance_error` on both registry + hooks.
- **`src/trading/adapters/ramp_live_adapter.py`**: accept optional `metrics_registry` kwarg; increment the counter in the buy/sell/outer-rebalance except blocks with exception-swallowing wrappers.
- **`scripts/trading/run_live_paper_trading.py`**: thread `metrics_registry` through `create_ramp_adapter` and both call sites (explicit `--strategy ramp` and multi-priority fallback).
- **`config/monitoring/grafana/dashboards/incident_review.json` (v3 → v4)**: new "Rebalance Errors (5m)" panel querying `sum(rate(hg_strategy_rebalance_errors_total[5m])) by (strategy, phase)` with `noValue: "0"` and yellow/red thresholds.
- **`scripts/trading/smoke_test_ibkr_paper.py` (new)**: CLI that connects directly to IBKR paper (bypassing `broker_routing.yaml` to avoid clientId conflict with the running service), fetches account + baseline positions, places a LIMIT BUY 50% below market, queries + cancels, repeats for LIMIT SELL 200% of market, verifies clean final state. Idempotent, safe after-hours.
- **`docs/monitoring/METRIC_SPEC.md`**: documented the new counter.

### Smoke-test iteration notes
The first smoke-test run failed at step 1 for two separate reasons, each fixed in-session:
1. `load_broker_routing()` tried to construct `AlpacaBroker` which requires `ALPACA_PAPER_KEY_ID` / `_SECRET_KEY` -- not present in the SSH subshell. Workaround: bypass routing, instantiate `IBKRBroker` directly.
2. `broker_routing.yaml` and `config/ibkr.yaml` both pin `client_id=10` -- already held by the running `homeguard-multi` service, so the test's own connect attempt got TWS error 326 "client id already in use". Also: `IBKRConfig(client_id=99)` kwarg didn't override the yaml value because of pydantic v2 `mode='before'` validator semantics in `_load_yaml_and_env`. Workaround: set `os.environ['IBKR_CLIENT_ID']` in-script before instantiation (env vars _do_ override the yaml in the existing code path).

## Commits
- `cdb64b1` fix(trading): RAMP IBKR execution + broker contract test + smoke test
- `524e5d3` fix(smoke-test): connect IBKR directly, use distinct clientId
- `1677e5a` fix(smoke-test): use IBKR_CLIENT_ID env to override ibkr.yaml

## Known Issues / Remaining Work
- **CSCM rebalance gap still open** (from the 20260423 progress log) -- CSCM has been computing signals daily for 30+ days but zero `"type": "rebalance"` entries are in the JSONL. Separate session.
- **Two concurrent CSCM services** on EC2 (`homeguard-cscm` runs `run_cscm_live.py`, `homeguard-cscm-demo` runs `run_cscm_demo.py`). Likely redundant; decide which to disable.
- **`_compute_strategy_equity`** formula (introduced by a linter/user edit before this session) returns `initial_capital + sum(unrealized_pnl on tagged positions)`. Correct today but won't handle realized PnL accrual once RAMP actually fills trades. Revisit after first successful rebalance.
- **IBKRConfig kwarg-override inconsistency**: `_load_yaml_and_env` comment says "YAML values are defaults; explicit kwargs override" but the merge order in pydantic v2 `mode='before'` seems to not behave that way. Worth investigating -- tests pass for env-override but not kwarg-override, which is exactly the trap that made `test_config_and_errors.py` flaky earlier in the week.
- **Alpaca broker instantiation in `load_broker_routing`** is aggressive -- if the credentials are missing it logs an error and falls through to default. Worth a graceful-skip for dev/test environments where only one broker is needed.
- **`BrokerInterface` deprecation**: the whole deprecated-composite pattern is the root of today's bug. Migrating all callers (strategies, execution engine, state manager) off it and deleting `BrokerInterface.place_order` et al. would prevent the same shape of bug from recurring with the next broker integration.

## Validation

### Local (pre-deploy)
- `pytest tests/monitoring tests/trading/brokers/test_broker_contract.py tests/trading/test_execution_engine.py tests/trading/test_ramp_live_adapter.py` -- **101 passed**, 0 failures
- The 24 new contract-test cases all pass: every (broker_cls × required_method) combo asserts `hasattr` and signature parameters
- JSON parse + `ast.parse` on all edited files

### EC2 (post-deploy, smoke test)
Ran `python scripts/trading/smoke_test_ibkr_paper.py --symbol SPY --qty 1` on EC2 connected to IBKR paper (account `DUN312807`, clientId=99 to avoid conflict with homeguard-multi on clientId=10):

| Step | Outcome |
|---|---|
| 1. Connect to IBKR paper | ✓ connected clientId=99 |
| 2. Account snapshot | equity $1,014,263.68, 0 baseline positions |
| 3. Fetch SPY last price | $714.20 (via delayed market data subscription) |
| 4. LIMIT BUY 1 SPY @ $357.10 | order_id=6, status=pending |
| 5. Retrieve order by id | status=pending ✓ |
| 6. Cancel + re-query | cancel_order → True, post-cancel status=cancelled ✓ |
| 7. LIMIT SELL 1 SPY @ $1,428.40 | order_id=7, status=pending |
| 8. Retrieve + cancel | cancel_order → True, post-cancel status=cancelled ✓ |
| 9. Final state | 0 positions (unchanged), no lingering open orders ✓ |
| 10. Disconnect | broker.stop() clean ✓ |

Result: `=== SMOKE TEST PASSED ===`. IBKR paper account verified in clean final state.

The IBKR warnings "Warning 399: order will not be placed at the exchange until 2026-04-27 09:30:00 US/Eastern" are correct after-hours behavior -- orders queued for Monday's open -- and are irrelevant to the test because both were cancelled before Monday.

### Next real trading window
Monday 2026-04-27 3:55 PM ET is the next RAMP rebalance. With `place_stock_order` now resolving on IBKRBroker, orders should actually submit to the exchange. The new `hg_strategy_rebalance_errors_total` counter + Incident Review dashboard panel will flag any silent failures this time instead of burying them in the logs.
