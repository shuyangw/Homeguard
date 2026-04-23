# Grafana Data-Correctness Pass + IBKR Routing/Cleanup - 2026-04-23

## Summary
Started from a user-visible Grafana complaint ("PnL charts can't go negative, CSCM doesn't show $100k"), audited every dashboard panel against the metrics actually being emitted, and surfaced three deeper issues underneath: the CSCM sidecar was reading positions from the wrong source, `homeguard-multi` was running a broken entry script that never emitted metrics at all, and once RAMP finally came up it was silently falling back to Alpaca instead of executing through IBKR. Along the way fixed a long-running IBKR `get_account()` subscription leak that was throwing ~134 error 322s per hour.

## Changes Made

### Dashboards + metrics surface
- **`config/monitoring/grafana/dashboards/strategy_breakdown.json`** (commit `b289d9f`): added `custom.axisCenteredZero: true` to Realized/Unrealized/Per-Symbol PnL panels so negative values render symmetrically around $0 instead of the Grafana no-data default range. Position Count got `min: 0` + integer decimals. Signal Freshness got `min: 0` + green/yellow@60s/red@300s thresholds. Capital Allocated panel renamed to "Strategy Capital (Deployed vs Starting)" and now plots both `hg_strategy_capital_allocated_usd` (solid) and the new `hg_strategy_initial_capital_usd` (dashed) so CSCM's $100k starting budget is visible on the chart.
- **`config/monitoring/grafana/dashboards/portfolio_overview.json`** (commit `b289d9f`): Day P&L got `axisCenteredZero: true`. Equity Curve got `min: 0`. Drawdown bounded at `max: 0` with red-base / yellow@-20% / green@-5% thresholds. Regime State Code got value mappings (0=STRONG_BULL green ... 4=BEAR red) so the raw integer is readable.
- **`config/monitoring/grafana/dashboards/incident_review.json`** (commit `b289d9f`): same drawdown clamp + Portfolio Equity `min: 0`.
- **`config/monitoring/grafana/dashboards/infrastructure_health.json`** (commit `b289d9f`): WebSocket Connected bounded 0-1 with connected/disconnected mappings; Broker Heartbeat Age thresholds @60s/300s; RAMP Cache Age thresholds @1h/24h.
- **`src/monitoring/registry.py` + `src/monitoring/hooks.py`** (commit `b289d9f`): new `update_strategy_initial_capital()` method and hook, emitting `hg_strategy_initial_capital_usd{strategy}` as a static per-session gauge.
- **`scripts/trading/run_cscm_live.py`** (commit `b289d9f`): sidecar emits the strategy's `--initial-capital` value (default $100k) once at startup.
- **`docs/monitoring/METRIC_SPEC.md`** (commit `b289d9f`): documented the new gauge; clarified that `capital_allocated` = sum of position market values (deployed), not budget.

### CSCM position-reading bug
- **`scripts/trading/run_cscm_live.py` `_emit_metrics_tick`** (commit `a8d0f09`): was querying `adapter.state_manager.get_positions('cscm')`, but `strategy_positions.json` is only populated by stock strategies (OMR/MP/RAMP) via `StrategyStateManager`. CSCM's actual holdings live in the broker's own persistence (DemoBroker's `~/.homeguard/demo/portfolio_state.json`, Coinbase/Alpaca server-side). Result: dashboards reported 0 positions even with a real BTC/USD 0.1 position held. Switched to `broker.get_crypto_positions()` — and dropped the per-symbol quote fetch since the broker already enriches each position dict with `current_price` / `market_value` / `unrealized_pnl`.

### homeguard-multi service drift
- **`/etc/systemd/system/homeguard-multi.service`** on EC2 had `ExecStart=...run_multi_strategy_streaming.py`, which crashed on startup with `TypeError: OMRLiveAdapter.__init__() missing 1 required keyword-only argument: 'broker_name'` (the kwarg was introduced by commit `e6c43a5`). The repo's `infra/ec2/homeguard-multi.service` already pointed at `run_live_paper_trading.py --strategy ramp` from commit `4705fb7`, but that updated unit file was never synced to the deployed path. Net effect: no RAMP metrics at all, and VM's `homeguard-ramp` scrape target had been `down` for an unknown period.
- **`infra/ec2/homeguard-multi.service`** (commit `3f27f8a`): added `ENABLE_METRICS=true` + `METRICS_PORT=8082` so the runner actually opens its scrape endpoint where VM expects it. Added `After=/Wants=homeguard-gateway.service` so the IBKR gateway is started alongside. Added `SyslogIdentifier=homeguard-multi` for journalctl consistency.
- **`infra/ec2/homeguard-multi.service`** (commit `23ce619`): `Wants=homeguard-gateway` only guarantees the unit is *started*, not *ready*. IB Gateway needs ~10s for Java + IBC automated login + port config before it binds 4002. Without waiting, RAMP's broker factory tries to connect immediately, gets `ConnectionRefusedError`, and the routing code silently falls back to `default_broker: alpaca` — which is why the Grafana portfolio label said `broker="alpaca"` despite `broker_routing.yaml` mapping RAMP to `ibkr`. Added `ExecStartPre` that polls `ss -ltn sport = :4002` until the gateway binds (max 120s).
- **`infra/ec2/homeguard-multi.service`** (commit `75c05a7`): the above ExecStartPre was hanging with "ss: command not found" because systemd units run with a minimal PATH and `ss` lives at `/usr/sbin/ss` on Amazon Linux 2023. Used the absolute path.
- Deployed all three by `sudo cp` to `/etc/systemd/system/`, `daemon-reload`, `restart`.

### IBKR account summary leak
- **`src/trading/brokers/ibkr/ibkr_broker.py` `_fetch_account`** (commit `d70b33f`): every `get_account()` tick was calling `ib.reqAccountSummaryAsync()` + `sleep(0.5)` + `ib.accountSummaryAsync(account_id)`. But `accountSummaryAsync()` is already idempotent — it calls `reqAccountSummaryAsync()` on its own only when the internal cache is empty, and returns cached values on every subsequent call. The explicit pre-subscribe opened a fresh IBKR server-side subscription on every tick, and nothing ever called `cancelAccountSummary`, so IBKR's per-client subscription count piled up until TWS returned error 322 "Maximum number of account summary requests exceeded" on every subsequent request. Observed rate: ~134 log lines / 67 failing reqIds (5 → 71) over 65 minutes. Account values still flowed through because the very first subscription was still delivering data, but the log noise was enormous. Dropped the explicit subscribe + sleep; let `accountSummaryAsync()` manage its own lifecycle.

## Commits
- `b289d9f` feat(monitoring): dashboards show negative PnL; emit CSCM initial capital
- `a8d0f09` fix(cscm-metrics): read positions from broker, not state manager
- `3f27f8a` fix(infra): homeguard-multi emits metrics; orders gateway dep
- `23ce619` fix(infra): wait for IBKR port 4002 before launching RAMP
- `75c05a7` fix(infra): use absolute path to ss in homeguard-multi ExecStartPre
- `d70b33f` fix(ibkr): stop leaking account summary subscriptions

## Known Issues / Remaining Work

- **`strategy_toggle.yaml` drift**: repo has `mp.enabled: false`, EC2 was manually toggled to `mp.enabled: true` on 2026-04-20. Because `homeguard-multi` is pinned to `--strategy ramp` in the unit file, the toggle is ignored for strategy selection — but the file-vs-reality mismatch remains. Not reconciled by this session; user's call whether to sync.
- **Cosmetic log bug in `run_live_paper_trading.py:1300`**: hardcoded `logger.success("Connected to Alpaca Paper Trading")` fires regardless of the actual broker. Prints even when `broker_name=ibkr`. Behavior is correct (IBKR `get_account()` is called and returns), only the log message is wrong.
- **Alpaca free-tier WS cap**: a zombie `homeguard-ramp` process (PID 5554, orphaned by the pre-fix crash loop) was holding a WS connection, tripping Alpaca's "1 concurrent WS per API key" limit with HTTP 429. Killing the zombie + restarting cleared it. No systemic fix landed — if another half-crash leaves orphans behind, it'll recur. A `KillMode=mixed` + `TimeoutStopSec=30` on the unit, or a startup precheck that kills matching orphans, would harden this.
- **`hg_websocket_symbols_subscribed=0`** during the verification window: market was closed (market hours in UTC: 13:30-20:00; verification ran 20:00+ and 00:00+ UTC). Streaming doesn't actively subscribe symbols outside market hours. Re-check during an open session before treating it as a bug.
- **MP/OMR scrape targets (ports 8083/8081) remain `down`**: expected — the `homeguard-mp.service` / `homeguard-omr.service` units are `disabled` and superseded by `homeguard-multi` (which is pinned to RAMP). If the priority is ever to revive MP or OMR, the unit files need explicit `ENABLE_METRICS=true` + `METRICS_PORT=...` adds similar to commit `3f27f8a`.
- **Pre-existing IBKR test failures**: `tests/trading/brokers/ibkr/test_config_and_errors.py::TestIBKRConfig::test_defaults` and `test_paper_detection` / `test_gateway_type_label` fail on stale assertions (test expects `client_id=1`, code uses `10`). Five errors in `test_contracts.py`. Not touched by this session but worth fixing next pass.

## Validation

- **Dashboards**: post-deploy spot check of Strategy Breakdown (Capital panel shows CSCM $100k dashed + $7,890 deployed), Portfolio Overview (Regime State Code shows "UNPREDICTABLE" text instead of `3.0`), Infrastructure Health (WebSocket Connected shows "connected" / "disconnected" text).
- **CSCM positions fix**: `curl http://127.0.0.1:8084/metrics` post-restart returns `hg_strategy_positions_count{strategy="cscm"} 1.0`, `hg_position_qty{symbol="BTC/USD"} 0.1`, `hg_position_unrealized_pnl_usd +$167.03`. VictoriaMetrics confirmed ingest of the new `hg_strategy_initial_capital_usd{strategy="cscm"} 100000.0` series.
- **homeguard-multi drift**: after three iterations (`3f27f8a` / `23ce619` / `75c05a7`), `systemctl is-active homeguard-multi` = `active`, `curl http://127.0.0.1:8082/metrics` returns 21 `hg_*` metrics, VM scrape target `homeguard-ramp:8082` transitioned from `down` → `up`.
- **Broker routing**: logs show `[Routing] Created broker: ibkr` + `[Routing] ramp execution broker: ibkr` + `[IBKR] Connected to IB Gateway (paper) at 127.0.0.1:4002 (clientId=10)`. Metrics labels flipped from `broker="alpaca"` ($100k Alpaca paper fallback) to `broker="ibkr"` ($1,014,088.82 IBKR paper).
- **IBKR 322 fix**: pre-fix PID 11403 logged 134 error-322 lines over 65m (~2/min). Post-fix PID 14066 ran for 5:46 with **0 error-322 lines**. Account metrics continue updating (equity, cash, buying_power all fresh and correct).
- **Unit tests**: `tests/monitoring` 37 passed. `tests/trading/brokers/ibkr` 82 passed (2 pre-existing failures + 5 pre-existing errors, all unrelated).
