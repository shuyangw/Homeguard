# Grafana Dashboard Polish + Per-Strategy Equity Metric - 2026-04-23

## Summary
Second pass on the dashboards after a user-led review surfaced six concerns the earlier session had not addressed: the equity curve was mixing three unrelated account series, the regime-state mapping was off-by-one so "SIDEWAYS" was being labeled "UNPREDICTABLE", order-rate panels rendered ambiguous "No data" instead of a flat zero, RAMP cache age thresholds flagged normal overnight state as a warning, and 31 panels had no inline explanation of what they showed. Added a new `hg_strategy_equity_usd{strategy}` gauge, rewrote the Equity Curve, fixed the regime mapping, cleaned up thresholds/noValue, and annotated every panel with a Grafana `description` tooltip. Also ran a read-only investigation into why CSCM holds only 0.1 BTC -- surfaced a separate, deeper issue (no rebalance events logged across 30+ days) which is out of scope for this PR but documented below.

## Changes Made

### New metric `hg_strategy_equity_usd{strategy}`
- **`src/monitoring/registry.py`**: added `update_strategy_equity(equity_usd)` method next to `update_strategy_initial_capital`. Wraps `set_gauge('hg_strategy_equity_usd', equity_usd, {'strategy': self.strategy})`.
- **`src/monitoring/hooks.py`**: matching optional-registry hook.
- **`scripts/trading/run_cscm_live.py` `_emit_metrics_tick`**: emits `equity = account.portfolio_value` on every tick alongside the existing portfolio metrics. No new broker call.
- **`scripts/trading/run_live_paper_trading.py`** (RAMP sidecar): emits the same right after `update_portfolio_metrics(...)`. Throttled to every 4 ticks like the other portfolio gauges.
- **`docs/monitoring/METRIC_SPEC.md`**: documented the new gauge.

Each strategy currently owns its whole broker, so `broker.get_account().portfolio_value` IS the strategy's total equity. If we ever share a broker across strategies, this will need per-strategy accounting -- flagged in-code.

### `config/monitoring/grafana/dashboards/portfolio_overview.json` (v2 -> v3)
- **Equity Curve** (panel 1): rewrote query from `max(hg_portfolio_equity_usd) by (broker)` to `hg_strategy_equity_usd` legend-by-strategy, added a dashed reference-line overlay querying `hg_strategy_initial_capital_usd`. Log-scale Y-axis (`custom.scaleDistribution: {type: "log", log: 10}`) so CSCM ~$97k and RAMP ~$1M both display legibly.
- **Regime State Code** (panel 4): value mappings were 0-4 but `MarketRegimeDetector.REGIMES` uses 1-5 (STRONG_BULL=1, WEAK_BULL=2, SIDEWAYS=3, UNPREDICTABLE=4, BEAR=5). Fixed mappings + bounded `min: 1, max: 5`. The live value `3` is SIDEWAYS, not UNPREDICTABLE as the dashboard had been claiming.

### `config/monitoring/grafana/dashboards/incident_review.json` (v2 -> v3)
- **Order Fill Rate** (panel 3) + **Order Reject Rate** (panel 4): added `fieldConfig.defaults.noValue: "0"` so the panel renders a flat zero line when the counter has never been incremented, instead of the ambiguous "No data" which looked identical to a broken metric pipeline.

### `config/monitoring/grafana/dashboards/infrastructure_health.json` (v2 -> v3)
- **Order Rejection Rate** (panel 3): same `noValue: "0"` fix.
- **RAMP Cache Age** (panel 5): thresholds moved from yellow@1h/red@24h to yellow@24h/red@48h. The cache is designed to refresh once per trading day (RAMP's `_load_cache_from_disk(max_age_days=1)`), so any age 0-24h is normal and should not warn.

### `config/monitoring/grafana/dashboards/strategy_breakdown.json` (v2 -> v3), `system_health.json` (v1 -> v2)
- Added `description` fields to every panel on every dashboard (31 total) explaining what the chart shows, the source metric and derivation, and what value ranges are normal / when to worry. Grafana renders these as a (i) icon at the top-left of each panel.

## Commits
- `db1c3ba` feat(monitoring): per-strategy equity metric; dashboard polish pass

## CSCM 0.1 BTC investigation (OUT OF SCOPE -- logged for follow-up)

User observed CSCM holds only `BTC/USD 0.1` (~$7.8k) of a $100k budget with `top_n=7`; expected ~$14.3k/coin across 7 coins. Ran read-only inspection on EC2:

### What we found
- **EC2 `config/trading/cscm_live.yaml`**: `top_n: 7`, `momentum_period: 28`, `btc_sma_period: 40`, `go_to_cash_in_bear: true`, `trailing_stop_pct: 0.25`, `weighting: equal`, `rebalance_day: sunday`.
- **Signal log history** `data/trading/logs/cscm/cscm_signals_*.jsonl` (30+ days from 2026-03-02 to 2026-04-01 visible): every entry is `{"type": "signal", ...}`. **ZERO entries of `{"type": "rebalance", ...}`** across the entire history. `grep '"type":\s*"rebalance"'` returns empty on every file.
- Each daily signal entry correctly computes top-7: `BTC/USD, ETH/USD, SUSHI/USD, LINK/USD, DOGE/USD, AVAX/USD, LTC/USD` with positive momentum scores. `regime: "bullish"`, `reduce_exposure: false`, `exposure_pct: 1.0`.
- **Demo portfolio state** (`~/.homeguard/demo/portfolio_state.json`): created 2026-01-12. Holds exactly one position `BTC/USD 0.1 @ $77,234.40`, `opened_at: 2026-04-18T05:06:53.649528`. `realized_pnl: -$3,407`.
- 2026-04-18 was a **Saturday**, NOT the scheduled `rebalance_day: sunday`.

### Interpretation
- CSCM appears to have been computing signals hourly and ranking top-7 coins correctly, but **never executed a scheduled rebalance** in the 30+ days of log history. Either `adapter.rebalance()` is never called, or it's called but `log_rebalance()` is never reached, or the logs are written elsewhere.
- The 0.1 BTC position was opened outside a rebalance window on a Saturday. Candidates: a manual test buy, an initial seed from a separate script, or a reconciliation/force-close path that only partially executed.
- Two concurrent CSCM processes run on EC2 -- `homeguard-cscm` runs `run_cscm_live.py` (with `CSCM_USE_DEMO_BROKER=true`) and `homeguard-cscm-demo` runs a separate `run_cscm_demo.py`. Both likely touch `~/.homeguard/demo/portfolio_state.json`. Which one is the source of truth for the 0.1 BTC position is unclear without inspecting each script.

### Recommended follow-up (separate session)
1. Confirm which of `run_cscm_live.py` vs `run_cscm_demo.py` writes to the demo portfolio and whether they race.
2. Add `log_rebalance()` instrumentation if it's missing on the live code path.
3. Trace `adapter.rebalance()` call frequency -- is `_should_rebalance()` ever returning True? On Sundays?
4. Decide: is `homeguard-cscm-demo` redundant now that `homeguard-cscm` is wired with demo-broker? If so, disable one.

None of the above is blocking the dashboard work -- the new `hg_strategy_equity_usd` panel will make it obvious that CSCM equity isn't growing (stuck at ~$96k cash), which is the right first signal that the rebalance is broken.

## Known Issues / Remaining Work
- CSCM rebalance execution gap (see above) -- biggest unresolved item.
- `homeguard-cscm` and `homeguard-cscm-demo` are both active -- likely redundant; one should be disabled.
- Panel descriptions are mechanical, not verified by a second reader -- if any are factually wrong, they're easy to patch in-place.
- Log-scale Y on the Equity Curve is a compromise. If it proves hard to read at a glance, an alternative is per-strategy normalized-return (equity / initial_capital) which would both start at 1.0 and be visually comparable.

## Validation

### Local (pre-deploy)
- `python -m json.tool` against all 5 dashboards -- all parse clean.
- `ast.parse` against all 4 edited Python files -- syntax OK.
- `pytest tests/monitoring -x -q` -- 37 passed, 0 failures.

### EC2 (post-deploy)
- `git pull` fast-forwarded `d70b33f` -> `db1c3ba`.
- `sudo cp config/monitoring/grafana/dashboards/*.json /var/lib/grafana/dashboards/homeguard/` -- Grafana reprovisioned from file-based dashboards dir.
- `sudo systemctl restart homeguard-cscm homeguard-multi` -- both `active` after 10s.
- `curl http://127.0.0.1:8084/metrics | grep hg_strategy_equity_usd` -> `hg_strategy_equity_usd{strategy="cscm"} 96607.54` (was not emitted before).
- `curl http://127.0.0.1:8082/metrics | grep hg_strategy_equity_usd` -> `hg_strategy_equity_usd{strategy="ramp"} 1014176.25` (was not emitted before).
- `curl http://127.0.0.1:8082/metrics | grep hg_regime_state_code` -> `3.0` -- this will now correctly render as "SIDEWAYS" in the dashboard after the mapping fix (was rendering as "UNPREDICTABLE" pre-fix).
- Dashboard files in `/var/lib/grafana/dashboards/homeguard/` updated with 2026-04-23 06:44 mtime -- Grafana's file provider picks these up within its poll interval.
