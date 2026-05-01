# Dashboard PnL Audit + CSCM Duplicate-Service Fix - 2026-05-01

## Summary
Multi-hour audit of the Realized P&L dashboard panel that turned up a real
double-write bug (two CSCM services concurrent) plus a series of metric/render
quirks. Fixed the underlying duplicate writes by disabling the redundant
`homeguard-cscm-demo` service and updating the bot-update/bot-restart aliases
to target only the actually-enabled units. Also tried (and reverted) a
cosmetic equity backfill that created phantom drawdowns.

## Changes Made

### Real fix: disabled redundant CSCM service
- `homeguard-cscm-demo.service` had been running concurrently with
  `homeguard-cscm.service` (the latter runs `run_cscm_live.py` with
  `CSCM_USE_DEMO_BROKER=true`, which subsumes what the demo service did).
  Both processes called `adapter.rebalance()` periodically and both wrote to
  the same `/home/ec2-user/logs/trades_<date>.jsonl` file (TradeLogWriter is
  a global singleton with hardcoded path). Each had its own `DemoBroker`
  instance with its own order_id sequence starting at 1, so when their
  rebalance windows overlapped (which they did several times on Apr 26)
  the trade log got 4-5 entry rows per actual broker fill.
- Action: `sudo systemctl stop homeguard-cscm-demo` + `sudo systemctl
  disable homeguard-cscm-demo`. Live `homeguard-cscm` continues running;
  `hg_strategy_positions_count{strategy="cscm"}` still reports 7 positions
  post-stop, confirming the live runner is the source of truth.
- This was the recommended follow-up from
  `docs/progress/20260423_DASHBOARD_POLISH.md` ("Decide: is
  `homeguard-cscm-demo` redundant now that `homeguard-cscm` is wired with
  demo-broker? If so, disable one"). Yes -- disabled.

### `infra/ec2/instance_setup_bashrc.sh`
- `bot-update`, `bot-restart`, `bot-start`, `bot-stop`, `bot-status`,
  `bot-logs`, `bot-logs-recent` aliases all targeted the old set
  (`homeguard-omr homeguard-mp homeguard-cscm-demo`). Updated to the
  currently-enabled set: `homeguard-multi homeguard-cscm`. Without this,
  running `bot-restart` after disabling `homeguard-cscm-demo` would have
  silently re-started it (systemctl restart works on disabled services).
- Per-strategy aliases (`omr-start`, `mp-restart`, etc.) left intact for
  future re-enabling.

### `scripts/ops/backfill_lifetime_pnl.py` (added then reverted)
- Tried in commit `d8bf5f9` to also backfill `hg_strategy_equity_usd` for
  RAMP, to smooth a small DD-panel step at the deploy time of `ee6d635`
  (the commit that corrected `_compute_strategy_equity` to include
  realized PnL).
- Made things much worse. VM has equity samples from THREE formula eras:
  - pre-`f29bae1`: equity = `broker.portfolio_value` ~$1,014,000 (full IBKR account)
  - `f29bae1` to `ee6d635`: equity = `initial + sum(unrealized)` ~$100,000
  - post-`ee6d635`: equity = `initial + lifetime_realized + sum(unrealized)` ~$103,000
- Inserting Era-3 backfill values at past timestamps clashed with Era-1/2
  live-emitted samples still in VM. Panels using `max_over_time` picked up
  the higher backfilled values, current was lower -> phantom -5% to -90%
  drawdowns where there had been a small -1.2% step.
- Reverted in commit `bbb7e27`. Also deleted the polluted `hg_strategy_equity_usd`
  series from VM (`POST .../delete_series` for `{strategy="ramp"}`); live
  emission rebuilt it cleanly with Era-3 values from now forward.
- Lesson documented as a "DO NOT" warning at the top of `backfill_lifetime_pnl.py`.
  The original small DD step is best left as a one-time deploy artifact.

## Commits
- `bbb7e27` revert(backfill): remove equity backfill -- caused phantom drawdowns
- `<this commit>` chore(infra): disable cscm-demo aliases + 2026-05-01 session log

## Audit findings (informational)

### RAMP lifetime gauge math is internally accurate
- 180 logged RAMP exits sum to **$3,872.84** (matches the live gauge exactly)
- All `pnl_dollars` recompute from `(exit_price - entry_price) * qty` to the
  penny (within rounding)
- Trade log starts 2025-12-05; pre-Dec-5 history is not in our logs but may be
  in IBKR's broker-side execution history (queryable via API if needed)
- IBKR account total grew ~$14,637 from $1M; attributed to logged trades:
  ~$3,872 RAMP + $600 OMR + $260 MP = $4,732 realized + ~-$1,143 RAMP unrealized
  = ~$3,589 attributed. The ~$11k unaccounted is most likely pre-trade-log-era
  trades.

### CSCM realized PnL is genuinely $0
- Even after correcting for the dual-write over-logging, the broker truth
  (DemoBroker `portfolio_state.json` with `realized_pnl: 0.0`) confirms zero
  realized PnL. CSCM has bought 7 positions on Apr 26 and held them since
  (no exits, no rebalance after the initial buy). The "rebalance never fires"
  issue is real and remains a separate concern.

### Dashboard panel evolution (this session)
The Cumulative Realized P&L panel went through several iterations:
1. Original: `hg_strategy_realized_pnl_usd` (today gauge, daily reset cliff)
2. Switched to `lifetime - first_over_time(...[$__range])` -- range-relative
3. Backfilled lifetime gauge from trade logs for historical visibility
4. Switched to raw `hg_strategy_realized_pnl_lifetime_usd` (cumulative)
5. Added `stepAfter + spanNulls + showPoints=never + lineWidth=2` for clean step rendering
6. Switched to `last_over_time(metric[90d])` for carry-forward across exit gaps
7. Added `and on(job, instance) (up == 1)` to hide dormant strategies (OMR/MP)
8. Removed `fillOpacity` (inconsistent appearance for $0 series like CSCM)
9. Added a separate "Today's Realized P&L" stat tile on Portfolio Overview

Each step was driven by a user observation. The final form is robust, honest,
and renders cleanly across time ranges.

## Known Issues / Remaining Work
- **DD panel still has a small step at ~22:00 ET Apr 30** -- the original
  Era-2 to Era-3 transition. Not fixable cosmetically without a much larger
  cleanup effort (delete pre-deploy equity data + careful re-backfill).
  Recommend living with it as a deploy artifact.
- **CSCM rebalance still fires only at startup** -- the bot bought 7 positions
  on Apr 26 and has not rebalanced since. `_should_rebalance()` returns False
  on subsequent `_run_once` cycles ("Already rebalanced today"). The Sunday
  rebalance gate hasn't been re-tested since the dual-service fix. If it
  doesn't fire next Sunday, that's the bigger CSCM bug to track down.
- **Trade log idempotency layer NOT added** -- decided against per user.
  The root cause (dual writers) is fixed, and a robust dedup key design is
  tricky given DemoBroker's reused order_ids. Re-evaluate if a similar bug
  recurs.
- **`infra/ec2/services/homeguard-cscm-demo.service`** still in repo but
  disabled on EC2. Could be deleted from repo for clarity, but leaving as
  a known-disabled artifact for now.
- **`cscm_demo_*.sh` helper scripts** still in `infra/ec2/`. Reference the
  disabled service. Mostly harmless but stale. Could clean up later.

## Validation
On EC2 post-disable:
- `sudo systemctl is-active homeguard-cscm-demo` = `inactive`
- `sudo systemctl is-enabled homeguard-cscm-demo` = `disabled`
- `sudo systemctl is-active homeguard-cscm` = `active` (sole CSCM writer)
- `ps -ef | grep cscm` shows ONE process (`homeguard-cscm`)
- `curl http://127.0.0.1:8084/metrics | grep hg_strategy_positions_count` returns
  `cscm 7.0` (live gauge healthy)

In Grafana (already verified earlier in session):
- "Cumulative Realized P&L by Strategy" panel: clean step lines for RAMP and
  CSCM, dormant strategies hidden by `up == 1` filter.
- "Today's Realized P&L" stat tile: shows current-session realized PnL.
