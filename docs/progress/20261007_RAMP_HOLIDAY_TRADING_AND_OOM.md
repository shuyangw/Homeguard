# RAMP Holiday Trading, Duplicate Orders and OOM Protection - 2026-10-07

## Summary
Found that RAMP trades on NYSE holidays and after early closes because `IBKRBroker.is_market_open()` is a plain weekday 09:30-16:00 clock check, confirmed it happened on Labor Day 2026-09-07, and found a second bug that re-sends orders when IBKR answers with a warning. Built the calendar fix (branch, not deployed) and protected `homeguard-multi` from the OOM killer (pushed, not yet active on the instance).

## Findings
- **Holiday trading (confirmed)**: at 15:54 ET on Mon 2026-09-07 (Labor Day) RAMP ran its rebalance in SELL-ONLY mode and placed market DAY sells for IP 54, GNRC 10, EIX 36, BLDR 31. IBKR answered Warning 399 ("will not be placed at the exchange until 2026-09-08 09:30") and queued them. Order 257539 (IP) filled 37/54 at $36.55 at 09:30:06 on 2026-09-08 (Loki, `homeguard-multi.service`).
- **Duplicate orders on IBKR warnings (confirmed)**: the broker maps the warning to status `ValidationError` and the order path re-sends, so every Labor Day sell went out 3 times (12 orders for 4 intended). IBKR validation warnings occurred only on 2026-09-07 (12) and 2026-09-08 (1, a 2161 price-cap notice on the queued fill) in the last 30 days, so normal rebalances have not hit it, but any warning (including 2161 during market hours) would.
- **Rebalance re-fires every ~15 s (confirmed)**: `should_run_now` de-duplicated only `entry`/`exit`, so RAMP's `rebalance` fired on every check inside the +/-60 s window: 5-8 full rebalances per day in the decision records since at least 2026-08-05.
- **Short positions in a long-only book (confirmed against IBKR, read-only clientId 98)**: the paper account holds 8 shorts, about $33.7k notional: ADSK -36, ALB -90, APO -25, BLDR -22, CME -14, EIX -25, GNRC -5, MOH -35. RAMP's own records match. 6-7 shorts date from at least 2026-08-05; BLDR, EIX and GNRC were added on 2026-09-08 when the triplicated Labor Day sells filled. RAMP has been in SELL-ONLY mode (over its position cap), which skips buys, so nothing covers them.
- **RAMP paused 2026-10-07 10:51 ET** (operator-approved): `strategy_toggle.yaml` ramp `enabled: false` (variant v11 kept), `modified_by: claude-code-pause-2026-10-07`. Positions, including the shorts, are left open and unmanaged. Re-enable only after the duplicate-order fixes are deployed and the shorts are dealt with.
- **Exposure ahead**: Thu 2026-11-26 (Thanksgiving), Fri 2026-12-25, Fri 2027-01-01 (closed all day) and Fri 2026-11-27, Thu 2026-12-24 (13:00 close). The instance's start/stop schedule also ignores holidays.
- **RAMP memory**: 30-day peak RSS 468MB on trading days, about 350MB on weekends (VictoriaMetrics `hg_process_rss_bytes`).
- Morning start 2026-10-07 08:00 came up cleanly; both status-check alarms OK.

## Changes Made
- **`infra/ec2/homeguard-multi.service`**: `OOMScoreAdjust=-900` (same as the legacy homeguard-ramp unit) and `MemoryMax=1G` (about 2x the peak). On main and the deploy branch; NOT active until the unit is reinstalled and `homeguard-multi` restarted.
- **`src/trading/brokers/ibkr/ibkr_broker.py`** (branch `feat/ibkr-calendar`, not merged): `is_market_open()` reads today's NYSE session from `pandas_market_calendars`, cached per day; closed on holidays and after early closes, open on [open, close). 14 tests in `tests/trading/brokers/ibkr/test_market_hours.py`.

## Commits
- `1b1404a` fix(infra): protect homeguard-multi from the OOM killer
- `65fa24d` fix(infra): cap homeguard-multi memory at 1G (main; deploy branch `f94b9a8`)
- `3c17ecc` / `137697b` test + fix(ibkr): NYSE calendar for is_market_open (branch `feat/ibkr-calendar`)

- **Duplicate-order fixes** (main `1ae74f4..67e97f2`, deploy branch `4521e1d..e83dd40`): ExecutionEngine retries only placement failures, never re-places an accepted order, and on a timeout cancels then polls the final state (a fill that lands during the cancel counts as success); the runner fires each scheduled action once per day, persisted in `fired_actions.json` in its log dir so a restart inside the window does not re-fire. Independent Opus review: ready with fixes; its Important #2 (fill during cancel) and #4 (restart inside window) were fixed with failing-first tests.
- **Operator decisions (2026-10-07)**: deploy calendar fix + OOM limits + duplicate-order fixes in one `homeguard-multi` restart at 16:20 ET (scheduled in-session); leave the 8 short positions open for now; RAMP stays paused until the operator re-enables it.

## Deploy (2026-10-07 16:20 ET)
- Pre-checks: execution lock free (RAMP's lock from 15:54 had expired at 15:58); RAMP disabled; its 15:55 decision skipped at the `strategy_enabled` gate.
- Instance pulled to `e83dd40`; `strategy_toggle.yaml` unchanged by the pull, RAMP still `enabled: false` (v11).
- `homeguard-multi` unit reinstalled and restarted at 16:22 ET: active in about 3 s, `OOMScoreAdjust=-900`, `MemoryMax=1073741824`, 0 restarts, no tracebacks, RAMP v11 loaded, IBKR market data farm OK.
- On-instance checks: `IBKRBroker().is_market_open()` is False after 16:00; `ExecutionEngine.settle_timeout` is 5.0; `LiveTradingRunner._mark_fired` exists.
- IBKR paper smoke test (`scripts/trading/smoke_test_ibkr_paper.py`, full mode): PASSED, exit 0; engine counters 2 successful, 0 failed, 0 retries; the after-hours orders drew IBKR Warning 399 and were placed once and cancelled (no duplicates); positions unchanged, no lingering orders.
- Read-only position check (clientId 98): 26 stock positions, 18 long, the same 8 shorts in the same quantities as at 10:51 ET, 0 open orders.
- New on reconnect: IBKR code 2172, "The version of the application you are running, 1037.1, needs to be upgraded, as it will be desupported on 20261215". Not caused by the deploy (absent from the journal earlier today and on 2026-10-06).

## Known Issues / Remaining Work
- **Upgrade IB Gateway before 2026-12-15** (IBKR code 2172: application version 1037.1 desupported on that date).
- RAMP is paused and the 8 shorts are open: re-enabling RAMP and handling the shorts are operator decisions.
- Known residual risk (review #1): a placement that raises AFTER IBKR accepted the order (run_sync timeout during the post-place sleep, or `_translate_order` raising) is still treated as "no order" and retried. Fix by keeping `trade` once `placeOrder` returns, or by an `orderRef` idempotency key.
- Review #3: partial fills reach callers only in the exception message, so RAMP state and the trade log can diverge until the next broker sync.
- Review minors deferred: cancel message ignores cancel's return value; OrderNotFound logged at ERROR; accepted-then-timed-out orders counted as rejections in metrics; cancel_order bypasses run_sync.
- Restart `homeguard-multi` after 16:15 ET (operator go-ahead) to activate the OOM limits; the 08:00-09:15 window was missed because Tailscale SSH was awaiting browser approval.
- Deploy the calendar fix (go-ahead needed): merge, cherry-pick to the deploy branch, same restart, then run `scripts/trading/smoke_test_ibkr_paper.py`. Check the instance venv has `pandas_market_calendars` first (requirements pin 4.4.1; this machine has 5.1.1). Deadline: 2026-11-26.
- Fix the duplicate-order bug: treat IBKR warnings 399/2161 as accepted, not as failures to retry.
- Confirm or rule out oversold positions from Labor Day (IP, GNRC, EIX, BLDR).
- The 2026-09-08 log shows the SELL-ONLY rebalance branch repeating about every 15 s near 15:55; worth checking whether the rebalance fires more than once per window.

## Validation
- Calendar fix: 14/14 new tests; wider suite `tests/trading tests/monitoring tests/console_agent` 1112 passed, 12 skipped.
- Evidence from Loki and VictoriaMetrics through the Grafana connector (instance reachable on the tailnet; SSH pending approval).
