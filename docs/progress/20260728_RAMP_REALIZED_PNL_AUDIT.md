# RAMP Realized-PnL Trade-Log Audit - 2026-07-28

## Summary

The `docs/progress/20260609_RAMP_PREFLIGHT_CRASHLOOP_FIX.md` postmortem deferred
an audit of whether the 2026-06-04 to 06-08 target-aware exits left trade-log
gaps that understated realized PnL. This is that audit.

**Verdict: realized PnL is not trustworthy, and the cause is ongoing rather than
historical.** `f7ea22f` fixed one of three unlogged paths. Over the retained
journal window (2026-04-20 to 2026-07-28), **80 realized-PnL events totalling
$292,267 of notional sales were never written to the trade log**, against a
reported lifetime realized PnL of **-$3,199.77**. The unrecorded notional is two
orders of magnitude larger than the figure it is supposed to feed, so
`hg_strategy_realized_pnl_lifetime_usd` and every metric derived from it cannot be
relied on.

This audit was scoped to data integrity, not strategy performance. It produces no
verdict on RAMP's edge and did not run a backtest, so it did not route through
`strategy-lead`.

## Root cause: three unlogged paths, one fixed

`_execute_rebalance_target_aware` in `src/trading/adapters/ramp_live_adapter.py`
is the deployed rebalance path (`use_target_planner=True`, set in the runner since
`ec4a01c`, 2026-05-15). On the deploy branch
`ramp-phase4-turnover-regime-research`:

| Path | Records trade log? | Updates state? | Status |
|---|---|---|---|
| Full exits (`plan.exits`) | `log_exit` | `remove_position` | **Fixed** by `f7ea22f` |
| **Trims** (partial reductions) | **nothing** | **nothing** | **STILL BROKEN** |
| **Buys** (entries and top-ups) | **nothing** | **nothing** | **STILL BROKEN** |

Measured directly on the deployed branch: within
`_execute_rebalance_target_aware`, `log_entry` appears **0** times and
`add_position` appears **0** times. `log_exit` appears only in the full-exit loop.

The trim block submits the order and logs nothing:

```python
self.execution_engine.execute_order(
    symbol=sym, quantity=shares_to_sell,
    side=OrderSide.SELL, order_type=OrderType.MARKET,
)
```

No `log_exit`, no state decrement, and no return-value check.

Note this also means **crash protection realizes PnL invisibly**. There is no
stop-loss path in the adapter; VIX/SPY-drawdown risk reduction acts by shrinking
`plan.exposure_pct`, which flows into target weights and is executed **entirely
through the unlogged trim loop**.

## Evidence

### The trade log's own shape gives it away

RAMP rows in `/home/ec2-user/logs/trades_*.jsonl` (105 files, 2025-12-05 to
2026-07-27): 267 entries, 318 exits.

Activity by phase:

| Period | Entries | Exits | Interpretation |
|---|---|---|---|
| 2025-12-09 to 2026-05-15 | present, roughly balanced | present | legacy path logged both sides |
| 2026-05-16 to 2026-06-09 | **zero rows at all** | **zero** | blackout (spans the crash-loop) |
| 2026-06-10 to 2026-07-27 | **0 across 15 trading days** | 78 | target-aware path: exits only |

**Zero entries across 15 trading days while 225 BUY orders were placed** is the
clean confirmation that the buy path never logs.

### Unlogged trims, quantified

From `journalctl -u homeguard-multi`, retained back to 2026-04-20. Trims emit a
`[RAMP] TRIM <sym>: SELL <n> (current $X -> target $Y)` line but no trade-log row,
so the journal is the only record they happened at all.

| Metric | Value |
|---|---|
| TRIM events (unlogged) | **80** |
| Logged full EXIT events, same window | 149 |
| **Share of realized-PnL events unrecorded** | **80 / 229 = 35%** |
| Shares sold unlogged | 2,299 |
| **Notional trimmed, unrecorded** | **$292,267** |
| Mean notional per trim | $3,653 |
| Distinct symbols affected | 52 |
| By month | May 23, Jun 12, **Jul 45** |

The July count (45 trims, $158,597 notional) confirms this is **active**, not a
closed historical episode. The 26 full exits the journal shows for July match the
26 exit rows in the trade log exactly, which validates the method: the logging
that exists works, and the trims are simply absent.

### Why the reported number cannot be salvaged by inspection

`compute_lifetime_realized_pnl` (`src/utils/trading_logger.py:518-554`) does no
entry/exit pairing whatsoever. It sums `pnl_dollars` over rows where
`trade_type == 'exit'`:

```python
if row.get('trade_type') != 'exit':
    continue
pnl = row.get('pnl_dollars')
```

So a missing exit contributes exactly 0 with no warning, and there is no
balance check that could ever surface the omission. `tests/utils/test_lifetime_pnl.py`
codifies this behaviour, so the blind spot is intentional at the function level
and simply has no compensating control anywhere else.

`scripts/ops/backfill_lifetime_pnl.py` reimplements the same filter and inherits
the identical blind spot.

## Blast radius

- **`hg_strategy_realized_pnl_lifetime_usd`** understates (or misstates) by the
  realized PnL of 80 trims.
- **`hg_strategy_equity_usd`** is computed as
  `initial_capital + compute_lifetime_realized_pnl + unrealized`
  (`run_live_paper_trading.py:593-642`), so it inherits the error, and its peak
  and drawdown series are therefore contaminated.
- The `strategy_breakdown` Grafana dashboard reads the lifetime gauge directly.
- Any judgement of RAMP's live performance based on these numbers is unsound.

Note `hg_portfolio_equity_usd` is read straight from the broker account and is
**not** affected. Broker-sourced numbers remain trustworthy; only
Homeguard-derived realized PnL and strategy equity are suspect.

## What this audit cannot determine

**The sign and magnitude of the missing PnL.** The TRIM journal line records
symbol, share count, and current/target position value, but **not entry price**,
and no exit row exists. There is also no broker reconciliation: `git grep
realized_pnl` in `src/trading/brokers/ibkr/ibkr_broker.py` returns only
`unrealized_pnl`, so IBKR's own realized-PnL figure is never fetched or compared.

So the error is bounded in *notional* ($292k) but not in *PnL*. Closing that gap
requires external ground truth from IBKR (Flex Query or TWS trade report),
reconciled per symbol and date.

Two further irrecoverable gaps:

- **Exits swept by `sync_with_broker`.** When broker qty is 0 it does
  `del positions[symbol]` (`strategy_state_manager.py:759`) and writes no trade-log
  row. Anything closed this way left no trace in Homeguard at all. This is also
  what masked the 2026-06-09 phantom position.
- **The 2026-05-16 to 06-09 blackout.** No rows exist to audit.

## Fix APPLIED 2026-07-28 (recommendations 1 and 2)

Both silent paths now record. Committed `a0e75f8` on `main`, cherry-picked to
`f009df5` on the deploy branch, deployed, and `homeguard-multi` restarted at
01:40 EDT (14h before the 15:55 ET rebalance).

- **Trims** call `log_exit` with the **filled partial qty** and `entry_price` from
  state, then decrement via `update_position_qty` rather than `remove_position`
  since the position survives. A trim that consumes the whole position (rounding,
  or a target collapsing below one share) falls back to `remove_position`.
- **Buys** call `log_entry` and persist via `add_or_update_position` (the same
  utility the legacy path uses, so a top-up accumulates instead of resetting qty
  and losing the original cost basis).
- **Both** now check the `execute_order` return value, which neither did. A failed
  order previously looked identical to a filled one; on a falsy result they log an
  error and touch neither the trade log nor state.

Five tests added, each mutation-verified: reverting the trim `log_exit`, reverting
the buy `log_entry`, or swapping the trim's decrement for a removal all fail the
suite. Failure attribution checked by stashing, so the 6 unrelated failures in
`test_adapters.py` and `test_streaming_integration.py` are confirmed identical with
and without the change.

Restart verified non-destructive: 17 position gauges before and after, no
exceptions from the new paths, `NRestarts: 0`, and all 7 alert rules still healthy.

Placement on the deploy branch was verified explicitly rather than trusted to the
auto-merge, since `ramp_live_adapter.py` differs by ~437 lines between branches:
within `_execute_rebalance_target_aware` there are now 2 `log_exit` (full exit and
trim), 1 `log_entry`, and each sits inside its own loop.

### Still unverified until the next rebalance

RAMP trades once daily at 15:55 ET, so **no trim or buy has executed under the new
code yet**. The confirmation to run after today's rebalance:

```bash
python3 - <<'PY'
import json, glob, collections, os
d = collections.Counter()
for p in glob.glob(os.path.expanduser('~/logs/trades_20260728.jsonl')):
    for line in open(p):
        r = json.loads(line)
        if r.get('strategy') == 'ramp':
            d[(r['trade_type'], (r.get('metadata') or {}).get('exit_reason'))] += 1
print(d)
PY
```

Expect **entry rows to appear for the first time since 2026-06-09**, and any trim
to produce an exit row with `metadata.exit_reason == 'trim'`. Cross-check the
counts against `journalctl -u homeguard-multi | grep -cE "RAMP\] (TRIM|BUY)"` for
the same day; they should now match, where previously trims and buys had no
corresponding rows at all.

## Recommendations

Recommendations 1 and 2 are DONE (see above). 3 to 5 remain open.

1. **Log the trim path** (`ramp_live_adapter.py`, trim loop). Call `log_exit` with
   the trimmed share count and decrement state qty, mirroring the full-exit loop
   that `f7ea22f` fixed. This stops the bleeding and is the smallest change here.
   Also check `execute_order`'s return value, since the trim loop currently
   ignores it and a failed trim is indistinguishable from a successful one.
2. **Log the buy path.** Call `log_entry` and `add_position`. This does not affect
   realized PnL directly (only exit rows are summed) but it is why exits currently
   depend on `adopt_broker_positions` stamping `entry_price` from broker average
   cost, and it makes the log unauditable.
3. **Add a balance invariant.** A cheap guard that flags when trade-log-implied
   holdings diverge from broker holdings would have caught all of this within a
   day. This is the control whose absence let a 35% gap persist unnoticed.
4. **Reconcile against IBKR.** Expose broker realized PnL and compare periodically.
   Without it there is no ground truth for any of these numbers.
5. **Treat historical realized PnL as unreliable** rather than attempting a
   correction from Homeguard data. Note `backfill_lifetime_pnl.py`'s own header
   warns VictoriaMetrics silently drops samples older than ~90 days, so the
   May-June window is at or past the re-backfill horizon anyway.

Fixing 1 and 2 changes the live order path and needs a `homeguard-multi` restart,
so it should be scheduled outside 15:55 ET.

## Method

- Trade logs parsed from `/home/ec2-user/logs/trades_*.jsonl` (105 files).
- Trim events extracted from `journalctl -u homeguard-multi`, retained to
  2026-04-20. Journal retention is the binding limit on the window, not the bug.
- Code paths verified against `origin/ramp-phase4-turnover-regime-research`, the
  branch EC2 actually deploys, not `main`.
- Cross-check that validates the method: journal EXIT count for July (26) equals
  the trade-log exit-row count for July (26).
