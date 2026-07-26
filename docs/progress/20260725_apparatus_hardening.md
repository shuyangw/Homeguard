# Apparatus Hardening Sweep - 2026-07-25

## Summary
Worked down the defect list surfaced by the #35 Kalman diagnostic. **Five fixes
merged**, four of them affecting every FX spread result and one affecting every
carry result. All defects were OPTIMISTIC, so prior FAIL verdicts stand a
fortiori -- but each would have made a future PASS untrustworthy, which is
exactly the situation we hit when the Kalman arm posted +0.42.

## Fixed

| # | Defect | Severity | Fix |
|---|---|---|---|
| 1 | **PBO stub truncation** | High | `_compute_pbo` truncated every window to the SHORTEST, so a 65-day stub cut all 13 columns from ~260 rows to 65 -- the reported PBO described 25% of the sample, and PBO gates the whole Track B vertical. Stubs (< half the median length) are now dropped, with retained/discarded counts LOGGED. |
| 2 | **Zero execution lag** | High | The spread simulator filled at the same bar's close the signal was computed from. Added `execution_lag`, DEFAULT 1. |
| 7 | **Silent-skip** | High | When the signal was unavailable the strategy emitted nothing (simulator flattens + charges a round trip) but KEPT its position state, so it phantom-re-entered later paying a SECOND round trip, with the holding period still measured from the ORIGINAL entry (silently breaking max_days). Fixed in all three FX spread strategies. |
| 4 | **False purge/embargo claim** | Medium | `_build_windows` emitted contiguous windows while the pre-registration claimed purge/embargo. Added `purge_days` (default 0, behaviour-neutral) with precise semantics, and corrected the pre-registration in place. |
| -- | **FRED publication lag** | High | (From the 07-22 audit, not the Kalman list.) Monthly FRED series are stamped at the FIRST of the month but hold that month's AVERAGE, so ffilling from the stamp let carry see the current month's rate from day 1: a 1-2 month lookahead on the carry SIGNAL. Monthly series now lag 60 days, daily policy rates 1 day. |

Defect 5 (vol-target same-bar leak) is **resolved as a consequence of fix 2**:
`_spread_sigma` is computed by the strategy at bar i and, with `execution_lag=1`,
is now consumed to size a fill at bar i+1. Causal.

## Two bugs I introduced and caught

Worth recording, because in both cases the unit tests passed and something else
caught it:

1. **The lag fix was initially wrong.** I shifted the BOOK keys but the simulator
   gated on its UNSHIFTED rebalance grid, so every lagged signal landed on a
   non-rebalance day and was silently dropped -- `execution_lag` meant "trade
   nothing", not "trade later". `_lag_book`'s unit tests passed because the
   shift itself was correct in isolation. What caught it was an end-to-end run
   returning IDENTICAL equity for lag=0 and lag=1, which should be impossible.
   The action grid now shifts with the book; regression test asserts a lagged
   run still trades.
2. **The FRED lag inference was too strict.** Requiring 3+ observations to infer
   spacing sent a short DAILY fixture down the conservative MONTHLY branch and
   NaN-ed the panel. Two points are enough for one diff.

## Still open (ranked)

1. **Inert event filter (High).** `config/macro_calendar/cb_decisions.yaml` holds
   exactly one RBA and one RBNZ date (both 2025), so the "+-7d blackout" is a
   no-op in 12 of 13 windows. Needs a historical RBA/RBNZ backfill. Until then
   no result validates or invalidates an event-aware variant.
2. **Spike cleaner is future-conditioned (Medium).** Nulls bar t using
   `r[t] + r[t+1]`. ~41 bars project-wide (0.035%). Needs a causal mode.
3. **Trade log lacks exit schema (Critical per methodology 11.9).** Blocks
   MAE/MFE, hold-time, and exit-reason diagnostics.
4. `trades_oos` boundary-day attribution (low); parameter-budget documentation.

## Commits (main = origin = c46937f)
- `ec06db8` PBO stub truncation + honest execution lag
- `10ceb23` silent-skip defect across all three FX spread strategies
- `eb9e972` purge_days support + corrected the false purge/embargo claim
- `c46937f` FRED publication lag

## Validation
2331 pass in tests/backtesting + tests/strategies + tests/data (was 2328 before
this sweep; net -1 failure, +3 tests). The 24 remaining failures are pre-existing
and unrelated: missing futures data on this Mac, DGS10 not downloaded, and a
sentiment-analyzer test. None are in the modules touched.

## Note on where the effort went
This sweep produced no new strategy results, by design. The Kalman diagnostic
delivered a positive Sharpe (+0.42) into an apparatus that could not be trusted
to evaluate it -- same-bar fills, a PBO computed on a quarter of the sample, a
carry signal seeing rates a month early. Hardening the apparatus IS the North
Star's stated job, and the ordering matters: a believable PASS requires a
believable harness first.
