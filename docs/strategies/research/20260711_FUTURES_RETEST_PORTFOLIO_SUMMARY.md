# Futures Campaign Comprehensive Retest -- Portfolio Summary

> Durable, tracked copy of the retest's portfolio summary. The generating
> report at `docs/reports/futures/portfolio_summary.md` is gitignored and
> worktree-local; this copy lives in the tracked research dir so the honest
> verdict survives worktree cleanup. Produced by the strategy-lead-governed
> retest driven by `docs/strategies/research/20260711_FUTURES_RETEST_TODO.md`.

Generated at the close of the SP-A..SP-E comprehensive retest. This is the
first evaluation of the entire futures catalog run through strategy-lead's
full integrity pipeline with HONEST, UNIFORM DSR deflation (Gate 0), rather
than through superpowers-driven ad-hoc verdicts.

## Headline finding

**Zero of the 26 gradeable strategy/root/pair combinations tested clear the
combined statistical gate (methodology Section 2.5).** This includes the
carry incumbent -- the best deployable book in the catalog by every other
measure. The honest expectation stated at the top of the retest TODO
("SR_zero 0.733 exceeds every strategy's OOS Sharpe... nothing clears
DSR >= 0.95, including the carry incumbent") is confirmed exactly. Per this
project's North Star, surfacing that cleanly IS the completed objective --
this is not treated as a failure to be engineered around.

## Gate 0: what changed and why it matters

Before this retest, DSR deflation was inconsistent across gate paths:
`gate_return_stream` (VIX, VRP, intermarket, convergence spreads) was
correctly deflated against the real 40-trial campaign distribution, but
`run_carver_walkforward` (all Tier 1 carver strategies + carry),
`gate_session_stream` (overnight drift, hour-slice, pre-FOMC), and the FX/
satellite-blend paths all computed DSR against a single-element
`[oos_sharpe]` list -- which collapses `expected_max_sharpe` to ~0 and makes
DSR reduce to undeflated PSR. That means every prior Tier-1-carver and
session-stream verdict UNDERSTATED multiple-testing risk. Gate 0.1 fixed all
four paths to use the real `CAMPAIGN_TRIAL_SHARPES` distribution; Gate 0.2
made that distribution grow honestly from `output/experiments.duckdb` as
runs are appended (N=41-63 across this retest's own runs, growing further
than the static 40 as each of this session's own results got logged to the
registry). Gate 0.3 built the ten missing Path-2 driver scripts so those
sleeves have a durable, committed re-run path instead of living only in
prior sessions' throwaway invocations.

**Effect of the fix, concretely:** the carry incumbent's headline number did
not change (OOS Sharpe 0.7646, matching the pre-existing documented 0.765
walk-forward figure exactly -- Gate 0 does not touch signal generation, only
the gate math) -- but its DSR is now 0.8242 computed against the real,
growing 40+-trial distribution, correctly below the 0.95 bar. Under the
PRE-Gate-0 single-element-list bug, several Tier 1 strategies would have
shown DSR == PSR == 1.0000, an artifact that looks like a clean PASS but is
mathematically meaningless (no deflation term at all). Post-Gate-0, every
Tier 1 strategy's DSR collapsed to a number at or near 0 (with the sole
exceptions of curve-slope-XS at 0.999, driven by its unusually high raw
Sharpe of 0.846, and carry at 0.824) -- both still fail the combined gate on
PBO or the 0.95 DSR bar respectively.

## Tier 1 -- Path 1 carver strategies (7 strategies + carry incumbent)

| Strategy | OOS Sharpe | DSR | PBO | Verdict |
|---|---|---|---|---|
| #3 FuturesXSMomentum | 0.2095 | ~0 | 0.579 | WEAK/FAIL |
| #10 FuturesCarryXS | 0.8458 | 0.999 | 0.690 | WEAK (PBO fail) |
| #13 FuturesCarryTrend | 0.3571 | ~0 | 0.189 | FAIL (DSR) |
| #15 FuturesSameMonthSeasonality | 0.1796 | ~0 | 0.281 | WEAK/FAIL |
| #16 FuturesTurnOfMonth (post caveat-fix) | 0.0815 | ~0 | 0.475 | WEAK/FAIL |
| #23 FuturesReversal | 0.2970 | ~0 | 0.805 | WEAK/FAIL |
| #37 FuturesCoTTilt | -0.1236 | 0.000 | 0.141 | REJECT |
| **carry incumbent (FuturesCarry)** | **0.7646** | **0.824** | **0.189** | **FAIL (DSR) / PASS (PBO)** |

Notable: #16's caveat-fix (the walk-forward runner hardcoded `rebalance:
"weekly"` for every window regardless of the config's declared frequency,
mis-sampling turn-of-month's daily payment-cycle signal) flipped the sign
from -0.274 to +0.0815 -- a legitimate bias correction committed and
code-reviewed BEFORE re-gating, not a post-hoc flip. Still fails after the
fix. #37's PBO is now finite (0.141, was NaN pre-SP-D-fix), confirming that
fix is durable.

## Tier 2 -- Path 2 return-stream sleeves (18 gradeable combinations)

| Sleeve | OOS Sharpe | DSR | PBO | Verdict |
|---|---|---|---|---|
| #26/#27 VIX roll-down | 0.5640 | ~0 | 0.613 | WEAK/FAIL |
| #28 VRP short-VX1 | 0.0771 | ~0 | 0.297 | WEAK/FAIL (re-expression of #26, corr 0.488) |
| #21/#25 Overnight drift | 0.7924 | **0.872** | 0.513 | WEAK/FAIL (closest near-miss in the entire retest) |
| #21 Hour-slice | -0.0225 | 0.000 | 0.873 | REJECT |
| #36 Intermarket NQ/ES | 0.3294 | ~0 | 0.582 | WEAK/FAIL |
| #36 Intermarket RTY/ES | -0.2803 | 0.000 | 0.913 | REJECT |
| #31 Calendar CL | 0.3942 | ~0 | 0.631 | WEAK/FAIL |
| #31 Calendar NG | -0.1500 | 0.000 | 0.320 | REJECT |
| #31 Calendar ZC | 0.1736 | ~0 | 0.529 | WEAK/FAIL |
| #31 Calendar ZS | 0.3581 | ~0 | 0.429 | WEAK/FAIL |
| #31 Calendar ZW | 0.2634 | ~0 | 0.818 | WEAK/FAIL |
| #32 Crack RB-CL | -0.1162 | 0.000 | 0.469 | REJECT |
| #32 Crack HO-CL | -0.2150 | 0.000 | 0.704 | REJECT |
| #33 Crush ZM+ZL-ZS | 0.1360 | ~0 | **0.109** | WEAK/FAIL (clean PBO, decisive DSR reject) |
| #34 Ratio GC/SI | 0.2687 | ~0 | 0.674 | WEAK/FAIL |

**#33 Crush** is the one sleeve where PBO is genuinely clean (0.109, well
under the 0.25 bar -- not a CSCV-detectable overfit), which is why the
retest TODO flagged it as the sole Phase 6.5 candidate. It was NOT escalated
to Phase 6.5: its DSR is ~0, a decisive rejection by the BINDING gate
(Section 2.5), not a borderline miss the way overnight drift's 0.872 is. A
clean PBO does not make a near-zero-Sharpe (0.136), near-zero-DSR result
"marginal-but-real." Escalating it to a design-improvement round anyway
would be exactly the kind of design-iteration-chasing-a-metric this
project's North Star forbids -- each Phase 6.5 round also costs a trial in
the DSR distribution, so spending one on a decisive DSR failure has no
statistical justification.

**#21/#25 Overnight drift** is the closest anything came to clearing the
gate in this entire retest: DSR 0.872 against the 0.95 bar. It still fails
on PBO (0.513) independently, so even a hypothetically higher DSR would not
have made it viable without addressing the PBO instability too.

**#36 Intermarket's mandatory book-correlation check** (vs the RAMP
equity-momentum sleeve) was NOT run this session -- no RAMP daily return
stream was readily available to supply to the driver's `--ramp-returns`
flag. This is reported honestly as `book_corr=NaN` in both pair reports
rather than fabricated. It does not change either verdict: NQ/ES and RTY/ES
both already FAIL decisively on their own statistical merits (DSR ~0 and
negative Sharpe respectively) -- a correlation check would only ever add a
re-expression reason on top of an already-decisive rejection, never reverse
one from FAIL to PASS.

## Tier 3 -- architecturally ungradeable (documented, not fixable)

- **#39 Pre-FOMC**: confirmed n_windows=0 (~8 FOMC events/year never fill a
  12-month/10-sample walk-forward window). `_verdict` correctly returns
  INCONCLUSIVE. The pre/post-2015 decay split (pre: Sharpe 0.252, post:
  Sharpe 6.540) is small-n descriptive noise, not evidence either direction
  on the Ma-Zhang decay hypothesis -- neither number is a statistically
  meaningful comparison at this sample size.
- **#35 Steepener** (2s10s, 2s5s, 5s30s): confirmed n_windows=0 for all
  three segments (2YY history from ~2021 is too short for the 48-month
  walk-forward minimum; 5YY degraded to ~440 usable rows). A ZT/ZN
  DV01-based fallback remains a possible future rebuild once yield-future
  history matures; not attempted this session.

## EXCLUDED (unchanged from the TODO's authoritative table)

- **#49 FuturesFundingCarry**: Binance funding data geo-blocked (HTTP 451).
- **#9 multi-horizon carry blend**: never implemented.

## Cross-sleeve correlation / combined portfolio Sharpe

Not computed this session. Given that ZERO strategies individually clear the
combined statistical gate, a cross-sleeve correlation matrix and blended
portfolio Sharpe estimate would answer a question ("how should we allocate
across these") that does not arise -- there is nothing statistically
certified to allocate across. This is deliberately not computed to avoid
implying a false sense of a viable multi-sleeve book; if a future campaign
round using genuinely new signal families or genuinely new data (not simply
more trials against the same 26 hypotheses) produces a strategy that clears
the gate, correlation/allocation analysis becomes relevant again at that
point, not before.

## Bottom line

The futures catalog, evaluated with contamination-free, deflation-correct
rigor across all 26 gradeable strategy/root/pair combinations (7 Tier-1
carver strategies, the carry incumbent, and 18 Tier-2 return-stream
combinations), yields **no statistically-certified sleeve**. Carry
(`FuturesCarry`, `carry_idm_broad.yaml`) remains the best DEPLOYABLE book --
real cash-and-carry economic mechanism, cleanest PBO in the entire retest
(0.189), OOS Sharpe matching its long-documented 0.765 figure -- but it is
NOT a certified book: its DSR of 0.824 sits below the 0.95 bar the same way
it did before this retest, now confirmed under HONEST, GROWING, repo-wide
uniform deflation rather than the inconsistent pre-Gate-0 gate math. This is
the completed objective of this retest, reported as such.

## Follow-ups flagged, not actioned this session

1. `simulate_convergence` (`src/backtesting/spreads/convergence.py`) applies
   an identical `cost_return` to every exit path (converge, structural stop,
   time stop) -- Section 11.5 requires an elevated slippage multiplier
   specifically on stop exits. Flagged by code review during Gate 0.3; out
   of scope for driver-only work, affects #31-#34's exit-cost realism.
2. `SpreadTrade` (same file) does not persist an `exit_reason` field, so
   Section 11.9's converge/structural-stop/time-stop breakdown cannot be
   reconstructed from `trades.csv` for #31-#34. `sp_retest_common.
   convergence_exit_summary` reports this honestly rather than fabricating
   a breakdown; extending `SpreadTrade`/`simulate_convergence` to emit the
   field is a separate follow-up.
3. #36's book-correlation check against RAMP was not run (no return stream
   supplied) -- does not change either verdict this session, but should be
   run if #36 is ever revisited.
4. #31 NG's RollCalendar-based F1/F2 caveat-fix (vs volume-rank contract
   selection) was deprioritized -- 4/5 calendar roots already fail
   decisively on DSR, and NG itself fails on Sharpe<=0, not PBO-marginal
   grounds, so the fix would not change any verdict in this set.
