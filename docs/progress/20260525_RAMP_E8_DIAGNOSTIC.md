# Experiment 8 -- V14 Action Convergence Diagnostic -- Session Log (2026-05-25)

## Summary

Post-hoc decomposition of the V14 factorial readiness convergence (V14a/b/c within 0.011 Sharpe at 5 bps near_close). Tested three pre-registered mechanism hypotheses (M1 rare-events ceiling, M2 action equivalence, M3 exit-timing failure) via six analyses. Verdict: **M1 inconclusive, M2 refuted, M3 refuted -> WS-3d (detector replacement) with expanded scope**. The dominant diagnostic finding is that BEAR-soft entries fire ~8 trading days AFTER the SPY drawdown trough on the median, so cash periods cover the recovery rather than the crash. This is a fourth mechanism the pre-registered hypotheses did not isolate.

## What ran

- 4 V14 / V11 backtests at 5 bps near_close (V11, V14a-cash, V14b-spy, V14c-dampen).
- 2 sanity-check backtests at 5 bps one_day_lag (V11, V14a-cash) to confirm conclusions don't flip across timing modes.
- 6 analyses (A1 coverage, A2 attribution, A3 per-event, A4 SPY/V11 corr, A5 exit timing, A6 tau_out sweep).
- Independent cross-check: Schmitt-trigger reconstruction from `score_BEAR` series overlaps engine `regime == 'BEAR_SOFT_CASH'` at 100.0% (366 == 366 days).
- Wall-clock: 6.63 min on local fintech env.
- No new variants, no new gates, no n_trials_project increment.

## Headline verdicts

| Mechanism | Verdict | Key numbers |
|---|:---:|---|
| M1 rare events | inconclusive | 366 BEAR-soft days = 15.54% of gated window (NOT rare); median duration 9 days (NOT short); but BEAR-soft contributes only 9.75% of V14a-V11 excess (supports rarity-on-impact view). 1-of-3 signals. |
| M2 action equivalence | refuted | Cross-variant per-event correlation V14a/V14b = 0.5092, V14a/V14c = 0.6359, V14b/V14c = 0.8573 (only V14b/V14c passes 0.85); pooled corr(SPY, V11) = 0.7927 (below 0.85). Both clauses fail. |
| M3 exit-timing failure | refuted | Median exit-to-SPY-low lag = -8 days (SPY low BEFORE exit; M3 predicted >5); mean 10d post-exit V14a-V11 excess = +0.16% (positive; M3 predicted <0). Both clauses fail in the opposite direction. |

**WS-3 track recommended**: WS-3d (detector replacement) with expanded scope.
**Fallback**: WS-3b (leading indicators) as the cheapest single-mechanism path if WS-3d is rejected on cost grounds.
**Alternative**: halt WS-3, accept V11's 0.5306 Sharpe as the RAMP target, redirect to orthogonal alpha sources.

## Key findings

1. **The V14 trigger fires AFTER the SPY trough on most events.** Median exit-to-SPY-low lag of -8 trading days (negative = SPY low was BEFORE the BEAR-soft exit) means the entry was already late and the cash window covers the recovery, not the crash. This mirrors V12's `gap_days = -3.42` from the 2026-05-24 V12 readiness diagnostic but is even more pronounced under V14's soft-score trigger.

2. **V14 actions are NOT economically equivalent per-event.** A3 shows V14a (cash) and V14b (SPY) correlate only 0.51 over the 25 BEAR-soft events; V14b and V14c (the two directional alternatives) correlate 0.86. Cash diverges meaningfully; the full-window Sharpe convergence is a noise-averaging artifact across many short events, not a true mechanism convergence.

3. **A6 tau_out sweep shows tau_out=0.4556 is at or near the local optimum.** Counterfactual Sharpes are monotone increasing in tau_out from 0.20 (0.4426) to 0.4556 (0.6146). There is no easy "extend cash periods" improvement.

4. **Coverage is moderate, not rare.** 15.54% of trading days in BEAR-soft mode across 2017-2026 -- materially higher than M1's 5% threshold. The BEAR-soft window concentrates in 2018 (86d), 2020 (40d), 2022 (141d), and 2025 (39d).

5. **Schmitt reconstruction matches engine perfectly.** The engine's `state.in_bear_soft_mode` derived from the variant's plan_fn matches an independently-computed Schmitt trigger applied to the raw `score_BEAR` series at 100.0% overlap (366 = 366 days). Diagnostic integrity confirmed.

## Decisions

- **WS-3d primary recommendation** based on the pre-registered decision matrix's catch-all branch ("none supported -> WS-3d with expanded scope"). The deeper interpretation (trigger fires on recovery, not drawdown) reinforces WS-3d as the only path that can address a structurally late detector -- WS-3a/b/c.1 each target a single failure mode but cannot fix detector timing.
- **Alternative path documented**: halt WS-3 and pursue alternative strategies (the spec's explicit fallback when the diagnostic doesn't disambiguate). This deserves a separate discussion: the cost of WS-3d vs the orthogonal-alpha opportunity cost.
- **V11 paper validation continues** independently (A7 counter on EC2). E8 does not change V11's path.

## Commits this session

- `<C1>` feat(diagnostics): E8 V14 action convergence -- M1/M2/M3 mechanism verdicts
- `<C2>` report(ramp): E8 V14 action convergence -- WS-3 track recommendation

(Replace placeholders with actual hashes after commit.)

## Branch state

Local `v12-bear-to-cash` at `<HEAD-after-commit>`. Local only; not pushed.

## Next session candidates (priority order)

1. **WS-3 track spec** -- formalize the WS-3d (detector replacement) spec with the expanded scope mandated by E8's verdict. Alternatively, write a halt-WS-3 spec that lays out the alternative-strategy pivot.
2. **WS-3b mini-spec** -- if WS-3d cost is unacceptable, scope WS-3b (leading indicators) as the fallback. E8 suggests a specific sub-problem ("make detector fire ~8 trading days earlier than the SPY trough on the median"), which is more concrete than the original WS-3b draft.
3. **Monitor V11 paper validation** -- A7 counter on EC2 continues regardless.
