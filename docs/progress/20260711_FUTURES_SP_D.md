# Futures SP-D + VRP Finalization - 2026-07-11

## Summary
Finalized the volatility-risk-premium sleeve of the futures campaign and, in the
process, discovered and fixed that the walk-forward gate's DSR had never actually
deflated. Once the DSR is honestly deflated against the 40-trial campaign, NOTHING in
the futures book clears DSR >= 0.95 -- including the incumbent carry. The VRP family
is closed as a negative result: #26 (VIX roll-down) fails deflation with an 81%
drawdown tail, and #28 (realized-vs-implied VRP) fails and is a degenerate
re-expression of #26. Built via subagent-driven development in an isolated worktree;
merged to main (52ce408) as a clean fast-forward.

## Changes Made
- **Honest DSR deflation** (`src/backtesting/walkforward_common.py`): the gate was
  doubly non-deflating -- `TRIAL_COUNT_PARAMETER_FREE = 1` AND a single-element
  trial-Sharpe list (which makes `expected_max_sharpe` return 0). Replaced with
  `CAMPAIGN_CUMULATIVE_TRIALS = 40` and `CAMPAIGN_TRIAL_SHARPES` (29 real OOS Sharpes
  from the SP-A/E/B/C ledgers, var 0.112) threaded into the `dsr()` call. Yields
  SR_zero = 0.733 (the 40-trial expected-max benchmark). Scope: the `gate_return_stream`
  path only (VIX/SP-C/#26/#28); the carver/fx/session/blend paths got the count
  relabeled but still pass single-element lists (documented follow-up).
- **PBO NaN fix** (`_compute_pbo`): raised the window-drop threshold to 2*s (32 rows)
  so a short trailing window no longer NaN-s the whole statistic.
- **#26 audit** (`vix_rolldown_eval.py`): `subperiod_audit` (per-year Sharpe, skew,
  worst day, max drawdown, Volmageddon/COVID month P&L).
- **`src/backtesting/vol/` module**: `option_symbol.py` (parser with ref_year year
  resolution), `atm_iv.py` (Black-76 ATM-IV from ES/NQ option prints, underlying =
  RAW front-future close, + VIX-validation gate), `har_rv.py` (Corsi HAR forecast,
  within-session RV), `vrp_strategy.py` (VRP = IV - HAR, percentile-sized short-VX1
  stream + re-expression check vs #26).

## Verdicts (deflated; SR_zero = 0.733; ledger docs/strategies/research/20260711_FUTURES_SP_D_TRIALS.md)
- **#26 VIX roll-down**: OOS Sharpe 0.564, DSR 8.88e-06, PBO 0.613 -> FAIL. Tail audit:
  skew -2.22, worst day -47.9%, max DD -81.1%, Volmageddon -12.6%.
- **#28 VRP (short-VX1)**: IV validated (corr 0.828 vs VIX). OOS Sharpe 0.055, DSR
  1.5e-124, PBO 0.363 -> FAIL. Re-expression: corr 0.479 to #26, marginal Sharpe 0.015
  -> a degenerate re-expression of #26, no distinct edge.
- **carry (incumbent, indicative)**: gated FuturesCarry 2010-2026 equity -> OOS Sharpe
  0.588, DSR 5.4e-14 (FAIL), PBO 0.093 (PASS, not overfit). Even carry's edge sits
  at/below SR_zero 0.733.

## Commits
Merged to main as fast-forward, 12 commits `d53c4f7..52ce408`. Key:
- `d53c4f7` honest trial count; `8a23dd7` PBO sub-s drop; `0602ea7` DSR deflates via
  trial-Sharpe distribution + PBO 2s
- `05e1737` #26 audit + re-gate
- `17b500a` option-symbol parser; `c372f75`/`f5896bd` Black-76 ATM-IV + raw-underlying fix
- `6dc644f`/`d7f1c40` HAR RV + overnight-gap fix
- `1eeac6b` #28 VRP + re-expression
- `fcd461c`/`52ce408` trial ledger + deflation-scope doc fix

## Known Issues / Remaining Work
- **Thread `CAMPAIGN_TRIAL_SHARPES` into the other four gate paths** (carver/fx/session/
  blend) so a future run doesn't emit an undeflated DSR under a trial_count=40 label.
  Out of this sub-project's scope; the spec limited re-deflation to carry/#26/#28.
- Add #28 to `CAMPAIGN_TRIAL_SHARPES` once it is a historical trial (not circular).
- 4/933 ES days show sub-1% IV (thin/stale prints); median 16.6% sane, not gate-relevant.
- HAR leak-detection test is low-power (degenerate flat-training); tighten to assert
  `fc.iloc[k-1]`.
- N=40 exactly meets the test's >=40 floor; optionally bank a buffer via baselines.
- Campaign status: SP-A + SP-E + SP-B + SP-C + SP-D done (all 5). The remaining owed
  items (VIX #26 deflation -- now DONE via SP-D; SP-B2 intraday remainder) are the only
  open threads. The campaign's honest conclusion: no futures sleeve clears the deflated
  gate; carry remains the best DEPLOYABLE book (real mechanism, passes PBO) but does not
  clear the multiple-testing-honest DSR bar.

## Validation
- Subagent-driven: per-task reviews caught real defects offline tests missed -- most
  importantly the DSR non-deflation (the count fix alone was inert), a HIGH-severity
  ratio-adjusted-vs-raw underlying bug in the IV extractor, and an overnight-gap
  contamination in the HAR RV. All fixed + re-reviewed.
- Final whole-branch review (opus): MERGE-READY, no Critical; one Important (deflation
  scope) resolved by correcting the ledger claim.
- Tests: 47 SP-D + touched-suite tests pass (VIX/session/spreads suites green -> the
  shared-gate change is behavior-preserving where a valid value existed). #26 and #28
  persisted returns.csv + gate.json.
