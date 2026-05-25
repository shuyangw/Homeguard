# Experiment 8: V14 Action Convergence Diagnostic -- Design

**Date**: 2026-05-24
**Status**: Proposed
**Branch**: v12-bear-to-cash (campaign continuation)
**Owner**: Shuyang
**Type**: Diagnostic (post-hoc analysis, no new variants, no new gates)
**Builds on**:
- `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md` (V14a/b/c TIER 4, Sharpe convergence within 0.011)
- `docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design.md` (V14 spec rev2)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (H4/H5 supported in original detector diagnostic)

## Position

The V14 factorial readiness produced a surprising-but-consistent result: V14a-cash (Sharpe 0.6146), V14b-spy (0.6035), and V14c-dampen (0.6131) landed within 0.011 of each other at 5 bps near_close. Under estimation noise of ~0.05-0.10 in Sharpe, these are statistically indistinguishable. The three actions span the meaningful action axis (full risk-off / full directional / partial reduction); their convergence implies that the action choice itself is not where the alpha lives.

The strongest reading is that **the consumer-layer fix is saturated**: with the soft-score trigger working, the choice of what to do on BEAR-soft entry has reached a ceiling. But the convergence is consistent with three different underlying mechanisms, each of which implies a different next intervention. This diagnostic disambiguates which mechanism is operative, so that the WS-3 detector intervention spec can be scoped against the actual constraint rather than against assumed ones.

The diagnostic uses existing V14 backtest output. No new backtests run, no new variants are proposed, no new gates fire, and no trials are added to `n_trials_project`. It is a notebook analysis whose output is a single decision: which WS-3 track to prioritize.

## Three mechanism hypotheses

The diagnostic tests three hypotheses, each with a falsifiable quantitative prediction. They are not mutually exclusive; the diagnostic reports the evidence for each separately.

**M1 -- Rare-events ceiling.** BEAR-soft mode fires too rarely under the registered tau pair (tau_in=0.556, tau_out=0.456) for any action to accumulate material P&L. The actions converge because they each apply to a small total fraction of trading days. Implication: the trigger is too restrictive; firing more often (via leading indicators or a lower tau) is the binding fix. Falsifiable prediction: total BEAR-soft days across 2017-2026 < 5% of the gated window, OR the variance of V14a-V11 P&L on BEAR-soft days is less than 30% of the variance on non-BEAR-soft days.

**M2 -- Action equivalence during real drawdowns.** BEAR-soft mode fires substantially, but the three actions are economically equivalent during the firing periods because SPY (V14b's action) and V11's average position (V14c's dampened action and V14a's avoided position) move similarly during real drawdown periods. The action axis doesn't matter because the targets are not differentiated. Implication: the trigger is correctly identifying drawdown periods but firing too late or with too much flicker; hysteresis on the existing inputs is the binding fix. Falsifiable prediction: during BEAR-soft periods, correlation(SPY daily return, V11-plan daily return) > 0.85.

**M3 -- Exit-timing failure.** BEAR-soft entries are well-timed but exits release too early because tau_out=0.456 is too high relative to the actual BEAR_score trajectory during drawdowns. All three actions get hurt symmetrically by re-entering the market at the wrong time -- V14a re-enters from cash, V14b re-allocates back from SPY, V14c restores V11's gross, all on the same day. Implication: the consumer-layer trigger is structurally close to correct; a smaller WS-3c.1 spec targeting exit logic could push V14 over the +0.10 lift bar without touching the detector. Falsifiable prediction: median lag from BEAR-soft exit date to subsequent SPY 20-day local minimum > 5 trading days, AND the post-exit window contributes negatively to V14a-V11 P&L on average.

## Decision matrix

The diagnostic returns a mechanism verdict per hypothesis (supported / refuted / inconclusive based on the quantitative predictions). The verdict combinations map to WS-3 track priorities as follows:

| M1 | M2 | M3 | Recommended next spec |
|---|---|---|---|
| supported | refuted | refuted | WS-3b (leading indicators) -- need detector to fire more often |
| refuted | supported | refuted | WS-3a (detector hysteresis) -- trigger is right but flickers |
| refuted | refuted | supported | WS-3c.1 (consumer exit logic) -- smaller scope, preserves V14 architecture |
| supported | supported | -- | WS-3d (detector replacement) -- augmentation hits multiple targets |
| -- | supported | supported | WS-3a + WS-3c.1 in parallel -- complementary fixes |
| supported | -- | supported | WS-3b primary, WS-3c.1 fallback |
| all three | -- | -- | WS-3d -- incremental fixes won't compound enough |
| none supported | inconclusive | inconclusive | WS-3d with expanded scope, OR halt WS-3 and pursue alternative strategies |

The mapping is pre-registered: the diagnostic's role is to identify which row applies, not to argue for any specific row. If the diagnostic's findings don't cleanly fall into one row (e.g., M1 supported but at the edge of the threshold), the synthesis report makes this explicit rather than picking a row to rationalize.

## Method

A single Jupyter notebook at `notebooks/research/v14_action_convergence_diagnostic.ipynb` performs six analyses against existing V14 backtest output. The notebook is parameterized on the readiness report's git SHA (6f55e37) so reruns against future readiness gates are mechanical.

### A1 -- Coverage statistics (tests M1)

Compute from the V14 backtest's per-day state:
- Total trading days in BEAR-soft mode across the V14-warm-start window.
- Number of distinct BEAR-soft events (contiguous mode-True runs).
- Distribution of event durations: count, median, P25, P75, max.
- Per-year breakdown of BEAR-soft day counts.

Output: a table and a histogram. **M1 support threshold**: total BEAR-soft days < 5% of gated window OR median event duration < 5 trading days.

### A2 -- Action-attribution decomposition (tests M1, M2)

For each variant V in {V14a, V14b, V14c}:
- Compute daily excess return (V - V11) over the gated window.
- Partition days into in_bear_soft_mode = True vs False.
- For each partition, compute sum, mean, std of (V - V11) daily excess.
- Compute the cumulative Sharpe contribution from each partition to V's annual Sharpe.

Output: a 3x2 table per variant (partition x summary stat) plus a stacked bar chart of cumulative excess returns by partition. **M1 support secondary**: BEAR-soft partition contributes < 30% of total V-V11 excess despite holding the strategy-differentiating positions. **M2 support primary**: BEAR-soft partition variance is comparable across V14a/b/c (within 20% of each other).

### A3 -- Per-event P&L by variant (tests M2)

For each BEAR-soft event:
- Compute V14a, V14b, V14c, V11 cumulative returns over the event window.
- Compute pairwise correlations of per-event returns across the three V14 variants.
- Rank events by absolute V14a-V11 P&L contribution.

Output: an event table (one row per BEAR-soft event) plus a correlation matrix. **M2 support**: cross-variant per-event return correlation > 0.85.

### A4 -- SPY vs V11-plan return correlation during BEAR-soft periods (tests M2 directly)

For each BEAR-soft event:
- Compute daily SPY return series and V11-plan daily return series over the event window.
- Compute correlation per event and across all events pooled.

Output: a correlation distribution plot. **M2 support primary**: pooled correlation(SPY return, V11-plan return) during BEAR-soft > 0.85.

### A5 -- Exit-timing analysis (tests M3)

For each BEAR-soft exit:
- Find the SPY local minimum within +/- 20 trading days of the exit date.
- Compute the lag in trading days from exit date to that minimum.
- Compute the 5-day, 10-day, 20-day post-exit V14a-V11 cumulative excess return.

Output: lag distribution plot, post-exit excess return distribution. **M3 support primary**: median exit-to-subsequent-SPY-low lag > 5 trading days AND mean 10-day post-exit V14a-V11 excess return < 0.

### A6 -- Counterfactual tau_out sweep (informational only)

For each tau_out in {0.20, 0.30, 0.40, 0.50}:
- Re-derive in_bear_soft_mode under the new tau_out (tau_in held at 0.556).
- Recompute V14a's Sharpe under this counterfactual.

Output: a Sharpe-vs-tau_out plot. **NOT a gate**. Purpose: visualize how much of V14a's headroom (or lack thereof) lives in the exit threshold. This output informs but does not determine WS-3c.1 sizing.

## Data inputs

All inputs come from existing artifacts; no new compute beyond the notebook itself.

| Input | Source | Notes |
|---|---|---|
| V14a/V14b/V14c daily returns | Readiness orchestrator output | Already persisted; cost grid @ 5 bps near_close used |
| V11 daily returns | Readiness orchestrator V11 reference | Same window as V14 variants |
| Per-day `in_bear_soft_mode` state | Orchestrator state log | Identical across V14a/b/c by construction |
| BEAR_score daily series | Diagnostic harness output | `diagnostics/regime/v0/labels.parquet` |
| SPY daily OHLC | Backtest panel input | Same source as V11 |
| V11 plan daily positions | Orchestrator state log | Needed for A4's plan-return computation |
| BEAR-soft event boundaries | Computed in-notebook | Derived from `in_bear_soft_mode` transitions |

If V11 plan daily positions are not persisted in the orchestrator state log (open question, see below), A4 falls back to V11's daily return as the V11-plan proxy. This is a minor analytical loss but does not block the diagnostic.

## What this diagnostic does NOT do

- **No new backtests**. All analyses use the 6f55e37 readiness output.
- **No new variants**. No code changes to `src/research/ramp_phase4/variants.py`.
- **No new gates**. PSR/DSR/PBO not recomputed; no Tier verdicts assigned.
- **No `n_trials_project` increment**. This is post-hoc decomposition of an existing variant family, not a new strategy.
- **No detector code changes**. `src/strategies/advanced/market_regime_detector.py` is read-only.
- **No track selection commitment**. The diagnostic returns a mechanism verdict; the WS-3 spec uses the verdict to select a track. Selection is committed in the WS-3 spec, not here.
- **No forward-OOS validation**. The diagnostic is on the same 2017-2026 window. Any track selected based on this diagnostic still requires forward OOS for any deployment.

## Risks and limitations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Mechanism verdicts are partial (e.g., M1 borderline-supported) and decision matrix doesn't cleanly map | Medium | Medium | Synthesis explicitly reports threshold values, not just verdicts. The decision matrix has explicit handling for "inconclusive" rows. If the data genuinely doesn't disambiguate, the verdict is "WS-3d with expanded scope" rather than a forced row pick. |
| V11 plan daily positions not persisted; A4 forced to use return proxy | Medium | Low | Documented; A4 falls back gracefully. Other tests (especially A3 and A5) are independent. |
| Diagnostic motivates a track that subsequently fails its own readiness | Medium | Medium | Acceptable. The diagnostic improves the prior on which track to try; it doesn't guarantee track success. WS-3 spec's own gates determine track viability. |
| Notebook analysis introduces a bug that changes verdicts | Low | High | Each analysis is independent and produces interpretable intermediate outputs. Sanity checks: total BEAR-soft days from A1 must match the count derived independently in A3 from event boundaries. |
| Multiple mechanisms appear supported, but the underlying truth is a single deeper mechanism the diagnostic doesn't measure | Medium | Medium | Synthesis flags this case. The WS-3d (detector replacement) track is the explicit fallback when the diagnostic doesn't cleanly identify a single mechanism. |
| Mechanisms are sample-specific to 2017-2026 and a future window would show different verdicts | Medium | Low | The diagnostic is honest about its window. Forward OOS validation is still required regardless of which track is selected. |

## Open questions to resolve before execution

1. **Is V11 plan daily position data persisted in the orchestrator state log?** Determines A4's primary vs fallback computation. Confirmable by inspecting `scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py` output directory. ~15 minutes to check.

2. **Is the BEAR-soft state log persisted with timestamps, or only as a derived signal from `in_bear_soft_mode` transitions in the variant?** A1 and A5 require event boundaries; if state isn't logged, the notebook reconstructs from per-day `bear_score >= tau_in` checks. The reconstruction matches the variant's state machine by construction but adds ~15 LOC to the notebook.

3. **Should the diagnostic run on `near_close` only, or also `one_day_lag`?** The factorial readiness reported both. Near_close is the production-relevant mode (live execution doesn't get one_day_lag treatment). Recommendation: near_close only for the primary mechanism analysis; one_day_lag as a sanity check that conclusions don't flip across timing modes.

## Output artifacts

- `notebooks/research/v14_action_convergence_diagnostic.ipynb` -- the analysis notebook
- `docs/reports/ramp/20260525_experiment8_action_convergence.md` -- synthesis report with mechanism verdicts and WS-3 track recommendation
- `docs/progress/20260525_RAMP_E8_DIAGNOSTIC.md` -- session log

## Sequencing

Execution is a single half-day session:

1. Resolve the 3 open questions (~30 min).
2. Build the notebook, running each analysis A1-A6 (~3 hours).
3. Write synthesis report with mechanism verdicts and track recommendation (~1 hour).
4. Update WS-3 spec (separate document) to reflect the verdict.

This runs in parallel with the WS-3 detector intervention spec drafting and does not block the V11 paper validation timer.

## Success criteria

The diagnostic succeeds if:

1. Each of M1/M2/M3 has a verdict (supported / refuted / inconclusive) backed by specific quantitative evidence against its pre-registered threshold.
2. The decision matrix produces a clear WS-3 track recommendation, OR the synthesis explicitly documents that the verdicts are inconclusive and falls through to WS-3d.
3. The notebook is executable end-to-end without manual intervention; reruns against future V14 family readiness reports require only the SHA change.

The diagnostic fails if it produces findings that are simultaneously consistent with all three mechanisms with no discriminating power -- in which case the appropriate response is to acknowledge the data limitation and proceed with WS-3d (detector replacement) as the most general intervention rather than picking a track on weak evidence.
