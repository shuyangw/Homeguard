# RAMP Regime-Detector Campaign Closure -- 2026-05-24 to 2026-06-02

## Summary

The 2026-05-24 RAMP regime-detector research campaign closes with a definitive
negative verdict on the regime-aware-RAMP research line under the current
detector paradigm. Nine experiments (E1-E8 + V14 factorial readiness) and three
WS-3d Gate 1 retraining attempts produced converging evidence that:

1. The v0 detector's lag is structural to its supervised paradigm at the V14a
   consumer threshold (tau_in = 0.5556).
2. Consumer-layer fixes -- argmax-BEAR consumption in V12 (TIER 3), V12c
   (TIER 4), and V13-bear-invert (TIER 4); soft-score Schmitt-trigger in
   V14a/b/c (all TIER 4) -- cannot recover days the detector itself missed at
   the V14a-style consumer threshold.
3. Replacing the detector with LightGBM on leading indicators (WS-3d) does NOT
   reduce lag at the consumer-relevant threshold (tau = 0.30). v0's rule-based
   score is already near-optimal at that threshold because SMA-crossing
   criteria fire on day 0-2 of every drawdown (v0 H5 median = 2.0 days at
   Schmitt-tau-0.30).
4. Training the WS-3d model on a leading target (G1_BEAR.shift(-10))
   marginally improved its lag (19.5d -> 11.0d at Schmitt-tau-0.30) but still
   failed the gating threshold of <= 5d and missed 1/5 G4 events entirely
   (2025_dec_drawdown).

The campaign produced permanent infrastructure (8 variant codes, 4
leading-indicator acquirers, soft-score replay, regime diagnostic,
action-convergence diagnostic, V20 LightGBM detector class) and a clear
theoretical understanding of why the consumer-layer ceiling exists at
roughly +0.08 Sharpe lift vs V11 at 5 bps near_close. Further incremental
work on this line is unlikely to produce TIER 1 outcomes.

Per user direction (2026-05-24), WS-3d is officially HALTED. The next
research direction is deferred to a fresh session.

## Campaign timeline

| Date | Phase / Experiment | Output | Verdict (one-line) |
|---|---|---|---|
| 2026-05-23 | V11 readiness (pre-campaign baseline) | `20260523_phase4_v11_readiness.md` | PARTIAL Tier 3; PSR 0.944/DSR 0.811 fail; paper-deployed |
| 2026-05-23 | Regime detector diagnostic (H1-H5) | `20260523_regime_detector_diagnostic.md` | H4 + H5 SUPPORTED; 14d SMA lag baseline of record |
| 2026-05-24 | V12 readiness (BEAR -> cash) | `20260524_phase4_v12_readiness.md` | Tier 3; gap_days = -3.42 (detector late vs trough) |
| 2026-05-24 | E2 UNPREDICTABLE hand-inspection | `20260525_experiment2_unpredictable_inspection.md` | AMBIGUOUS (53.6% top-3 attribution; COVID-dominant) |
| 2026-05-24 | E3 soft-score extraction | `20260525_experiment3_soft_scores.md` | WS-3c verdict (median argmax_lag 24d at tau=0.30) |
| 2026-05-24 | E4 lag-asymmetry decomposition | `20260525_experiment4_lag_asymmetry.md` | DIFFUSE (38.1% transition share) |
| 2026-05-24 | E1 V13-bear-invert readiness | `20260525_phase4_v13_readiness.md` | TIER 4 (BEAR-as-buy spurious; PBO 0.629) |
| 2026-05-24 | E5 OMR cross-check | `20260526_experiment5_omr_cross_check.md` | AMBIGUOUS (OMR screens BEAR/UNPREDICTABLE) |
| 2026-05-24 | E6 V12c readiness | `20260526_phase4_v12c_readiness.md` | TIER 4 (PBO 0.7085; DSR 0.834) |
| 2026-05-24 | V14 factorial readiness (a/b/c) | `20260526_phase4_v14_factorial_readiness.md` | All TIER 4 (PBO 0.9528; +0.08 lift below +0.10 bar) |
| 2026-05-25 | E8 action convergence | `20260525_experiment8_action_convergence.md` | (M1=inc, M2=ref, M3=ref) + lag-structural finding -> WS-3d |
| 2026-05-25 | WS-3d full implementation spec | `2026-05-25-ws3d-detector-replacement-design.md` | 4 amendments + 5 pre-commitments; trial-chain reset |
| 2026-05-25 | WS-3d Gate 0: data ingestion | `src/data/leading_indicators/` | 4 acquirers landed (VIX term, HY proxy, breadth, SKEW) |
| 2026-06-01 | WS-3d Gate 1 round 1 (G1_BEAR target) | `20260601_ws3d_regime_diagnostic_rerun.md` | BLOCKED (H5 lag 21d vs v0 14d at argmax-0.5) |
| 2026-06-01 | WS-3d Gate 1 round 2 (Amendment 6, Schmitt-tau-0.30) | `20260601_ws3d_regime_diagnostic_rerun.md` | BLOCKED (H5 lag 19.5d vs v0 2.0d) |
| 2026-06-02 | WS-3d Gate 1 round 3 (G1_BEAR.shift(-10) leading target) | `20260602_ws3d_diagnostic_g1_shift10.md` | BLOCKED (H5 lag 11.0d, 4/5 G4 events fired) |
| 2026-06-02 | WS-3d HALT decision | (this doc) | Closure; V20+ family TIER 4; v0 detector preserved |

## Verdicts table

| Experiment / Variant | Type | Window | Verdict |
|---|---|---|---|
| V11 readiness (prior) | Variant readiness | 2017-2026 | PARTIAL / Tier 3 (PSR 0.944 / DSR 0.811 fail; paper-deployed 2026-05-23) |
| V12 readiness | Variant readiness | 2017-2026 | Tier 3 (PSR 0.788, DSR 0.542, PBO 0.393 PASS; gap_days = -3.42) |
| E2 UNPREDICTABLE inspection | Diagnostic | 2017-2026 | AMBIGUOUS (53.6% top-3 share; COVID-dominant 43.7%) |
| E3 soft-score extraction | Diagnostic | 2017-2026 | WS-3c (median argmax_lag 24d at tau=0.30; r= -0.213 at h=10d) |
| E4 lag asymmetry | Diagnostic | 2017-2026 | DIFFUSE (38.1% transition share; gap = +0.397 Sharpe lag - near_close) |
| E5 OMR cross-check | Diagnostic | 2018-2024 | AMBIGUOUS (OMR Bayesian screen filters BEAR/UNPREDICTABLE entries) |
| E1 V13-bear-invert | Variant readiness | 2017-2026 | TIER 4 (PBO 0.629; Sharpe 0.400 vs V11 0.528) |
| E6 V12c readiness | Variant readiness | 2017-2026 | TIER 4 (PBO 0.7085; DSR 0.8337 at n_trials=23) |
| V14 factorial (a/b/c) | Variant readiness | 2017-2026 | All TIER 4 (PBO 0.9528; max delta vs V11 = +0.087) |
| E8 action convergence | Diagnostic | 2017-2026 | (M1=inc, M2=ref, M3=ref) + lag-structural finding; median exit-to-SPY-low = -8d |
| WS-3d Gate 1 round 1 | Detector replacement | 2017-2026 | BLOCKED (H5 lag 21d at argmax-0.5 vs v0 14d; -50%) |
| WS-3d Gate 1 round 2 (Amendment 6) | Detector replacement | 2017-2026 | BLOCKED (H5 lag 19.5d at Schmitt-tau-0.30 vs v0 2.0d; -875%) |
| WS-3d Gate 1 round 3 (leading target) | Detector replacement | 2017-2026 | BLOCKED (H5 lag 11.0d at Schmitt-tau-0.30; 4/5 events fired) |

## The lag-structural finding (the campaign's most important result)

The campaign accumulated three independent measurements of v0 detector lag,
each on a different consumption pattern and methodology:

| Measurement | Value | Source |
|---|---|---|
| V12 onset alignment, mean gap_days | -3.42 trading days (59 BEAR onsets) | `20260524_phase4_v12_readiness.md` |
| Original H5 diagnostic (argmax-at-0.5, G4 basis) | 14 days median SMA lag | `20260523_regime_detector_diagnostic.md` |
| E8 BEAR-soft exit-to-SPY-low lag | -8 trading days median | `20260525_experiment8_action_convergence.md` |

All three pointed at the same conclusion: the v0 detector fires AFTER the SPY
drawdown trough on average. WS-3d was designed as the evidence-driven response
to this finding -- a fresh detector on leading indicators, with a trial-chain
reset justified on three independent grounds (new inputs, new scoring, new
architecture).

### The Amendment 6 reframing (round 2)

WS-3d Gate 1 round 1 used the legacy argmax-at-0.5 evaluation and produced
v1 H5 lag = 21d vs v0 14d (BLOCKED). Round 2 introduced Amendment 6: change
Gate 1's binding metric to Schmitt-trigger first-crossing at tau_eval = 0.30,
matching V20-rd-bear-cash's consumer pattern. This produced two unexpected
numbers:

- v0 H5 median lag at Schmitt-tau-0.30 = **2.0 days** (5/5 G4 events fired)
- v1 H5 median lag at Schmitt-tau-0.30 = **19.5 days** (4/5 G4 events fired)

The 2.0d v0 result was the surprise. v0's rule-based `score_BEAR` is a
fraction-of-criteria-met (above_20 / above_50 / above_200 / vix_percentile)
that flips on the first SMA-crossing of a drawdown. At tau = 0.30, only one
of the four binary criteria need fire; this happens on day 1-2 of every
drawdown event. v1's LightGBM `bear_proba` is a smooth posterior that takes
weeks to cross 0.30.

This reframed the original "v0 detector is structurally late by ~8d"
finding: the lag is real but tau-specific. At V14a's tau_in = 0.5556 (G1_BEAR
median), v0 takes ~8d to cross. At tau = 0.30, v0 takes 2d. The V14a
threshold was the binding constraint, not a fundamental detector property.

For any future re-analysis: lower thresholds eliminate the lag at v0 but
trade off specificity (more false positives). The mechanism that V14
discovered was not "the detector is late" but "V14's consumer threshold is
high enough that v0's lag-to-cross becomes binding." This is a
methodologically important post-hoc clarification documented here so future
researchers do not re-derive the false framing.

### The round-3 falsification

Round 3 retrained the LightGBM model on `G1_BEAR.shift(-10)` so the model
predicts whether G1_BEAR will fire 10 days in the future. Pre-registered
round-3 criteria (set before observing the round-3 model):

- PASS iff: v1 median H5 lag at Schmitt-tau-0.30 (G4 basis) <= 5d AND v1
  fires on >= 5/5 G4 drawdown events.
- FAIL otherwise.

Result: median = 11.0d, 4/5 events fired (2025_dec_drawdown missed).
Both criteria failed.

Per-event G4 detail (round 3 v1 vs v0 at Schmitt-tau-0.30):

- Q4_2018_selloff: v0=2, v1=8 (worse by 6d)
- COVID_crash: v0=2, v1=5 (worse by 3d)
- 2022_bear_market: v0=14, v1=14 (tie)
- 2025_tariff_drawdown: v0=2, v1=19 (worse by 17d)
- 2025_dec_drawdown: v0=36, v1=did not fire

The leading-target hypothesis was the cleanest test of "can supervised
learning with leading indicators anticipate the drawdown?" The answer was no
under the Schmitt-tau-0.30 binding metric.

## Why WS-3d failed

The campaign's converging evidence isolates the failure mechanism:

1. **G1_BEAR is a confirmation label by construction** -- it fires after SPY
   has declined >= 10% from its trailing 252-day peak. A LightGBM model
   trained to predict G1_BEAR produces P(BEAR | indicators) that crosses
   0.5 around the time G1_BEAR itself fires. The argmax label tracks
   confirmation rather than precedes it.

2. **The probability trace IS smooth, not threshold-shaped.** Even when
   v1's `bear_proba` reaches 0.25-0.30 before reaching 0.5, the function
   is smooth and rises gradually. v0's rule-based `score_BEAR` is a
   fraction-of-criteria-met -- each criterion is binary and independent --
   so the score flips abruptly on the first SMA-crossing. At tau = 0.30, v0
   wins because one of four criteria needs to fire vs LightGBM's joint
   posterior smoothly accumulating evidence.

3. **The leading target (G1_BEAR.shift(-10)) helps but doesn't suffice.**
   Round 3's model learned mostly from `skew_close` (importance 69),
   `hy_proxy_ratio` (66), and `vix_percentile` (50) -- the leading
   indicators DO carry signal. But the signal is encoded into the
   posterior smoothly, not into threshold rules. The supervised paradigm
   cannot materially anticipate a label by more than the label's implicit
   lead time, regardless of input set.

4. **The fundamental tension.** Supervised learning on confirmation /
   near-confirmation labels cannot anticipate the label by more than its
   implicit lead time. The G1_BEAR.shift(-10) target shifts the
   confirmation by 10 days but compresses class imbalance and makes the
   pattern harder to learn (the round-3 CV logloss was 0.38289, the round-1
   model was 0.347 -- the leading target was actually harder to learn).

Any future detector improvement work would need to abandon the supervised
paradigm at the consumer threshold v0 already dominates. Options:

- HMM with persistence built into the transition matrix (mode-collapse
  risk under EM, not stable across rolling fits)
- A hand-crafted threshold ensemble that mirrors v0's rule-based design
  but adds leading-indicator criteria (would NOT be a trial-chain reset
  on the trial-budget logic)
- A fundamentally different ground-truth labeler -- something that fires
  BEFORE drawdowns rather than during/after them. This is a research
  question outside the supervised-classifier paradigm.

## Permanent infrastructure produced

The campaign produced infrastructure that survives the closure and remains
available for future regime-aware research, OMR cross-strategy work, or
diagnostic continuity:

**Variant codes (8 entries in REGISTRY)**:
- V01, V04, V05, V06 (Phase 4 wave 1 baselines)
- V11 (production paper, deployed 2026-05-23)
- V12 (BEAR -> cash, Tier 3)
- V12c (BEAR + UNPREDICTABLE -> cash, TIER 4)
- V13-bear-invert (BEAR -> 100% SPY, TIER 4)
- V14a-soft-bear-cash / V14b-soft-bear-spy / V14c-soft-bear-dampen (Schmitt-trigger BEAR_score consumer, all TIER 4)

**Engine + state machine**:
- `engine.py::_engine_pre_variant_update_soft_bear` (Schmitt-trigger BEAR-soft state engine)
- `HarnessState.in_bear_soft_mode: bool` (V14 engine state field)
- `src/research/ramp_phase4/plans.py::_SentinelPlan` + `PLAN_CASH_BEAR_SOFT` (V14 no-exposure marker)
- `MarketRegimeDetector.last_classification_timestamp` (V14 freshness assertion)

**Readiness orchestrators (5 distinct gates)**:
- V11 readiness
- V12 readiness
- V13 readiness
- V12c readiness
- V14 factorial readiness (3 variants in one gate; DSR n_trials = 36)

**Leading-indicator acquirers (WS-3d Gate 0)**:
- `src/data/leading_indicators/` -- 4 acquirers: VIX/VIX3M term-structure
  ratio, HY OAS proxy (HYG/IEF), NYSE breadth (% S&P 500 above 50-day MA),
  CBOE SKEW

**WS-3d detector pipeline**:
- `src/strategies/advanced/market_regime_detector_v1.py::MarketRegimeDetectorV1`
  (LightGBM detector class, schema-compatible with v0)
- `scripts/diagnostics/train_detector_v1.py` (CPCV training pipeline; supports
  `--target {g1_bear, g1_shift10, g2_bear}`)
- `scripts/diagnostics/regime_detector_v1_replay.py` (day-by-day replay)
- `scripts/diagnostics/regime_detector_v1_diagnostic.py` (H1-H5 rerun)
- 11 detector unit tests

**Diagnostic infrastructure**:
- `scripts/diagnostics/regime_score_replay.py` (E3 soft-score replay)
- `notebooks/research/experiment8_v14_action_convergence.py` (E8 mechanism diagnostic)
- v0 replays: `diagnostics/regime/v0/labels.parquet`,
  `diagnostics/regime/v0_scores/labels.parquet`
- v1 replays: `diagnostics/regime/v1/labels.parquet`,
  `diagnostics/regime/v1_g1_shift10/labels.parquet`

**Methodological frameworks**:
- V14 spec rev2 honesty discipline (sensitivity panels NOT gate-influencing)
- DSR n_trials_project audit (36-trial v0 chain; clean reset rule for new families)
- CPCV via `src/backtesting/validation/cpcv.py` (already existed; first
  detector-side use)
- V20+ trial-chain reset justification (new inputs + new scoring + new architecture)
- Amendment 6 framework: evaluate at consumer threshold, not at argmax-at-0.5

**Documentation outputs**:
- 12+ readiness / diagnostic reports under `docs/reports/ramp/`
- 8+ session logs under `docs/progress/`
- WS-3 track-conditional intervention spec
  (`2026-05-24-ws3-detector-intervention-design.md`)
- WS-3d full implementation spec with 6 amendments
  (`2026-05-25-ws3d-detector-replacement-design.md`)
- E8 action convergence spec
  (`2026-05-24-experiment8-action-convergence-diagnostic.md`)
- V14 factorial spec rev2
  (`2026-05-24-v14-soft-bear-factorial-design.md`)
- RAMP_VARIANTS canonical glossary updates (V12-V14, V20+ reserved)

These artifacts remain in the codebase for any future regime-aware-RAMP
work or for OMR / cross-strategy research that wants to consume the same
detector outputs.

## Decisions

- **WS-3d halted.** V20+ family closed as TIER 4 verdict; variant code
  preserved in REGISTRY for diagnostic continuity.
- **v0 detector preserved.** V11/V12/V12c/V13/V14a/b/c continue consuming it;
  no migration.
- **V11 paper validation continues independently.** A7 counter on EC2
  unaffected.
- **Next research direction deferred.** User to choose in fresh session from:
  universe expansion, RAMP-OMR portfolio construction, alternative signal
  stacks, Darwinex-inspired FX, or close-and-ship-V11.

## Commits this campaign (chronological)

28 commits between `871db66..HEAD` on branch `v12-bear-to-cash`:

- `faf7abe` feat(diagnostics): pre-register V14 tau constants from G1_BEAR median
- `983ac61` fix(diagnostics): V14 tau constants -- forward-slash paths + git-log guard
- `d26aa69` feat(detector): last_classification_timestamp field for V14 freshness assertion
- `c1f467f` feat(plans): _SentinelPlan class for V14 no-exposure marker
- `108de8a` feat(engine): V14 Schmitt-trigger state + _SentinelPlan dispatch
- `9674bd5` feat(config): V14 tau / dampen fields + JSON loader + predicate validation
- `8aae219` feat(variants): V14a-soft-bear-cash via Schmitt-trigger BEAR_score
- `9078633` fix(variants): V14a freshness assertion must not be swallowed
- `7844f1d` feat(variants): V14b-soft-bear-spy via Schmitt-trigger BEAR_score
- `1bb4e8a` feat(variants): V14c-soft-bear-dampen via Schmitt-trigger BEAR_score
- `6f55e37` feat(orchestrator): V14 factorial readiness gate -- 3 variants, DSR n_trials=36
- `dd2e37b` report(ramp): V14 factorial readiness -- WS-3c soft-score verdict TIER 4
- `1060efa` docs(progress): V14 factorial readiness session log + RAMP_VARIANTS V14 sections
- `9f2f86f` spec(ws3): E8 diagnostic + WS-3 track-conditional intervention design
- `7a52279` feat(diagnostics): E8 V14 action convergence -- M1/M2/M3 mechanism verdicts
- `afee3a8` report(ramp): E8 V14 action convergence -- WS-3 track recommendation
- `36bf435` spec(ws3): add lag-structural decision tree row -- WS-3d evidence-driven
- `9e35246` spec(ws3d): full implementation spec + V20+ reservation + track-selection log
- `f100b93` feat(data): leading_indicators package -- VIX term, HY OAS, breadth, SKEW acquirers
- `80d2a8d` feat(data): WS-3d Amendment 5 -- HYG/IEF proxy substitutes FRED HY OAS
- `e192f5e` feat(detector): MarketRegimeDetectorV1 -- LightGBM on leading indicators (WS-3d)
- `f3c11c9` feat(diagnostics): WS-3d detector replay + H1-H5 diagnostic rerun
- `bb97e56` docs(progress): WS-3d Gate 1 session log -- BLOCKED at diagnostic gate
- `1b24b7b` spec(ws3d): Amendment 6 -- Gate 1 H5 metric changes to Schmitt-tau-0.30 first-crossing
- `146e5d2` diag(ws3d): Gate 1 H5 rerun under Amendment 6 -- STILL BLOCKED at Schmitt-tau-0.30
- `8c476f0` feat(detector): WS-3d Gate 1 round 3 -- leading target G1_BEAR.shift(-10) retrain
- `e3c030d` diag(ws3d): Gate 1 round 3 verdict at G1_BEAR.shift(-10) -- BLOCKED at Schmitt-tau-0.30
- `c1a68b1` docs(progress): fill in commit hashes for WS-3d Gate 1 round 3 session log

## Cross-experiment context for future researchers

The campaign's accumulated evidence offers a few takeaways for any future
regime-aware-RAMP work:

- **The consumer-layer ceiling is roughly +0.08 Sharpe vs V11 at 5 bps
  near_close.** V14a/b/c all converged within 0.011 Sharpe and all came in
  at +0.075 to +0.087 over V11. None cleared the +0.10 TIER 1 lift bar.

- **The +0.10 TIER 1 lift bar with DSR n_trials = 36 is genuinely tight.**
  Future detector variants need either (a) a clean trial-chain reset
  justified on new inputs + new scoring + new architecture (which is what
  WS-3d attempted and what V20+ would have continued), OR (b) a significantly
  larger Sharpe lift than +0.08 to overcome the multi-trial penalty.

- **The v0 detector's rule-based score at Schmitt-tau-0.30 is essentially
  a 1-2 day SMA-crossing alarm.** This is hard to beat with ML on the same
  input class, and any supervised model trained on a near-confirmation label
  inherits the label's implicit lead time as its ceiling.

- **The leading-indicator features have signal but it's not threshold-firing
  signal.** Round-3 feature importances put `skew_close` (69), `hy_proxy_ratio`
  (66), and `vix_percentile` (50) at the top, but the signal was encoded
  smoothly into the posterior. These features would likely benefit more from
  an unsupervised regime model (HMM) or a hand-crafted threshold ensemble
  that mirrors v0's rule-based design than from a supervised classifier.

- **The lag-structural finding from V12 + E8 was tau-specific, not a
  fundamental detector defect.** Under Amendment 6 (Schmitt-tau-0.30), v0
  fires fast (2.0d median); the lag observed in V12 (-3.42 gap_days) and
  E8 (-8d exit-to-low) lives at V14's specific tau_in = 0.5556. Future
  detector evaluations should evaluate at the consumer's intended threshold,
  not at argmax-at-0.5.

- **The Bayesian-bucket screen in OMR insulates it from the failure modes
  RAMP exhibits.** E5 showed OMR's Sharpe-by-regime range/max = 16.6% across
  the three regimes the screen lets through. The detector's BEAR / UNPREDICTABLE
  decisions never reach OMR's trade log. This is a useful design pattern for
  any future strategy that wants to be robust to detector lag at the cost of
  regime-conditional alpha.

- **Variants with the same engine state often have action-convergent
  per-event P&L paths but divergent full-window Sharpes.** V14a/b/c had
  full-window Sharpes within 0.011 but per-event correlations of only 0.51
  (V14a-V14b) to 0.86 (V14b-V14c). The per-event noise averaged out over
  the 25-event window. Future spec design should test for both per-event
  divergence (does the action matter at all?) and full-window convergence
  (does it matter for the headline metric?).
