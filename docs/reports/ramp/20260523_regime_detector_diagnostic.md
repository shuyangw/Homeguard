# Regime Detector Diagnostic Report

**Status**: COMPLETE -- Phases 0-5 done

**Branch**: `regime-detector-diagnostic`

**Spec**: `docs/superpowers/specs/2026-05-23-regime-detector-diagnostic-design.md`

**Plan**: `docs/superpowers/plans/2026-05-23-regime-detector-diagnostic.md`

---

## Phase 0: Code archaeology

Source of truth: `src/strategies/advanced/market_regime_detector.py` (541 lines).
This Phase 0 writeup is built from direct code inspection, not from the spec or
the OMR/RAMP strategy docs.

### What the detector actually does (read from code, not from docs)

**Signature**

```
MarketRegimeDetector(lookback_window: int = 252)

classify_regime(
    spy_data: pd.DataFrame,        # OHLCV; 'close' column is the only one read
    vix_data: pd.DataFrame,        # 'close' column only
    timestamp: datetime,           # treated as label only -- detector reads
                                   # latest row via .iloc[-1], so caller must
                                   # truncate slices to <= timestamp before
                                   # calling
    *,
    min_coverage_pct: float = 0.95,  # soft-block threshold
    hard_block_pct: float = 0.80,    # hard-block threshold
) -> Tuple[str, float]              # (regime_name, confidence_score in [0,1])
```

**Algorithm (end-to-end, as implemented)**

1. **Data coverage gate** (lines 141-158). If `spy_data['close'].notna().mean()`
   < `hard_block_pct` -> `DataInsufficientError(hard_block=True)`. If
   < `min_coverage_pct` -> `DataInsufficientError(hard_block=False)`. Below
   200 SPY rows OR below `lookback_window` VIX rows -> log warning, return
   `('SIDEWAYS', 0.5)` (legacy length safeguard, lines 162-164).
2. **Indicator calculation** (`_calculate_indicators`, lines 190-243). All
   indicators are computed on the FULL passed-in slice and only the `.iloc[-1]`
   value is used. Indicators:
   - `sma_20`, `sma_50`, `sma_200`: trailing simple moving averages of
     `spy_data['close']` (no Kalman, no EWMA).
   - `above_20`, `above_50`, `above_200`: bools comparing `current_price`
     against each SMA.
   - `momentum_slope`: `(sma_20[-1] - sma_20[-20]) / sma_20[-20]` -- 20-day
     percentage change of the 20-day SMA itself, NOT raw price ROC.
   - `vix`: `vix_data['close'].iloc[-1]`.
   - `vix_percentile`: `(rolling_window < current_vix).sum() / window_len * 100`
     using the LAST `lookback_window` rows of VIX close. This is an
     "actual <= threshold" percentile, value range 0-100 (not 0-1).
   - `realized_vol`: 20-day rolling std of daily pct_change, annualized by
     `sqrt(252)`.
   - `volatility_spike`: `current_vix > 1.5 * vix_data['close'].rolling(20).mean().iloc[-1]`.
     Hard-coded 1.5x multiplier on a 20-day VIX SMA.
   - `sma_20_slope`, `sma_50_slope`: short-window slope proxies (5-bar and
     10-bar percent change of the respective SMAs). Computed but NOT used by
     the scoring rules in `REGIME_CRITERIA`.
3. **Score each regime** (`_score_regime`, lines 260-325). For each of the 5
   regimes, walk through the criteria dict and award +1 for each satisfied
   criterion, treating `above_mas` / `below_mas` as a fractional score
   (`matches_within_list / len(list)`). Total score is divided by the
   `criteria_count` (number of distinct criterion KEYS checked, not the count
   of MAs) to produce a normalized confidence in `[0, 1]`. **Each criterion is
   evaluated with an independent hard inequality** -- there is no continuous
   activation, no smoothing, and no cross-criterion weighting.
4. **Argmax over the 5 scores**. `best_regime = max(regime_scores, key=...)`.
   Returns `(best_regime, regime_scores[best_regime])`.

**REGIME_CRITERIA (verbatim from lines 57-89)**

| Regime | Criteria checked by `_score_regime` |
|---|---|
| STRONG_BULL | momentum_slope >= 0.02; vix_percentile <= 30; above 20/50/200 SMA; (volatility_regime label, ignored by scorer) |
| WEAK_BULL | 0.0 <= momentum_slope <= 0.02; vix_percentile <= 50; above 20/50 SMA; (volatility_regime label, ignored) |
| SIDEWAYS | -0.01 <= momentum_slope <= 0.01; 30 <= vix_percentile <= 60; (volatility_regime label, ignored) |
| UNPREDICTABLE | vix_percentile >= 60; volatility_spike == True; (volatility_regime label, ignored) |
| BEAR | momentum_slope <= -0.02; vix_percentile >= 70; below 20/50/200 SMA; (volatility_regime label, ignored) |

Notes on the criteria dict:
- The `'volatility_regime'` key (`'low' | 'moderate' | 'high'`) is present in
  every regime's criteria dict but is NOT inspected by `_score_regime` -- it
  is dead metadata.
- BEAR's `criteria_count` works out to 3 (momentum_slope_max, vix_percentile_min,
  below_mas). The MA portion contributes a fraction in `[0, 1]` based on how
  many of {20, 50, 200} the price is below; the other two are binary 0/1.
  Maximum BEAR score is 3 / 3 = 1.0; getting "2 out of 3 MAs below" while
  satisfying the other two criteria yields (1 + 1 + 2/3) / 3 = 0.889.
- STRONG_BULL has 4 distinct criterion keys (momentum_min, vix_max,
  above_mas, no momentum_max). WEAK_BULL has 5 (momentum_min + momentum_max +
  vix_max + above_mas). Different regimes have different `criteria_count`
  denominators, so their score scales are NOT comparable on the same axis --
  this is a structural quirk worth flagging for Phase 5.

### Why this is significant

The spec's H1 ("BEAR conjunction is structurally too restrictive") is partially
based on the wrong model. **The detector is already a score-based argmax
classifier, NOT a 5-AND hard conjunction.** A regime can fire even if not all
its criteria are satisfied -- it just needs to be the best fit relative to the
other four. BEAR may still rarely win because:

1. Its individual criteria ARE hard: VIX percentile >= 70, momentum <= -2%, AND
   below all three SMAs all need partial activation to push BEAR's normalized
   score above the competing regimes.
2. UNPREDICTABLE (`vix_pct >= 60` + `volatility_spike`) and SIDEWAYS
   (`-0.01 <= momentum <= 0.01` + `30 <= vix_pct <= 60`) often score higher
   even on bearish days because:
   - UNPREDICTABLE has only 2 keyed criteria, so each binary win contributes
     0.5 to its score -- a low bar.
   - SIDEWAYS satisfies its momentum band trivially when slope is near zero, and
     it has no MA criterion to fail on.
3. The normalization-by-criteria-count means BEAR with "2 of 3 criteria
   satisfied" scores 0.667 but UNPREDICTABLE with "1 of 2 criteria satisfied"
   scores 0.500 -- so BEAR can win, but the gap is fragile and any partial
   miss on the MA fraction pulls it below SIDEWAYS / UNPREDICTABLE.

**H1 must be reframed** as: "BEAR is not the argmax winner often enough" (a
relative score deficit between BEAR and {UNPREDICTABLE, SIDEWAYS} on stress
days) rather than "BEAR's AND-conjunction is too strict" (an absolute hard
cutoff).

Phase 5's option ranking must reflect that **Option E (score-based
reformulation) is already partially the current design**. The remaining design
space for Option E is replacing the per-criterion hard pass/fail with
continuous activation (e.g., `sigmoid((vix_pct - 70) / 10)` instead of
`vix_pct >= 70`) and normalizing the per-regime score scales so that
denominators (`criteria_count`) are comparable.

### Cached state (used by Phase 2 driver)

After each `classify_regime` call, the detector populates two instance
attributes (lines 102-109, written at lines 169 and 179):

- `self.last_indicators: Optional[Dict]` -- all values computed in
  `_calculate_indicators`: `current_price`, `sma_{20,50,200}`,
  `above_{20,50,200}`, `momentum_slope`, `vix`, `vix_percentile`,
  `realized_vol`, `volatility_spike`, `sma_20_slope`, `sma_50_slope`.
- `self.last_regime_scores: Optional[Dict[str, float]]` -- the 5-element
  score vector BEFORE the argmax collapse.

Phase 2's driver can read these directly to populate `labels.parquet`
columns without recomputing.

### Data path in production (verified via variants.py:36-90)

The Phase 4 research harness pattern in
`src/research/ramp_phase4/variants.py:35-80` -- the closest existing offline
usage to what Phase 2 will build -- is:

1. Load a wide panel keyed by symbol (`panel.SPY`, `panel.VIX` are Series).
2. Slice as-of-`t`: `spy_slice = spy.loc[:t]`, `vix_slice = vix.loc[:t]`.
3. Guard against short history: `if len(spy_slice) < 252 or len(vix_slice) < 252: return None`.
4. Build minimal per-symbol DataFrames on the fly:

```python
spy_df = pd.DataFrame({
    'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
    'volume': 1e6,
})
vix_df = pd.DataFrame({'close': vix_slice})
```

5. Call `_DETECTOR.classify_regime(spy_df, vix_df, t)`.
6. Read `_DETECTOR.last_regime_scores` for the full vector.

Phase 2's driver mirrors this pattern but reads from a flat Parquet (Phase 1
output) instead of the harness panel. Only the `close` column is functionally
required (detector reads no OHLV inputs); the others can be filled with
`close` as the variants harness does, or dropped.

Note: production live trading goes through
`RAMPSignals.detect_regime` (`src/strategies/advanced/ramp_strategy.py:155`),
which is a thin wrapper that builds the `spy_df` / `vix_df` and delegates to
`MarketRegimeDetector.classify_regime` (line 199). The `detect_regime(date,
data)` calls under `src/backtesting_v2/adapters/*.py` reference a DIFFERENT
signature that does NOT exist on `MarketRegimeDetector` -- those adapters are
not the production path for the live RAMP / OMR services and are out of scope
for this diagnostic.

### Open questions resolved

1. **Kalman vs raw SMAs**: **Raw SMAs.** `_calculate_indicators` uses
   `spy_data['close'].rolling(20|50|200).mean()` (lines 199-201). No Kalman
   filter, no EWMA. The memory note referencing Kalman was inaccurate or
   referred to a different file (likely `src/features/` or planned future
   work).
2. **VIX percentile window**: **Hard-coded default 252 trading days**, set in
   `__init__(lookback_window: int = 252)` (line 91). Parametrizable via the
   constructor but every production call site uses the default. Computed as
   `(window < current_vix).sum() / len(window) * 100` over the trailing
   `lookback_window` rows of `vix_data['close']` (lines 245-258). It is a
   trading-day lookback (positional indexing via `.iloc[-lookback_window:]`),
   NOT a calendar-time lookback.
3. **Missing data**: Raises `DataInsufficientError` when SPY close coverage
   is below either threshold. `< hard_block_pct (default 0.80)` -> hard block
   (no fallback). `< min_coverage_pct (default 0.95)` -> soft block (planner
   may use `safe_mode=True`). Coverage is measured as
   `spy_data['close'].notna().mean()` over the passed-in slice. A separate
   legacy length check on lines 162-164 returns `('SIDEWAYS', 0.5)` if length
   is below 200 SPY rows or below `lookback_window` VIX rows -- this fires
   only when callers pass `hard_block_pct=0.0` to bypass the coverage check.
4. **Volatility spike formula**: `current_vix > 1.5 * vix_20_avg` where
   `vix_20_avg = vix_data['close'].rolling(20).mean().iloc[-1]` (lines 220-222).
   The 1.5x multiplier is hard-coded; the 20-day window is hard-coded. This is
   the only criterion in `REGIME_CRITERIA` that is fundamentally non-monotonic
   in VIX (because the denominator moves with the VIX), so it fires on
   *sudden* spikes rather than sustained high-VIX regimes. NOT documented
   anywhere in the OMR or RAMP strategy markdown docs.
5. **Test coverage**: **Only one direct test** of the production detector
   exists: `tests/strategies/test_market_regime_detector.py` (25 lines, 1
   test). It verifies that `last_regime_scores` is populated after
   `classify_regime` and contains exactly the 5 expected regime keys. No
   tests cover:
   - The actual classification logic (does BEAR actually fire on bear days?).
   - The `DataInsufficientError` paths.
   - The legacy length safeguard.
   - `_calculate_indicators` outputs (e.g., VIX percentile correctness).
   - `_score_regime` per-criterion behavior.
   - `analyze_regime_history` end-to-end.

   Other regime-related test files are unrelated to `MarketRegimeDetector`:
   - `tests/backtesting/regimes/test_detector.py` (321 lines) tests
     `TrendDetector` / `VolatilityDetector` / `DrawdownDetector` in
     `backtesting/regimes/detector.py` -- a separate, unrelated module.
   - `tests/test_regime_analysis_toggle.py` (108 lines) tests a feature
     toggle, not the detector.
   - `tests/ops/test_backfill_regime_state.py` (279 lines) tests an ops
     script.

   **This thin test coverage is itself a finding**: the Phase 5 synthesis
   should consider whether characterization tests around
   `_calculate_indicators` and `_score_regime` (covering the corner cases
   surfaced by Phases 2-4) are a deliverable.

### Deviations from the plan template / surprises found

- **Surprise 1 (low-impact)**: `volatility_regime` keys in `REGIME_CRITERIA`
  are dead metadata -- `_score_regime` never reads them. Documented above so
  Phase 5 doesn't propose tuning a parameter that has no effect.
- **Surprise 2 (low-impact)**: `sma_20_slope` and `sma_50_slope` are computed
  in `_calculate_indicators` but never used by the scoring code. They are
  available on `last_indicators` for downstream consumers.
- **Surprise 3 (medium-impact)**: BEAR and STRONG_BULL have different
  `criteria_count` denominators (3 vs 4), so their normalized scores are NOT
  directly comparable. This is a structural asymmetry distinct from the
  conjunction-vs-score reframing. Worth flagging in the Phase 5 options.
- **Surprise 4 (informational)**: `backtesting_v2` adapters call a
  `detect_regime(date, data)` method on what they think is a regime detector,
  but `MarketRegimeDetector` does not implement that signature. Those code
  paths would AttributeError at runtime; they are NOT the production path
  (production goes through `ExecutionEngine` -> `RAMPSignals.detect_regime`
  -> `MarketRegimeDetector.classify_regime`). Out of scope for this
  diagnostic but should be filed as a separate cleanup.

No spec-breaking contradictions surfaced. Detector returns 5 regimes (matches
spec). Uses raw SMAs (no Kalman). Lookback is trading-day positional (not
calendar-time). `volatility_spike` formula is `vix > 1.5 * vix_20_sma` (matches
the "vol spike" concept; just was not previously documented). The H1 reframing
is the substantive correction; everything else aligns with the spec.

---

## Phase 5: Synthesis

All numeric evidence below is drawn from `diagnostics/regime/v0/phase4_summary.txt`
and the Phase 4 figures (`diagnostics/regime/v0/figures/analysis_{A,B,C,E,F}.png`),
covering 2360 replay days (2017-01-03 through 2026-05-22) on the v0 detector.

The standard Sharpe-SE caveat (Sharpe SE ~= 0.17 over the 5y RAMP window;
differences below 0.30 in Sharpe are within one SE) applies to any downstream
PnL comparison built on this diagnostic; the diagnostic itself reports
classification statistics, not Sharpe.

### Hypothesis verdicts

#### H1: BEAR conjunction structurally too restrictive

**Verdict**: REFUTED (in its literal form); SUPPORTED (in its reframed form
"BEAR is not the argmax winner often enough on stress days")

**Evidence**:
- Phase 4 Analysis A: BEAR fires on 16.19% of all 2360 days. The spec's
  literal H1 prediction "BEAR < 5% of any year" holds in only 4 of 10 years
  (2017: 2.8%, 2021: 1.6%, 2023: 4.4%, 2024: 6.7%). In stress years BEAR
  dominates: 2018 31.5%, 2022 54.2%, 2025 18.0%, 2026 20.4%. The absolute
  firing rate is therefore NOT structurally restrictive in aggregate.
- Analysis E: on the 200 G1_BEAR ground-truth days that the detector did NOT
  label BEAR (53.9% miss rate against G1_BEAR), momentum_fail (77.5%) and
  vix_pct_fail (~73%) dominate; below-50 fails on 65%, below-20 on 59%,
  below-200 on only 27%. So when the detector misses a G1_BEAR day, it is
  almost always because the momentum_slope band (<= -2%) and/or 252d VIX
  percentile (>= 70) have not yet activated -- not because the SMA criteria
  are simultaneously failing.

**Reframing note**: per Phase 0, the detector is a score-based argmax over
five regimes (each criterion normalized by its `criteria_count`), not a 5-AND
conjunction. The corrected H1 ("BEAR is not the argmax winner often enough
on stress days") is supported by the 53.9% G1_BEAR miss rate and by the
consistent observation that the momentum (>= -2% slope) and VIX (252d pct
>= 70) gates are the binding constraints. The literal H1 ("conjunction too
restrictive") is refuted by the 16.19% overall firing rate and the
60+% BEAR fractions seen in 2018 and 2022.

#### H2: UNPREDICTABLE dead zones in uptrends

**Verdict**: REFUTED

**Evidence**:
- Phase 4 Analysis A (regime distribution by year): UNPREDICTABLE is a near-
  empty regime over the full 2017-2026 window -- it is visibly absent in
  2017, 2019, 2021, 2023, 2025, 2026 and reaches a maximum of ~7% only in
  2020 (the COVID period). It cannot be acting as a "dead zone" absorbing
  uptrend stress days because the detector almost never assigns days to it
  in the first place.
- Analysis B (run lengths): UNPREDICTABLE has the smallest run-length
  histogram of all five regimes (peak bar height ~7 vs WEAK_BULL ~90,
  SIDEWAYS ~61, BEAR ~33). Median UNPREDICTABLE run length is 1.5 days --
  the lowest of any regime, indicating that even when UNPREDICTABLE fires it
  is a single-day flash, not a sustained label.
- Analysis C (transition matrix): when UNPREDICTABLE does fire, it transitions
  to BEAR 22.0% of the time and stays in UNPREDICTABLE 65.9% of the time --
  consistent with a brief vol-spike artifact during stress, not a "dead zone
  in uptrends." There is no row in the transition matrix where UNPREDICTABLE
  follows a bull regime with non-trivial probability.

The H2 cross-tab (days with VIX > 25 AND SPY > 50-SMA, by labeled regime)
could not be computed in this environment due to a pyarrow 19 vs writer-22
parquet incompatibility on `diagnostics/regime/v0/labels.parquet` (writer
emits Parquet 2.6 + repetition histograms that pyarrow 19 cannot parse). The
distributional evidence above (UNPREDICTABLE near-zero overall) is
sufficient to refute the literal H2; the missing cross-tab would only refine
WHERE the few UNPREDICTABLE days fall (in stress periods, not in uptrends),
not change the verdict.

#### H3: 252d VIX percentile compresses adaptively

**Verdict**: REFUTED

**Evidence**:
- Phase 4 Analysis F: the firing rate of "VIX 252d-percentile >= 70" is
  28.7%. Across alternative lookbacks the range is narrow: 63d 30.7%, 126d
  27.5%, 252d 28.7%, 504d 32.2%. The total spread is 4.7 percentage points
  (or about 16% relative to the 252d baseline) across a 8x change in window
  length.
- If adaptive compression were the dominant pathology, a much longer window
  (504d, closer to a two-year baseline) would fire materially more often
  than the 252d (it does, but only by 3.5 pp absolute). A shorter window
  (63d) would compress harder during prolonged stress; it does, but again
  by only 2.0 pp. The VIX percentile is therefore not the primary loose
  constraint -- the binding constraint in Analysis E's ablation is the
  momentum_slope, with vix_pct_fail a close second (73% of misses) but not
  the dominant driver.

#### H4: No hysteresis -> label flicker

**Verdict**: SUPPORTED

**Evidence**:
- Phase 4 Analysis B (median run lengths): STRONG_BULL 4.0, WEAK_BULL 2.0,
  SIDEWAYS 2.0, UNPREDICTABLE 1.5, BEAR 2.0 days. Four of the five regimes
  have median runs of 2 days or fewer. The run-length histograms (figure B)
  show a sharp peak at run length = 1-2 days for WEAK_BULL, SIDEWAYS,
  UNPREDICTABLE, and BEAR.
- Analysis C (transition matrix): mean diagonal mass is 0.761. Per-regime
  self-transition probabilities are BEAR 0.835, STRONG_BULL 0.839, WEAK_BULL
  0.810, SIDEWAYS 0.664, UNPREDICTABLE 0.659. SIDEWAYS and UNPREDICTABLE in
  particular have one-day self-persistence below 0.70 -- those days flip out
  within one or two bars.
- The SIDEWAYS <-> WEAK_BULL band is especially leaky: SIDEWAYS->WEAK_BULL
  21.0%, WEAK_BULL->SIDEWAYS 8.9%. This is consistent with a momentum_slope
  variable that hovers within the +/-1% SIDEWAYS band and just barely
  crosses the 0% WEAK_BULL boundary on adjacent days. A hysteresis band of
  e.g. require 3-day persistence before label switch would absorb these.

The BEAR median run of 2.0 days plus the BEAR->BEAR self-transition of 0.835
indicates that BEAR fires AND persists during true stress (e.g. the 2022 run
visible up to ~50 days in the histogram), but also flickers as 1-2 day
flashes outside of those sustained episodes. Hysteresis would not suppress
the genuine BEAR runs (they are sticky already) but would suppress the
isolated 1-day BEAR flashes that contribute noise to RAMP's regime overlay.

#### H5: SMA-based inputs lag regime onset

**Verdict**: SUPPORTED (qualitatively, with a small-sample caveat)

**Evidence**:
- Phase 4 Analysis D: across 5 ground-truth drawdown events (G4 labeler),
  BEAR fired in 5 of 5 cases (100% capture). The median onset lag from
  event start to first BEAR label was 14.0 days. With only 5 events the
  precision on the median is low (the inter-quartile range was not
  reported in the summary), but a two-week lag is consistent with the
  expected behavior of a 20-day momentum slope (`(sma_20[-1] - sma_20[-20])
  / sma_20[-20]`) which needs roughly 20 trading days of price weakness
  before crossing the -2% threshold.
- Analysis E reinforces this: momentum_fail is the #1 reason BEAR is missed
  on G1_BEAR days (77.5%). Momentum_slope is built on a trailing 20-day SMA
  with a 20-day lookback -- it is structurally a lagging indicator and is
  the binding constraint when BEAR fails to fire.

The verdict is SUPPORTED but the n=5 event sample limits the strength. A
leading-indicator candidate (Kalman-filtered slope, intraday range
expansion, breadth) would need to be tested directly against the same
event set; that is the Option D work.

### Remediation option ranking

Rationale: H4 (hysteresis / flicker) is strongly supported by every
diagonal-mass and run-length statistic in Phase 4; addressing it is local
(state-machine wrapper, no model retraining) and improves all five regimes
simultaneously. H1's reframed version and H5 both point at the same
underlying input (the 20-day SMA-on-SMA momentum_slope), so Option D
(leading inputs) addresses both with a single change. Option E (continuous
score / criteria normalization) addresses the structural argmax asymmetry
flagged in Phase 0 (BEAR has 3 keyed criteria vs STRONG_BULL's 4, so their
scores are not directly comparable) and is a one-time refactor with broad
downstream impact. Options A (threshold recalibration) and C (VIX lookback
adjustment) are the lowest-leverage changes: Phase 4 Analysis F shows the
VIX lookback choice moves the firing rate by only ~4.7 pp across a 8x
window change, and the momentum_slope threshold is already binding -- moving
it by 50 bps without changing the input would just push noise across the
boundary.

| Rank | Option | Hypotheses addressed | Evidence-based rationale |
|------|--------|----------------------|--------------------------|
| 1 | B: Hysteresis layer | H4 | Strongest evidence base. Median run length is 2 days or less for 4 of 5 regimes; SIDEWAYS->WEAK_BULL transitions 21.0%. A minimal state-machine wrapper (e.g. require 3 consecutive same-label classifications before switching) leaves the detector code untouched and directly attacks the most-supported hypothesis. Cheapest to implement, simplest to unit-test, isolates the change from the score model. |
| 2 | E: Score-based reformulation (continuous + normalized criteria_count) | H1 (reframed), H2, H4 | Phase 0 already established that the detector IS a score model, so this is partly a refactor. The remaining design space (continuous criterion activation + a common denominator across regimes) addresses the BEAR-vs-UNPREDICTABLE / SIDEWAYS argmax asymmetry that drives the 53.9% G1_BEAR miss rate. Higher implementation cost than B but covers three hypotheses. |
| 3 | D: Leading indicators (replace SMA-on-SMA momentum with Kalman / shorter-horizon proxy) | H5, H1 (reframed) | Median onset lag 14.0 days + momentum_fail dominant (77.5%) in G1_BEAR misses indicate that the 20-day SMA-on-SMA momentum is the binding lagging input. A leading replacement (Kalman slope, 5d ROC, breadth) would shorten the lag, but introducing a new feature carries overfitting and OOS-degradation risk and requires its own walk-forward validation. Higher cost than E, narrower coverage. |
| 4 | A: Threshold recalibration (e.g. lower momentum from -2% to -1.5%, lower VIX pct from 70 to 60) | H1, H3 | The thresholds are binding but the issue is structural (lagging inputs + score asymmetry), not the threshold values. Tuning thresholds without addressing the underlying inputs invites overfitting to the in-sample event set. Useful as a sanity check (sensitivity sweep) but not a standalone fix. |
| 5 | C: VIX lookback adjustment (e.g. 252d -> 126d) | H3 | Lowest leverage. Phase 4 Analysis F shows the firing rate range is 27.5%-32.2% across 63d/126d/252d/504d -- 4.7 pp spread for an 8x change in window length. The VIX percentile is not the primary loose constraint. Avoid changing this in isolation. |

### Next-step recommendation

**(c) Both in parallel**, with strong asymmetry between the two tracks:

1. **Track 1 (highest priority): RAMP BEAR-day cash logic.** This is the May
   2026 root-cause's higher-leverage path. Even if the detector is improved,
   RAMP only beats V1 (vanilla momentum, no regime overlay) at 1.5x costs
   IF the overlay actually adds value on BEAR days. Fix the cash logic
   first; that work is independent of the detector and unblocks the
   critical question of whether the regime overlay is the bottleneck at
   all.
2. **Track 2 (parallel, lower priority): Regime detector v1 design with
   Option B (hysteresis layer) as the top-1 remediation.** Hysteresis is
   the cheapest change with the strongest evidence base. Defer Options D
   and E until v1+hysteresis has been EXT-OOS evaluated; if hysteresis
   alone closes the gap to V1, we save the implementation cost of the
   bigger refactors.

Tracks 1 and 2 share no code, so they can be developed and validated
independently. The convergence point is the EXT-OOS comparison (see below).

### The critical reminder

Any regime-detector improvement (v1, v2, or beyond) must beat V1 (vanilla
momentum, no regime overlay) on EXT-OOS at 1.5x costs. If improved-detector
+ RAMP does not beat V1, the detector was not the bottleneck -- the regime
overlay itself (the act of conditioning position sizing on detector output)
is the problem, and the next investigation should be into the overlay's
expected value at the position-sizing layer (Track 1), not into more
detector refinements. This same reminder applies to the BEAR-day cash
logic: if Track 1 alone closes the gap to V1, Track 2 has no remaining
business case beyond paying down technical debt.
