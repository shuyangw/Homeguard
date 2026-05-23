# Regime Detector Diagnostic Report

**Status**: WIP -- Phase 0 complete

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
