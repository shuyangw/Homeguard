# Regime Detector Diagnostic Backtest - Design

**Date**: 2026-05-23
**Status**: Approved (brainstorming -> spec)
**Owner**: Shuyang
**Type**: Diagnostic / research (no production code changes)
**Related**:
- `docs/reports/ramp/20260505_root_cause_investigation.md`
- `docs/reports/ramp/20260505_phase3b_bear_optimizer.md`
- `docs/reports/ramp/20260505_variant_exploration_phase3a.md`
- `src/strategies/advanced/market_regime_detector.py` (subject under test)

---

## Problem statement

A 30-day Grafana panel (Apr 23 - May 21, 2026) shows the production `MarketRegimeDetector` confined to `SIDEWAYS`, `WEAK_BULL`, and `STRONG_BULL`, with rapid same-week flickering between the three. `BEAR` and `UNPREDICTABLE` are absent.

For this specific window -- SPY in a quiet uptrend -- the absence of `BEAR` is structurally correct: the detector's `BEAR` criterion requires SPY below all three SMAs simultaneously. But the chart also exposes two robustness issues that are *not* window-specific:

1. **Visual flicker** between adjacent regimes within single weeks, which indicates the detector has no persistence or hysteresis layer.
2. **Suspected dead zones** in `UNPREDICTABLE` (no path to that regime when SPY is above all three SMAs, regardless of VIX behavior).

These are independent of, but compound with, the root cause already identified in `20260505_root_cause_investigation.md`: BEAR-regime stock selection in RAMP loses money on the days the detector *does* fire BEAR (Sharpe -2.17 across 64 days in 2025-2026), and removing the regime overlay entirely (V1) outperforms the regime-aware variant (V0) by ~0.24 Sharpe in EXT-OOS.

This plan does **not** propose detector changes. It proposes a diagnostic backtest that quantifies how the current detector behaves over the full 2017-2026 sample, so that any future detector revision can be evaluated against a concrete baseline.

## Hypotheses under test

| ID | Hypothesis | Mechanism | Falsifiable prediction |
|----|------------|-----------|------------------------|
| H1 | `BEAR` conjunction is structurally too restrictive | Requires 5-AND alignment (below 20/50/200, slope < -2%, VIX pct > 70). Asymmetric with `STRONG_BULL` (also 5-AND but conditions much easier to satisfy in real markets). | `BEAR` fires on < 5% of total trading days in 2017-2026; median onset lag vs SPY peak > 20 trading days. |
| H2 | `UNPREDICTABLE` has dead zones in uptrends | The logic only reaches `UNPREDICTABLE` when SPY is below all SMAs OR mixed-vs-SMAs with VIX > 60th pct. There is no path to `UNPREDICTABLE` when SPY is above all SMAs, even during VIX spikes. | Days with VIX > 25 (absolute, not percentile) while SPY > 50-SMA are labeled `STRONG_BULL` or `WEAK_BULL`, not `UNPREDICTABLE`. |
| H3 | 252-day VIX percentile compresses adaptively to bull-era vol | After sustained calm, the 70th percentile of trailing 252-day VIX falls to ~17-18, making the BEAR vol condition trivially satisfied during any vol uptick. The reverse holds after 2020: percentile sits at 25+ for 18+ months, making BEAR's vol condition unreachable during minor drawdowns. | Re-running the detector with VIX percentile lookback in {63, 126, 252, 504} produces materially different `BEAR` durations and `UNPREDICTABLE` frequencies. |
| H4 | No hysteresis -> label flicker | Each day's classification is independent; no minimum dwell time. Inputs near thresholds produce day-to-day oscillation. | Median run-length per regime < 3 trading days for at least 2 of 5 regimes. |
| H5 | SMA-based inputs lag regime onset | 200-SMA takes weeks to break in a real drawdown. By the time `BEAR` triggers, the drawdown is well underway. | For each independently-labeled drawdown event > 10%, median lag from SPY local peak to first `BEAR` label > 15 trading days. |

These hypotheses are not mutually exclusive. The diagnostic is designed to apportion blame across them, not to pick a winner.

## Remediation options (Phase 5 ranking inputs)

These five candidate detector revisions are the option set Phase 5's synthesis ranks against the evidence. Each option may address one or more hypotheses; mapping below is the primary linkage, not exclusive coverage.

| Option | Description | Hypothesis primarily addressed |
|----|-------------|-------------|
| **A - Threshold recalibration** | Keep the existing 5-regime structure and 5-AND conjunction logic; tune specific thresholds (e.g., BEAR vol_percentile from 70 to 60, momentum slope from -2% to -1.5%, drawdown trigger from 5% to 3%). Minimal architectural change. | H1, H3 |
| **B - Hysteresis layer** | Add a minimum dwell-time gate (e.g., a regime must persist >= 3 trading days before flipping) or a threshold band around regime boundaries that resists oscillation. Suppresses single-day flickers without changing the underlying classification logic. | H4 |
| **C - VIX lookback adjustment** | Replace the hard-coded 252-day VIX percentile with a shorter (63d, 126d) or longer (504d) window, or switch the BEAR vol condition from a percentile to an absolute-vol criterion (e.g., VIX > 25). | H3 |
| **D - Leading indicators** | Augment the detector input set with non-lagging signals: VIX term structure (VIX / VIX3M ratio as a stand-in), credit spreads (HYG-LQD), market breadth (NYSE advance-decline). The SMA/momentum inputs lag; these complement. | H5 |
| **E - Score-based reformulation** | Replace the 5-AND conjunction logic with a continuous score per regime and classify by argmax. Each input contributes a soft vote rather than a hard pass/fail; transitions become gradual instead of binary. The existing `_score_regime()` helper hints at this direction. | H1, H2, H4 |

Phase 5's synthesis evaluates each option's expected impact against the evidence weight from Phases 4A-F, then ranks them. The ranking is not a recommendation by itself -- per Appendix B, even the highest-ranked detector revision is conditional on the broader finding that BEAR-day stock selection (or BEAR-to-cash) is the higher-leverage RAMP intervention.

## Goals

The diagnostic produces three artifacts:

1. **A per-day record of the detector's outputs and intermediate values** over 2017-2026, stored as Hive-partitioned Parquet, suitable for ad-hoc analysis and re-use.
2. **A set of independent ground-truth regime labels** (drawdown-based, vol-spike-based, hand-curated NBER-style), against which the detector can be compared.
3. **A synthesis report** that tests H1-H5 quantitatively and recommends which of the five remediation options (A-E above) is best supported by the evidence.

The infrastructure is built to be reusable: a future revised detector (`v1`) can be evaluated by re-running the same driver with a different class injected, producing directly comparable artifacts.

## Non-goals

Out of scope for this plan: building alternative detectors (HMM, score-based, leading-indicator-augmented), re-optimizing RAMP parameters, deploying any changes to live trading, integration with the monitoring stack (`docs/planning/20260418_MONITORING_SYSTEM_PLAN.md`), and BEAR-regime stock selection logic (separate work stream).

## Phased plan

The diagnostic is structured to surface failures early. Each phase has explicit success criteria; failure of an earlier phase must be resolved before later phases proceed.

### Phase 0 - Code archaeology (~30 minutes)

Before designing the driver, verify the *actual* current code in `src/strategies/advanced/market_regime_detector.py` matches the model we have been reasoning from. The OMR strategy doc and the architecture doc both describe a five-AND conjunction logic, but the cached memory note also references three parallel Kalman filters preserving the `above_20`/`above_50`/`above_200` structure used in `_score_regime()`. These two descriptions are not consistent. The diagnostic needs to test the code that actually runs in production, not the code documented two years ago.

**Commands:**

```bash
# Read the current detector implementation in full
cat src/strategies/advanced/market_regime_detector.py

# Identify every call site
grep -rn "MarketRegimeDetector\|classify_regime\|detect_regime" src/ --include="*.py" | grep -v test_

# Identify the exact data sources used in production
grep -rn "spy_data\|vix_data" src/strategies/advanced/ src/trading/adapters/ --include="*.py"
```

**Questions to answer in writing before proceeding:**

- What is the exact signature of the public entry point (`classify_regime` vs `detect_regime`)?
- What is the actual lookback window for the VIX percentile? Is it parametrizable?
- Does the code use Kalman-smoothed trend estimates or raw SMAs? If Kalman, what are the Q/R parameters?
- What does the detector return -- a string label, an enum, a (label, confidence) tuple, or something richer?
- How is point-in-time data delivered to it in production? Does the caller pass a sliced dataframe, or a full series with a `timestamp` argument that the detector slices internally?

Output of this phase: a one-page summary at the top of the eventual diagnostic report describing what the detector actually does, written from the code (not from docs).

**Exit criterion:** A correct call signature and input contract is documented. Without this, the driver in Phase 2 will silently produce wrong results.

### Phase 1 - Data pipeline (~1-2 hours)

The diagnostic needs SPY and VIX daily OHLCV from 2017-01-01 through the most recent trading day, with enough prior history to satisfy the longest indicator lookback (252-day VIX percentile + 200-day SMA = ~300 days of pre-roll).

**Data sources in priority order:**

1. **Primary: same source production uses.** Phase 0 will identify this. Most likely Alpaca for SPY, yfinance for VIX (Alpaca free tier does not carry VIX as a direct symbol on all data plans). If production uses a mix, mirror the mix.
2. **Sanity-check overlay:** pull the same series from a second source (yfinance if production uses Alpaca, or vice versa) and verify daily-bar agreement at >= 99.5% on close prices. Any mismatch on critical event days (e.g., 2020-03-12, 2022-06-13) is a stop condition.

**Storage:** save as `diagnostics/data/spy_vix_2016_2026.parquet` with columns `[date, spy_open, spy_high, spy_low, spy_close, spy_volume, vix_open, vix_high, vix_low, vix_close]`. The 2016 prefix is the pre-roll for indicator warm-up.

**Exit criterion:** Two independent sources agree on closes within 0.1% on every day; no missing trading days vs NYSE calendar.

### Phase 2 - Diagnostic driver (~2-3 hours)

A script that replays the detector day-by-day across the full sample with strict point-in-time discipline.

**Path:** `scripts/diagnostics/regime_detector_replay.py`

**Behavior:** For each trading day `t` from 2017-01-01 onward:

1. Slice the SPY and VIX dataframes to `[t - 400d, t]` (inclusive of `t`'s close -- production calls run after the close).
2. Invoke `MarketRegimeDetector().classify_regime(spy_slice, vix_slice, t)`.
3. Independently compute and log all underlying values used by the detector: `above_20`, `above_50`, `above_200`, `momentum_slope`, `vix_percentile_252d`, plus the 5 conditions that would trigger each regime branch.
4. Additionally compute parametrized alternatives that do not affect the output but inform later analysis: `vix_percentile_63d`, `vix_percentile_126d`, `vix_percentile_504d`, `realized_vol_20d`, `realized_vol_60d`, `vix_term_proxy` (VIX / VIX 5-day MA as a poor-man's term structure stand-in until VIX3M is added).

**Output schema** -- `diagnostics/regime/v0/labels.parquet`, Hive-partitioned by year:

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading day (close) |
| `regime` | str | One of `{STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR}` |
| `confidence` | float | Detector's confidence score |
| `above_20` | bool | SPY close > 20-day SMA |
| `above_50` | bool | SPY close > 50-day SMA |
| `above_200` | bool | SPY close > 200-day SMA |
| `momentum_slope` | float | (SMA20[t] - SMA20[t-20]) / SMA20[t-20] |
| `vix_close` | float | Raw VIX close |
| `vix_percentile_252d` | float | Production VIX percentile |
| `vix_percentile_63d` | float | Short-lookback alternative |
| `vix_percentile_126d` | float | Medium-lookback alternative |
| `vix_percentile_504d` | float | Long-lookback alternative |
| `realized_vol_20d` | float | Annualized 20-day realized vol of SPY |
| `realized_vol_60d` | float | Annualized 60-day realized vol of SPY |
| `vix_5d_ma_ratio` | float | VIX / VIX 5-day MA |
| `branch_taken` | str | Which `if`/`elif` branch produced the label, for input ablation |
| `spy_close` | float | SPY close |
| `spy_drawdown_from_252d_high` | float | Current drawdown from trailing 252-day high |

**Critical correctness checks:**

- No look-ahead: assert that for every day `t`, no value in the output row depends on data with date > `t`.
- Idempotency: re-running the driver on the same data must produce byte-identical output.
- Production parity: pick 10 random recent dates, re-run production's detector against the same inputs, confirm `regime` and `confidence` match exactly.

**Exit criterion:** All three correctness checks pass; output parquet is generated for the full 2017-2026 sample.

### Phase 3 - Independent ground-truth labelers (~1-2 hours)

The detector's labels need something to be measured against. There is no single "true" regime label, so we build several independent labelers that triangulate. None are forward-looking unless explicitly noted.

**G1 - Drawdown-based BEAR (concurrent, observable).** A day is `G1_BEAR` if SPY's drawdown from the trailing 252-day high exceeds 10%. Threshold and lookback are configurable. This is a *strict* definition -- it catches sustained corrections but not flash crashes.

**G2 - Forward-window BEAR (forward-looking, in-sample only).** A day is `G2_BEAR` if the forward 30-day SPY return is < -5% AND forward 30-day realized vol > 25% annualized. This is *not* a basis for any live decision -- its only purpose is to evaluate whether the detector identified BEAR days early enough to be actionable. Lag analysis = days from G2_BEAR onset to first `BEAR` label.

**G3 - Vol-spike UNPREDICTABLE.** A day is `G3_VOL_SPIKE` if VIX > 30 (absolute level) OR VIX increased > 50% over a trailing 5-day window. This is the test for H2 (dead zone in uptrends): days that are clearly volatility events but where SPY is still above SMAs.

**G4 - Hand-curated event labels.** A small CSV (`config/diagnostics/regime_events_2017_2026.csv`) identifying obvious regime periods by hand: Q4 2018 selloff, Feb-Mar 2020 COVID, 2022 full-year drawdown, 2025-2026 drawdown periods, plus any major single-day vol spikes (Aug 2024, Apr 2025 if applicable). Each row has `[event_name, start_date, end_date, event_type]` where `event_type` is one of `{drawdown, vol_spike, regime_change}`. This is the most conservative ground truth.

**Reuse opportunity:** `src/backtesting/regimes/detector.py` already has `TrendDetector`, `VolatilityDetector`, `DrawdownDetector`. Use them where possible rather than re-implementing. The hand-curated labels are unique to this diagnostic.

**Exit criterion:** All four labelings are computed and written to `diagnostics/regime/ground_truth.parquet`, with one row per trading day and one column per labeler.

### Phase 4 - Analysis (~3-4 hours)

A Jupyter notebook (`notebooks/diagnostics/regime_detector_v0_analysis.ipynb`) that produces the six diagnostic plots and statistical tests listed below. Each maps to one or more hypotheses.

**A. Regime distribution** (tests H1). Bar chart of % time in each regime, broken out by year. Expected pattern if H1 holds: `BEAR` < 5% in any year; very low in calm years (2017, 2019, 2021, 2023-2024).

**B. Run-length distribution** (tests H4). Histogram of consecutive days per regime, separately for each of the 5 labels. Median, P25, P75. Expected pattern if H4 holds: median run-length < 3 days for at least two regimes; long tail dominated by `WEAK_BULL` / `STRONG_BULL`.

**C. Empirical transition matrix** (tests H4 and connects to H1). 5x5 matrix of P(r_{t+1} | r_t) estimated from data. Diagonal entries near 1.0 indicate persistent regimes; off-diagonal mass indicates flickering. Sparse off-diagonals between non-adjacent regimes (e.g., `STRONG_BULL` -> `BEAR` directly) indicate the detector cannot reach extreme regimes without passing through intermediate states.

**D. Lag-to-event** (tests H5). For each G4 event of type `drawdown`, measure the lag from event start date to the detector's first `BEAR` label within the event window. Report median, P25, P75, and the fraction of events where `BEAR` was never labeled before the event ended.

**E. Input ablation** (tests H1 and H2; this is the most actionable analysis). Restrict to days where `G1_BEAR` is true but the detector label is not `BEAR`. For each such day, decompose: of the 5 conjunction conditions, which one(s) failed? Histogram. If 80% of failures are due to "VIX percentile < 70" while the other 4 conditions are satisfied, that is a smoking gun for H3. If failures are spread across multiple conditions, the 5-AND structure itself is the problem (supporting H1).

**F. Lookback-window sensitivity** (tests H3). Re-classify the entire sample five times, varying only the VIX percentile lookback in {63, 126, 252, 504} days. Compare regime distributions and `BEAR`-day count across the four versions. If the shorter lookbacks fire `BEAR` substantially more often during 2018 and 2025-2026 drawdowns, H3 is supported.

**Statistical caveat to be carried through the entire analysis:** Sharpe and Sharpe-like ratios over EXT-OOS windows have standard errors ~0.17 (per the May 2026 root-cause investigation). Where this diagnostic compares per-regime performance metrics, differences smaller than 0.2 in Sharpe are not statistically meaningful and should be flagged as such.

**Exit criterion:** Notebook executes end-to-end without errors; all six analyses produce output; preliminary findings can be summarized in 3-5 sentences per hypothesis.

### Phase 5 - Synthesis report (~1-2 hours)

A short report at `docs/reports/ramp/20260523_regime_detector_diagnostic.md` synthesizing Phase 4's findings into:

1. A verdict on each hypothesis H1-H5 (supported / refuted / inconclusive, with the specific quantitative evidence).
2. A ranking of the five remediation options (A-E enumerated above) by the diagnostic's evidence weight. For example, if H4 (no hysteresis) is strongly supported but H1 (BEAR criterion too restrictive) is refuted, then option B (hysteresis layer) is the higher-leverage fix and option A (threshold recalibration) is deprioritized.
3. An explicit recommendation for the next planning doc to write (regime detector v1 design, RAMP BEAR-day cash logic, or both in parallel).
4. The critical reminder, repeated explicitly: any regime detector improvement must be validated against the V1 baseline (vanilla momentum, no regime overlay) on EXT-OOS at 1.5x costs. If improved-detector + RAMP does not beat V1, the detector was not the bottleneck.

## Risk table

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 0 reveals current code differs materially from the model we've been reasoning from (e.g., uses Kalman not raw SMAs) | Medium | High | Phase 0 is explicitly designed to surface this. Rewrite Phase 2's intermediate-value logging to match the actual code before proceeding. |
| VIX data source mismatch between production and diagnostic produces non-representative results | Medium | Medium | Phase 1's dual-source sanity check. Run a small live-vs-diagnostic comparison on 20 recent days before declaring Phase 2 done. |
| Point-in-time bugs in the replay driver introduce look-ahead | Low | High | Explicit assertion in Phase 2; production parity check on 10 random recent dates. |
| Hand-curated G4 labels reflect the analyst's bias rather than independent truth | Medium | Low | Treat G4 as the *most conservative* ground truth, not the only one. Conclusions must replicate across G1, G2, G3, G4. |
| Diagnostic produces findings that contradict the May 2026 root cause investigation | Low | High | Re-read both reports before publishing the synthesis. Disagreement is interesting, not disqualifying -- investigate which is correct rather than papering over the conflict. |
| Findings recommend a detector overhaul, but BEAR-regime stock selection is still the dominant fix per prior work | High | Low | Synthesis report must explicitly carry the "detector != bottleneck" caveat. The diagnostic informs *which* remediation is best supported but does not change the priority ordering -- fixing BEAR-day stock selection (or going BEAR-to-cash) remains the higher-leverage RAMP intervention regardless. |

## Success criteria

The diagnostic succeeds if, at the end of Phase 5, three things are true:

1. Each hypothesis H1-H5 has a verdict backed by specific quantitative evidence from the 2017-2026 sample.
2. The next-step recommendation between options A-E is supported by that evidence and would survive review by another quant.
3. The diagnostic infrastructure (driver + labelers + notebook) can be re-run against a hypothetical revised detector (`v1`) with no more than 30 minutes of incremental work, producing directly comparable artifacts.

The diagnostic *fails* if Phase 4 produces findings that fit multiple incompatible narratives equally well. In that case, the synthesis report explicitly says so, names the data limitations, and recommends what additional data (longer history, intraday data, additional sources) would resolve the ambiguity.

## Decision gates

After Phase 0: Stop and re-plan if the actual code is materially different from this plan's assumptions.

After Phase 2: Stop and fix if production-parity checks fail. The downstream analysis is worthless if the driver does not faithfully replay the production detector.

After Phase 4 but before Phase 5: Pause for a quick sanity review. If Phase 4's output looks "too clean" (e.g., one hypothesis explains everything with no residual), suspect a bug. Real diagnostics produce messy results.

After Phase 5: The synthesis report is the input to whichever follow-on planning doc gets written next. Do not begin v1 detector implementation before this report is in the repo.

## Scope decision (recorded during brainstorming)

The full 0-5 diagnostic is specified as one document. The downstream implementation plan (produced by the writing-plans skill) will chunk it into per-session execution batches. Phase 0 acts as the early gate: if the actual code differs materially from the plan's assumptions, re-plan before proceeding to Phase 2.

Alternative scopings considered and rejected:
- Phase 0+1 only (foundation): fragments planning work without clear benefit; Phase 0's findings can be folded into the writing-plans output.
- Split into two specs (driver+labelers vs analysis+synthesis): adds bureaucratic overhead; the artifacts of the first half are clear enough to feed the second half without a separate brainstorm.

## Appendix A - File and module touchpoints

Production code to be exercised (read-only):

- `src/strategies/advanced/market_regime_detector.py` -- subject under test
- `src/backtesting/regimes/detector.py` -- reused for G1, G3 labelers

New diagnostic code to be created:

- `scripts/diagnostics/regime_detector_replay.py` -- Phase 2 driver
- `scripts/diagnostics/ground_truth_labelers.py` -- Phase 3 labelers
- `notebooks/diagnostics/regime_detector_v0_analysis.ipynb` -- Phase 4 analysis
- `config/diagnostics/regime_events_2017_2026.csv` -- Phase 3 hand-curated labels

New artifacts to be produced:

- `diagnostics/data/spy_vix_2016_2026.parquet`
- `diagnostics/regime/v0/labels.parquet` (Hive-partitioned by year)
- `diagnostics/regime/ground_truth.parquet`
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md`

No production code changes. No edits to live trading configuration. No edits to `config/trading/strategy_toggle.yaml`.

## Appendix B - Why this is diagnosis, not treatment

The May 2026 root cause investigation already established that V1 (vanilla momentum, no regime overlay) beats V0 (regime-aware production) by ~0.24 Sharpe on EXT-OOS, and that V8 (V0 + BEAR-to-cash) beats V1 by another ~0.26 Sharpe but fails the cost-sensitivity test. The clearest reading of those results is that the highest-leverage fix is **what RAMP does on BEAR days**, not **how reliably the detector identifies them**.

So why run this diagnostic at all?

Three reasons. First, the detector's reliability bounds how much improvement BEAR-day cash logic can deliver. If `BEAR` fires only 15 days a year and misses the worst 10 of them, even perfect BEAR-day cash logic recovers only a small fraction of the potential improvement. Second, the diagnostic infrastructure built here is reusable for any future detector revision, so the cost amortizes across multiple investigations. Third, even if the detector is not the bottleneck for RAMP, it is also used by OMR (and any future RAMP-CSP and regime-conditional FX strategy), and improving its robustness pays off across the portfolio.

This plan is positioned as a prerequisite, not a competitor, to BEAR-day stock-selection work. Both can proceed in parallel.

## Appendix C - Open questions to resolve during Phase 0

These cannot be answered from documentation alone; they need direct inspection of the current code:

1. Does the detector accept raw SMAs or Kalman-smoothed trend estimates? The memory note about three parallel Kalman filters suggests the latter, but the docs and the OMR strategy file show the former. Phase 0 resolves this.
2. Is the VIX percentile window hard-coded at 252 days or parametrizable via constructor? Affects how H3 is tested.
3. What does the detector do when SPY or VIX data is missing for the current day? Does it fall through to the previous day, raise, or return a sentinel?
4. Are there any code paths (e.g., volatility spike detection mentioned in the architecture doc) that are documented but not actually implemented, or implemented but not documented?
5. Is there any test coverage for the detector that fixes its behavior (and that we'd need to update if we eventually revise it)?

These belong in Phase 0's written summary at the top of the eventual diagnostic report.
