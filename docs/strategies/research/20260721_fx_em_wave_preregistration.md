# FX Emerging-Market Wave -- Pre-Registration

**Date:** 2026-07-21 | **Status:** LOCKED 2026-07-21 (approved; no post-hoc edits to specs/params/gate) | **Owner:** main-loop -> strategy-lead for verdicts

This document is a PRE-REGISTRATION per the North Star. It fixes the hypotheses,
universe, exact specs, validation gate, trial count, and PASS/FAIL BEFORE any
backtest is run. Once locked, the set below IS the search; specs are not added,
dropped, or re-parameterized after seeing results (doing so is p-hacking and is
refused). Every spec run is a counted trial that deflates the DSR.

## 1. Motivation and why this is NOT re-running failed G10 mechanisms

G10 carry (#15/#16) and G10 trend/momentum (#3/#4) already FAILED under honest
walk-forward net of costs. Re-running the same mechanisms on a new universe
purely hoping for a better number would be universe p-hacking. The pre-registered
economic reason EM is a genuinely different test, not a re-roll:

- **Carry magnitude and basis differ by an order of magnitude.** G10 rate
  differentials are ~0-4% and the G10 carry premium compressed post-2008 (well
  documented decay). EM differentials are 1.5-38% (MXN 6.8, ZAR 7.1, PLN 3.9,
  HUF 6.0, CNH 1.5, TRY 38.8, INR 5.5 vs USD ~4.3). The EM carry premium is a
  distinct, better-documented risk premium: compensation for crash/political/
  convertibility risk, not the same faded G10 signal.
- **The real question is adversarial, not optimistic.** EM carry pays precisely
  BECAUSE EM currencies periodically devalue 20-40% (TRY 2018/2021, ZAR selloffs).
  The pre-registered hypothesis is therefore: *does the EM carry premium survive
  its own crash risk AND realistic (wide) EM transaction costs, or does it get
  eaten -- the EM analogue of G10 carry decay?* A negative result is a success
  (surfacing "EM carry dies after costs/crashes" completes the objective).
- **EM trend/reversion dynamics are structurally different** (persistent
  depreciation trends, managed-float regimes, discrete devaluations), so testing
  trend/mean-reversion on EM is a distinct pre-registered idea, not a G10 re-roll.

## 2. Universe and data (validated 2026-07-21)

**EM7 = USDMXN, USDZAR, USDPLN, USDHUF, USDCNH, USDTRY, USDINR** (USD-per-foreign
convention; "long the EM currency" = SHORT the pair). BRL excluded (Massive-only
spot with holiday thin-print artifacts, not Dukascopy-backfillable).

Spot data: G10-grade daily cache, independently validated vs yfinance + FRED H.10
(see `docs/progress/20260721_fx_em_cache_backfill.md`). ZAR artifact-fixed. INR is
a NON-DELIVERABLE currency: spot is clean but retail-untradeable; INR is included
as a SIGNAL leg for cross-sectional/basket breadth, flagged untradeable, and any
INR-dependent PASS must be re-checked NDF-tradeable before deployment.

Short rates (FRED, decimal, ffilled to daily), current through 2026-05/06:

| CCY | FRED series | Family | Latest |
|---|---|---|---|
| USD | DFF | policy | ~4.3% |
| MXN | IR3TIB01MXM156N | OECD 3M interbank | 6.76% |
| ZAR | IR3TIB01ZAM156N | OECD 3M interbank | 7.11% |
| PLN | IR3TIB01PLM156N | OECD 3M interbank | 3.85% |
| HUF | IR3TIB01HUM156N | OECD 3M interbank | 5.98% |
| CNH | IR3TIB01CNM156N | OECD 3M interbank (onshore proxy) | 1.51% |
| TRY | INTDSRTRM193N | CBRT discount (fallback) | 38.75% |
| INR | IRSTCI01INM156N | call money (fallback) | 5.50% |

Rate caveats pre-registered: CNH uses onshore China 3M as an offshore proxy;
TRY/INR use a different rate family (discount/call-money) than the OECD interbank
core, introducing a small basis. Acceptable for carry-DIFFERENTIAL signals where
EM spreads are 5-15x the basis; noted, not hidden.

## 3. The wave -- 6 pre-registered specs

Sign/vol conventions shared: forecasts vol-scaled to ~0.03/instrument, portfolio
vol cap applied (EM pairs are highly USD-correlated -- a book-level cap is
required, per the 2026-07-06 G10 calibration note). All run on the existing daily
`forecast_panel` + `FxSpotPortfolioSimulator` engine.

1. **EM-CARRY** (primary). Forecast_i = annualized (r_i - r_USD) carry signal;
   vol-targeted continuous basket, long positive-carry EM / short USD. Rebalance:
   weekly AND daily (2 trials). Mirrors #15 mechanism on EM's large differentials.
2. **EM-CARRY-SEATBELT** (primary variant). Spec 1 + crash filter: reduce/flatten
   a leg when its carry-unwind score fires (trend-down + vol-spike), reusing
   `src/backtesting/signals/carry_unwind.py` generalized to EM. Motivated because
   EM carry's dominant risk is the crash. Weekly (1 trial).
3. **EM-TSMOM**. Time-series momentum, continuous EWMA trend forecast (reuse
   FxTrend/FxTSMOM param family: fast/slow EWMA crossover, vol-targeted). Weekly
   (1 trial).
4. **EM-XSMOM**. Cross-sectional momentum: rank EM7 by trailing 3M (63d) return,
   long top-2 / short bottom-2, market-neutral within EM. Weekly (1 trial).
5. **EM-CARRY-MOM**. Additive combination of the spec-1 carry and spec-3 momentum
   forecasts (equal-weight blend), the classic EM carry+momentum diversification.
   Weekly (1 trial).
6. **EM-MEANREV**. Daily CLOSE-only z-reversion (z of price vs 60d rolling mean;
   enter when |z|>2, exit at z->0; per-pair, vol-targeted). Uses close only to
   avoid the unbuilt OHLC-into-forecast_panel wiring. Weekly (1 trial).

**Pre-committed trial count for this wave: 7** (spec 1 = 2 cadences; specs 2-6 =
1 each). No parameter sweeps beyond what is listed; the params above are fixed
before running. Any additional variant discovered mid-run is a NEW trial and must
be pre-registered as a wave-2, not folded into these results.

## 4. Validation protocol (authoritative: docs/methodology/backtesting.md)

- **Walk-forward** with purging + embargo (Section 3), same window scheme as
  `run_fx_walkforward.py`. FULL available data range (2011-2026 for most pairs;
  CNH from 2014). IDM on.
- **Combined statistical gate** (Section 2): Sharpe, PSR (>0.95), DSR>0 using the
  PROJECT-WIDE cumulative trial count (prior FX campaign trials + this wave's),
  PBO (<0.5).
- **Cost model, EM-specific.** EM spreads are far wider than G10 and are the
  crux of whether carry survives. Pre-committed conservative per-side half-spreads
  (bps of notional), applied every rebalance: MXN 3, ZAR 6, PLN 4, HUF 5, CNH 5,
  TRY 15, INR 8 (INR NDF-indicative). PLUS the mandatory 1.5x cost-sensitivity
  gate (Section 4): a spec that only passes at 1x FAILS.
- **Benchmark bar:** OOS Sharpe vs S&P 500 buy-hold (~0.68, the wave-2 bar) and a
  market-neutrality/correlation check (a market-neutral EM spread with low S&P
  corr is valued on marginal book contribution, not standalone level).
- **Fill logging MANDATORY** (run-scoped FillSink -> trades_oos.csv.gz), per the
  standing mandate. A run that discards fills is rejected.

## 5. Pre-committed PASS / FAIL

- **PASS** (all must hold): OOS Sharpe > 0 AND positive at 1.5x cost AND PSR>0.95
  AND DSR>0 (deflated for cumulative N) AND PBO<0.5 AND (beats S&P bar OR shows
  positive deflated marginal book contribution at low S&P corr).
- **FAIL**: any of -- non-positive OOS Sharpe, sign-flip/negative at 1.5x cost,
  DSR~0, PBO>=0.5. A near-miss needing any post-hoc degree of freedom to reach the
  gate is a FAIL, not a "promising lead."

## 6. Stopping rule

If all 7 trials FAIL net of costs under honest walk-forward, the EM carry/trend
catalog extension is declared exhausted and we STOP (no wave-2 EM, no ML) -- the
finding ("EM carry/trend dies after realistic EM costs + crash risk") is recorded
as a completed objective. If a spec PASSES, it proceeds to book-level evaluation
(marginal deflated cost-net contribution vs the existing portfolio) before any
deployment consideration. Optional-stopping is refused: we do not halt early the
instant one spec passes, nor keep going past the 7 hunting for one that does.

## 7. Prerequisite BUILD tasks (subagent-driven-development; NOT verdicts)

These are build/plumbing, executed before the verdict phases; verdict phases go
through strategy-lead.

1. **Wire EM rate series** into `CURRENCY_FRED_SERIES` (src/data/fx_rates.py): add
   PLN/HUF/CN/TR/IN (MXN/ZAR already mapped); fetch each to the local FRED cache
   `alt_data/fred/<id>/daily.parquet` via the keyless FRED path. Validate each is
   current (no stale-series bug) at load.
2. **EM cost model**: encode the Section-4 per-pair half-spreads in the FX cost
   config the WF runner reads (EM spreads, not G10 defaults).
3. **EM universe wiring**: confirm EM7 loads through `load_fx_daily_panel` and the
   `run_fx_walkforward.py` universe hook; portfolio-level vol cap enabled.
4. Spec 6 uses close-only z-reversion (no OHLC dependency); if OHLC Bollinger/
   Keltner is preferred later, that is a separate wave requiring the OHLC-into-
   forecast_panel build.

## 8. Trial accounting note

Prior FX campaign: 12 gated specs across 8+ mechanisms (all FAIL), plus quick
in-sample screens. This wave adds 7 trials. The DSR for every spec in this wave
is computed against the updated cumulative N. N is logged in the experiment
registry per methodology Section 9; it is never undercounted to help a spec pass.
