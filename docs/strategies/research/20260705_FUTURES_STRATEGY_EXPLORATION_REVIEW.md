# Futures Strategy Exploration - Comprehensive Review (2026-07-01 .. 2026-07-05)

**Scope:** Several days of systematic-futures research: building a dedicated futures backtest
harness, testing signal families through the statistical gate, and lifting the best book from a
gate-pass toward higher Sharpe. This is the authoritative chronicle -- what we tested, what the
numbers were, what worked, what didn't, and why.

**One-line outcome:** Best deployable futures book went from *nothing through the gate* to
**IDM-weighted carry (OOS Sharpe 0.76, PBO 0.19)**, then to a **carry + 15%-crypto satellite blend
(~0.79-0.81, PBO 0.177)** -- a genuine, non-fitting, regime-robust improvement. Sharpe > 1.0 was
NOT reached and is not honestly reachable on this data without imprudent concentration.

---

## 1. The methodology (how every number below was judged)

All strategies are **parameter-free by doctrine** (Carver scalars/speeds/caps, carry conventions,
IDM/FDM constants are fixed, never optimized) so the DSR trial count stays honest. Every result is
a **walk-forward** OOS number (train 36m / test 12m / step 12m, purged), judged by the combined
statistical gate (`docs/methodology/backtesting.md`): **PSR > 0.95, DSR > 0.95, PBO < 0.25**, plus a
**1.5x cost-sensitivity gate**. Runs are 8-thread capped, RunStatus-tracked, trade-logged, and
registered in `output/experiments.duckdb`. No weight or parameter was ever fit to a Sharpe target;
every construction and combination weight was **pre-registered before seeing its result**.

---

## 2. Signal families tested (walk-forward OOS, 33-root broad basket unless noted)

| Family | Construction | OOS Sharpe | PBO | Verdict |
|---|---|---|---|---|
| Carver TSMOM (trend) | multi-speed EWMAC, vol-normalized | 0.11 | 0.44 | WEAK (see 4.2) |
| Absolute carry | Carver roll-yield carry, EWMA-smoothed | 0.85 | 0.33 | just over gate |
| **IDM-weighted carry** | carry + cluster-risk div_mult sizing | **0.76** | **0.19** | **PASS (first gate-clear)** |
| XS carry | within-asset-class demeaned carry | 0.77 | 0.46 | WEAK |
| XS + IDM | both de-concentration levers | 0.77 | 0.53 | WEAK (worst) |
| Value | Asness 5yr-to-1yr reversal | -0.22 | 0.64 | REJECT |
| Crypto carry | CME BTC/ETH calendar roll-yield | 0.61 | 0.24 | PASS (uncorrelated pillar) |

---

## 3. Combination results (harvesting crypto's diversification)

Crypto carry's daily-return correlation with the carry book is **rho = -0.065** (near-zero) -- the
key property that made it worth combining. Combination outcomes:

| Combination | Method | OOS Sharpe | PBO | Note |
|---|---|---|---|---|
| carry + crypto (naive) | crypto as a full IDM cluster (1/8 risk) | 0.42 | 0.10 | over-allocated -> HURTS Sharpe |
| carry + crypto (capped) | per-instrument div_mult cap 1.5 | 0.55 | 0.086 | better, still < 0.76 |
| **carry + crypto @15%** | **core-satellite return-stream blend** | **~0.81** | **0.177** | **beats 0.76 on BOTH axes** |

The **spanning bound** (optimal in-sample weights) was 1.007, but reaching it needs ~45% risk in
crypto -- imprudent for a 2-instrument, short-history sleeve. A **prudent 15% satellite** (weight
pre-registered) captures a real, defensible slice: +0.05 Sharpe and better PBO.

---

## 4. Key findings (the "why")

### 4.1 De-concentration: sizing-side works, signal-side backfires
Carry sat *just over* the 0.25 PBO gate (0.33). Two orthogonal, parameter-free de-concentration
levers were tested as 3 pre-committed trials. **IDM (sizing-side cluster risk-weighting) cut PBO
0.33 -> 0.19** for a modest Sharpe give-up and flipped skew positive -- carry's first gate-clear.
**XS (signal-side within-class demeaning) BACKFIRED** (PBO up to 0.46 alone, 0.53 stacked): turning
absolute carry into relative-value carry produced a less window-stable signal. Lesson:
de-concentrate risk, don't reshape the signal.

### 4.2 Trend is honestly weak here, and the scary number was a ghost
Trend initially showed OOS Sharpe **-0.45** -- alarming for a premium as robust as futures trend.
Diagnosis: it was a **stale pre-fix artifact** of a negative-equity `pct_change` explosion (the
simulator let equity cross zero; a since-added bankruptcy floor fixed it). The corrected trend is
**+0.11** -- weak but correctly signed, consistent with the 2011-2019 trend drought. Trend is
retained only as a small crisis-insurance sleeve; adding real weight to it LOWERS book Sharpe (the
combination arithmetic: a weak diversifier drags a strong book down).

### 4.3 Value failed; crypto is the one real diversifier in our data
Data scoping killed most second-pillar candidates: **skew is blocked** (futures options on disk
cover only ES/NQ), **universe expansion is mostly redundant micros** (a Micro-ES is the same bet as
ES). **Value** (the one price-only candidate) came in **negative (-0.22)** -- the Asness reversal
was anti-predictive on this basket 2015-2026 (long-horizon momentum persisted; the raw signal was
~+0.22, but per pre-registration we did NOT flip the sign after seeing the result). **Crypto carry**
was the lone genuinely uncorrelated diversifier (crypto price/carry is driven by forces orthogonal
to macro futures), which is exactly why it earned a place where value did not.

### 4.4 The harvest lesson: size crypto as a small satellite, not an IDM cluster
Naive universe-expansion (crypto as an 8th IDM cluster) gave the 2-root crypto sleeve a full 1/8 of
the book's risk -- its high vol + the 2022 crash then dominated, dropping Sharpe 0.76 -> 0.42 (PBO
improved, but Sharpe cratered; kurtosis rose 22 -> 28). A per-instrument cap helped (0.55) but still
lagged. The **core-satellite blend** (carry core 85%, crypto satellite 15%, blended at the
return-stream level) is the right mechanism: **~0.81 Sharpe, PBO 0.177**, beating carry on both axes.

### 4.5 Robustness (B6): the crypto edge is regime-spread, not one bull run
Subperiod Sharpe (carry / crypto / blend@15%): 2020-21 bull **1.59 / +1.16 / 1.76** (+0.17); 2022
crash **0.52 / -1.57 / 0.47** (-0.05); 2023-26 recovery **0.69 / +0.67 / 0.83** (+0.14). Crypto is
positive in TWO separate periods (not pure regime luck), the blend adds in normal/bull regimes, and
the small 15% weight contained the 2022 crash drag to -0.05. Per-window crypto Sharpes span
[2.05 .. -1.88] -- volatile but not a single-window fluke.

---

## 5. Infrastructure built / fixed

- **Futures backtest harness** (dedicated path, separate from equity/crypto): `run_futures_backtest`
  + `FuturesPortfolioSimulator.run_sized` with **equity-feedback sizing + bankruptcy floor** (fixed
  the negative-equity `pct_change` contamination that produced skew -30 / kurt 1332 phantom stats).
- **IDM sizing** (`compute_div_mult`): cluster equal-risk weights x fixed-correlation IDM x N_scale,
  plus an optional per-instrument `idm_cap`. **FDM/combiner groundwork** and a **satellite blend**
  (`blend_books`) for return-stream combination with the standard gate.
- **A1 daily-panel + roll-volume cache** (the keystone infra win): each walk-forward window used to
  re-aggregate raw 1-min data TWICE (continuous close + per-contract roll-volume). Caching both to
  disk dropped **per-window RSS 5.6 GB -> 0.33 GB** and **walk-forward time 47 min -> 17 s (~165x)**,
  with results **byte-identical** (equivalence-gated). This eliminated the parallel OOM and unblocked
  fast crypto iteration.
- **Trade-logging + RunStatus** made mandatory for all asset classes / long runs (a killed run now
  leaves a diagnosable stale-RUNNING sentinel).

### Bugs found and fixed
Negative-equity contamination (bankruptcy floor); loader silent-basket-shrink (over-broad
`except`); GC/CL/crypto cache stubs; inert bond carry (implemented FRED CMT-DFF); the 5.6 GB
per-window OOM (daily-panel cache); a `__main__`-guard spawn-bomb in throwaway verify scripts (not
shipped code) that masqueraded as an OOM; the ~15-min CLI kill (also memory-driven, resolved by the
cache).

---

## 6. Operational lessons / gotchas

- **~60-min background-job cap:** the harness reaps background runs at ~60 min; never chain multiple
  long runs in one job (one run per job, or split `--jobs`). Diagnose kills via RunStatus + event log
  BEFORE guessing OOM.
- **8-thread cap is a hard preference:** `--jobs` caps processes only; polars defaults to 32
  threads/process. Set `POLARS_MAX_THREADS=1 ... --jobs 8` for 8 threads total.
- **Windows spawn needs `if __name__ == "__main__"`:** direct multiprocessing scripts without the
  guard spawn-bomb into `BrokenProcessPool` -- easy to misdiagnose as OOM. Measure memory in-process
  before blaming RAM.
- **Diagnose before fixing:** the -0.45 trend "bug" and the "crypto OOM" were both partly
  mis-attributed until measured directly. Every fix here was preceded by a measurement.
- **Report all outcomes, no survivorship:** every rejected/weak trial is recorded, not just winners.

---

## 7. Deploy candidate + what remains

**Candidate: carry_idm (core) + crypto carry (15% satellite), ~0.79-0.81 OOS Sharpe, PBO 0.177.**
Provably clean (baseline byte-unchanged, blend causal), non-fitting, regime-robust.

**Before live deployment (deploy-readiness checklist):**
1. **1.5x cost gate on the blend** (currently 1x-only; a flagged placeholder) -- needed for the full gate.
2. **Best-of-N deflation** -- quantify the deflated significance of 0.81 vs 0.76 given the multiple
   pillars/combinations tried this campaign.
3. **Crypto capacity** -- 2 CME instruments; ADV/OI limits on the satellite sleeve.
4. **Portfolio integration** -- correlation vs OMR/RAMP/CSCM, marginal portfolio Sharpe, then IBKR paper.

**Open research leads:** perp-funding crypto carry (stronger than CME calendar; needs external data);
multi-horizon carry + buffering (~+0.02-0.08 each); empirical-C IDM.

---

## 8. Honest bottom line

We did not reach Sharpe > 1.0, and we resisted the temptation to fit our way there (the naive
"optimal" weight would have put ~45% risk in a fragile crypto sleeve). What we produced instead is a
**disciplined, documented improvement** -- from no gate-passing futures book, to IDM-carry at 0.76,
to a regime-robust ~0.81 core-satellite book -- plus a **165x-faster backtest stack** that makes the
next campaign cheap. Every failure (value, XS, trend, naive/capped crypto) is recorded, and the one
success is earned honestly.

## Key artifacts
- Results log: `docs/progress/20260704_OVERNIGHT_RESULTS.md`
- Campaign spec: `docs/strategies/research/20260704_SHARPE_UPLIFT_CAMPAIGN_SPEC.md`
- Phase-1 summary: `docs/progress/20260704_SHARPE_UPLIFT_PHASE1_SUMMARY.md`
- De-concentration: `docs/strategies/research/20260703_CARRY_DECONCENTRATION_DESIGN.md`
- Cache + blend plans: `docs/strategies/research/20260704_DAILY_PANEL_CACHE_PLAN.md`,
  `20260705_CRYPTO_SATELLITE_BLEND_PLAN.md`
- Code: `src/strategies/advanced/futures_carry_strategy.py`, `src/backtesting/utils/idm_weights.py`,
  `src/backtesting/blend/satellite_blend.py`, `src/data/carry_calculator.py`,
  `src/data/continuous_contract_loader.py`, `scripts/backtest_scripts/run_{carver_walkforward,satellite_blend}.py`
