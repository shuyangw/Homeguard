# Futures Sharpe-Uplift Campaign - Phase 1 Summary (2026-07-04)

**Branch:** feat/futures-sharpe-uplift (NOT merged, NOT pushed - your review).
**Spec:** docs/strategies/research/20260704_SHARPE_UPLIFT_CAMPAIGN_SPEC.md.
**Running log (all raw numbers):** docs/progress/20260704_OVERNIGHT_RESULTS.md.

## Bottom line
**We did NOT reach OOS Sharpe > 1.0. The incumbent stays IDM-weighted carry at 0.76.**
The honest, valuable finding: **crypto carry is a genuine uncorrelated second pillar, but it
cannot be harvested into a Sharpe uplift under any pre-committed sizing we tried** - it improves
robustness (PBO) while diluting Sharpe. Value failed outright. Trend stays crisis-insurance only.

## Goal
Lift the futures book from IDM-carry (OOS Sharpe 0.76, PBO 0.19 - our best gate-passing
strategy) toward > 1.0 without losing the PBO < 0.25 gate. You directed: push hard for > 1.0,
prioritizing crypto-carry (the only lever with a plausible path per the spanning math).

## What was built (all committed, TDD + reviewed)
- Crypto CME-calendar carry: `crypto` branch in CarryCalculator + BTC/ETH cluster/asset_class maps (4f2a1cc).
- FuturesValue signal (Asness 5yr-to-1yr reversal) - built earlier (4854afa).
- Standalone configs (crypto_carry_broad, value_broad) + pillar-correlation tool (7eb100f, 5ff47e9).
- Per-instrument IDM div_mult cap (`idm_cap`), threaded through sizing + walk-forward (f56455c).

## Experiment results (ALL of them - no survivorship)

| Experiment | OOS Sharpe (1x/1.5x) | PBO | kurt | Verdict |
|---|---|---|---|---|
| carry_idm (INCUMBENT) | 0.76 / -- | 0.19 | 22.2 | gate PASS, best book |
| VALUE standalone (Asness reversal) | -0.22 / -0.23 | 0.64 | 8.3 | REJECT -> EXCLUDE |
| CRYPTO carry standalone (BTC/ETH) | 0.61 / 0.61 | 0.24 | 11.3 | PASS, qualifies as pillar |
| carry + crypto (naive IDM cluster) | 0.42 / 0.40 | 0.10 | 27.6 | worse Sharpe, better PBO |
| carry + crypto (per-instrument cap 1.5) | 0.55 / 0.53 | 0.086 | 21.9 | worse Sharpe, better PBO |
| carry + cap 1.5 (no crypto, Minor-4) | 0.71 / 0.71 | 0.186 | 23.9 | no improvement vs 0.76 |

Correlation: rho(crypto, carry) = **-0.065** (near-zero -> crypto qualifies for FULL weight tier).

## Why crypto does not help (the core finding)
- Spanning bound with carry 0.76 + crypto 0.61 at rho -0.065 = **1.007** IF optimally weighted.
- But that bound needs in-sample-optimal weights (fitting). Our pre-committed IDM equal-cluster-
  risk gives the 2-root crypto sleeve a full 1/8 of the book's risk (same as the 8-root fx
  cluster). Crypto's high vol -> it dominates the book's risk far beyond its 0.61/low-rho merit;
  its 2022 crash injects tail risk (kurt 22.2 -> 27.6). Net: Sharpe DROPS 0.76 -> 0.42.
- The per-instrument cap (1.5) trims crypto 2.5x -> 1.5x, recovering Sharpe to 0.55 and fixing
  kurtosis (21.9), but still < 0.76. Even capped, crypto over-allocates.
- Harvesting crypto would need a SMALL pre-committed satellite weight (portfolio-level, not IDM
  cluster) - but choosing that weight edges into fitting, which we did NOT do (per spec/strategy-
  lead integrity rules). So we STOPPED rather than sweep weights toward a target.

## Robustness-vs-Sharpe note
The combined books (capped 0.55 / PBO 0.086) are MORE robust than carry alone (0.76 / PBO 0.19).
A risk-averse mandate might prefer the more-robust lower-Sharpe book. For the Sharpe>1 goal, none beat carry.

## Bugs found (real, logged; not yet fixed)
1. **Parallel OOM**: 8 workers each loading crypto's large 1min data (BTC 2017+/ETH 2021+) on top
   of 33 macro roots exhausts RAM -> a worker is OOM-killed -> `BrokenProcessPool`. Serial (max_workers=1)
   runs all 13 windows fine. Real fix: cache daily panels / cap crypto worker memory / lower --jobs
   for crypto-inclusive runs. (Combined gates were computed serially to get the numbers.)
2. **CLI walk-forward killed ~13-16min in (GENERAL, not crypto-specific)**: three CLI runs were
   reaped ~13-16min in, right after the standard-report save -- including a 33-root NO-crypto run
   (carry_idm_cap15). So it is NOT the crypto OOM (that is bug #1) and NOT the 60-min bg cap. Cause
   unresolved (CLI late-phase instability or external reaper). WORKAROUND: direct serial gate scripts
   (walk_forward_carver max_workers=1) complete reliably every time -- all combined/gate numbers here
   were obtained that way. Some CLI runs (value, plain carry) did complete; the trigger is not fully pinned.

## DSR trial-count (project-wide N)
Pre-committed pillar trials this campaign: value (N+1), crypto standalone (N+1), plus combination
+ cap variants. Each parameter-free (trial_count=1 per run). Best-of-2 pillar selection (crypto
over value) is mild; crypto's PBO 0.24 is thin and its sample short (2 roots, 7 windows, regime-heavy)
- so even the standalone PASS is LOW-CONFIDENCE and would need a proper deflation before any deploy.

## Recommendation / next steps (your call)
1. **Bank carry_idm (0.76) as the futures deploy candidate.** It is the honest best; > 1 is not
   reachable on this data without weight-fitting.
2. If pursuing crypto further: (a) proper portfolio small-sleeve sizing (a real Phase-4 combiner
   with a pre-registered satellite weight), (b) acquire perp funding-rate data (stronger crypto
   carry than CME calendar), (c) fix the parallel OOM so crypto runs are not serial-only.
3. Incremental carry polish: the per-instrument IDM cap 1.5 was TESTED on carry alone -> 0.71
   (WORSE than 0.76; capping beneficial LE/HE/ES concentration hurts). So IDM-cap is NOT a win for
   carry. Multi-horizon / buffering remain untested (need combiner / buffer build); strategy-lead
   estimate +0.02-0.08 each, not > 1. Lower priority.
4. Nothing merged/pushed - review the branch and decide what (if anything) lands on main.
