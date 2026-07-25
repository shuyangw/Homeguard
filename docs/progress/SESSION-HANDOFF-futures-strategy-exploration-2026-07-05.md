# Session Handoff: Futures Strategy Exploration (de-concentration -> Sharpe-uplift -> deploy candidate)

**Date:** 2026-07-05 · **Working dir:** `C:\Users\qwqw1\Dropbox\cs\github\Homeguard` · **Branch:** `main` @ `60fb125` (pushed to origin, aligned)

## Resume Here (read this first)
- **Goal:** Get a futures strategy through the PSR/DSR/PBO statistical gate, then lift its OOS Sharpe as high as honestly possible (target was >1.0).
- **Status:** DONE for now, all merged + pushed. Best deployable book went from nothing-through-gate to **IDM-carry (OOS Sharpe 0.76, PBO 0.19)** to a **carry + 15%-crypto satellite blend (~0.79-0.81, PBO 0.177)**. Sharpe >1.0 NOT reached (would need imprudent ~45% crypto weight); we refused to fit. A **daily-panel cache (A1)** made walk-forwards ~165x faster (47min -> 17s) and killed the OOM.
- **Next steps (deploy-readiness for the 0.81 blend, none started):**
  1. **1.5x cost gate on the blend** (`satellite_blend.py` currently sets `oos_sharpe_1_5x_cost = oos_sharpe` placeholder, 1x only). Needed for the full methodology gate.
  2. **Best-of-N deflation** on 0.81 vs 0.76 (multiple pillars/weights were tried this campaign).
  3. **Crypto capacity** check (2 CME instruments, BTC/ETH ADV/OI limits on the satellite sleeve).
  4. **Portfolio integration** (corr vs OMR/RAMP/CSCM, marginal Sharpe), then IBKR paper. This is D10.
- **Open research leads:** B5 perp-funding crypto carry (external data, stronger than CME calendar); C7 multi-horizon carry; C8 buffering.
- **Blockers / open questions:** None blocking. Crypto is low-confidence (2 roots, 7 windows) but B6 showed its edge is regime-spread (not one bull run).
- **To resume, you need:** conda env `fintech` (`/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`), `PYTHONPATH=.`, the 8-thread env prefix (below). No auth. Working TODO (UNTRACKED, do not commit): `docs/progress/20260704_NEXT_STEPS_TODO.md`.

## Original Task (verbatim intent, across the session)
Session resumed mid-flight on the carry de-concentration campaign, then the user drove: "what else can we test" -> Sharpe-uplift campaign; "push hard for >1.0" via crypto; "why are we exploring crypto wrt futures" (answered: our crypto IS CME futures, only uncorrelated diversifier available); ran an unattended overnight experiment sweep; "how much RAM did we even use" (led to A1 diagnosis); "Merge and push"; "Continue on" (B4); "Do 1 then 2" (merge B4, do B6); "Push and write a comprehensive doc"; then this handoff.

## Subtasks & Progress
- [x] **Carry de-concentration campaign** (resumed) -- XS carry WEAK (0.77/PBO 0.46), **IDM-carry PASS (0.76/0.19)** first gate-clear, XS+IDM WEAK (0.77/0.53). Merged earlier (`fbefab3`).
- [x] **Trend diagnosis** -- the -0.45 was a STALE pre-fix artifact (negative-equity pct_change explosion, since fixed by bankruptcy floor). Real trend = **0.11/0.44 WEAK**. Demoted to crisis-insurance.
- [x] **Value (Asness 5yr-to-1yr reversal)** built + tested -- **REJECT (-0.22/0.64)**. Sign NOT flipped post-hoc (pre-registration).
- [x] **Crypto CME-calendar carry** (BTC/ETH) built -- **PASS (0.61/0.24)**, rho(crypto,carry) = **-0.065** (uncorrelated).
- [x] **Naive combine** (crypto as IDM cluster) = 0.42 (over-allocated, HURTS). **Capped** (idm_cap 1.5) = 0.55. Both < 0.76.
- [x] **A1 daily-panel + roll-volume cache** -- per-window RSS 5.6GB->0.33GB, WF 47min->17s (~165x), byte-identical. Merged+pushed (`acd0db7`).
- [x] **B4 core-satellite blend** (carry 85% + crypto 15%, pre-registered) = **0.81/PBO 0.177**, beats 0.76 on both axes. Merged (`a88d5ba`).
- [x] **B6 robustness (subperiod)** -- crypto positive in 2020-21 AND 2023-26 (not one regime); blend adds across regimes; 2022 crash drag contained to -0.05. Reassuring.
- [x] **Comprehensive review doc** written + pushed (`60fb125`).
- [ ] **Deploy-readiness** (1.5x cost gate on blend, deflation, capacity, portfolio integration) -- NOT started.
- [ ] **B5 perp-funding data / C7 multi-horizon / C8 buffering** -- NOT started.

## Key Decisions & Tradeoffs
- **Pre-registered 15% crypto weight, no sweep.** Why: spanning-optimal weight is ~45% (imprudent for a fragile 2-instrument sleeve); sweeping weights = fitting. Tradeoff: captured ~0.81 not the theoretical ~1.0.
- **Core-satellite return-stream blend, NOT universe expansion.** Why: IDM gives a 2-root cluster a full 1/8 risk budget -> over-allocates crypto -> Sharpe cratered to 0.42. Blending two books at a small fixed weight is the honest harvest.
- **A1 Option 1 (cache daily RAW OHLCV, ratio-adjust on daily), not full-range adjusted cache.** Why: preserves per-window anchoring -> results byte-identical. Gate: float-equivalence + carry_idm 0.7646 byte-identical (both passed exactly, max diff 0.0).
- **Did NOT flip the value sign** after seeing -0.22 (raw was ~+0.22). Why: post-hoc sign flip = data-snooping. Recorded as a legitimate negative.
- **Diagnose before fixing.** The -0.45 trend and the "crypto OOM" were both mis-attributed until measured directly (mem_probe showed 5.59GB; the `BrokenProcessPool` in throwaway scripts was a `__main__`-guard spawn bomb, not OOM).

## Discussion Summary
Campaign arc: carry sat just over the PBO gate (0.33); IDM sizing-side de-concentration cut it to 0.19 (first pass), signal-side XS backfired. To lift Sharpe, needed a second pillar. Data scoping killed skew (options=ES/NQ only) and universe expansion (redundant micros); value failed; **crypto carry was the lone uncorrelated diversifier** (crypto = CME futures, orthogonal to macro). Spanning math: carry 0.76 + crypto 0.61 at rho -0.065 could reach 1.007 at OPTIMAL weights, but that needs ~45% risk in crypto (imprudent). Naive/capped combines over-allocated crypto and hurt. The **core-satellite blend at a prudent pre-registered 15%** delivered ~0.81 with better PBO. B6 confirmed the crypto edge is regime-spread. Mid-campaign, the crypto walk-forwards kept dying (~15min or BrokenProcessPool); the user's "how much RAM" question triggered the A1 investigation: each WF window re-aggregated raw 1-min data TWICE (continuous close + per-contract roll-volume); caching both to disk fixed it (165x faster, OOM gone, byte-identical). We refused to chase >1.0 by fitting weights; produced a disciplined, documented improvement instead.

## Commands & Outputs (load-bearing)
8-thread prefix used on all runs:
```
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe ...
```
A1 acceptance gate (both caches, jobs=8):
```
CARRY_VERIFY Sharpe 0.7646484 (expect 0.7646) | PBO 0.188656 (expect 0.1887) | MATCH=True   [17s, was ~47min]
mem_probe: PEAK RSS 0.33 GB (was 5.59) | 8-way ~2.6 GB (was ~45)
35-root carry+crypto via CLI jobs=8: oos_sharpe=0.4217 pbo=0.1019  [18s, was 15.8min OOM]
```
B4 blend result:
```
run_satellite_blend.py --core-config carry_idm_broad.yaml --sat-config crypto_carry_broad.yaml --sat-weight 0.15
core carry 0.7646/PBO0.1887 | crypto 0.6130/PBO0.2433 | BLEND@15% 0.8095/PBO0.1768 (skew 1.21 kurt 21.7)
```
B6 subperiod (carry / crypto / blend@15%): 2020-21 1.59/+1.16/1.76; 2022 0.52/-1.57/0.47; 2023-26 0.69/+0.67/0.83; full 0.75/0.57/0.79. Crypto per-window Sharpes: [2.05, 0.27, -1.57, 0.20, 1.70, 0.31, -1.88].
Build cache (one-time, jobs=4 to avoid build-time OOM):
```
scripts/data/build_daily_raw_cache.py --roots <35> --jobs 4     -> H:\Stock_Data\futures\daily_raw (35 parquets)
scripts/data/build_roll_volume_cache.py --roots <35> --jobs 4   -> H:\Stock_Data\futures\roll_volume
```

## Git / commits (all on `main`, pushed to origin `shuyangw/Homeguard`)
- De-concentration merged earlier -> `fbefab3`.
- Sharpe-uplift campaign: crypto carry `4f2a1cc`, value `4854afa`, idm_cap `f56455c`, correlation tool `5ff47e9`, configs `7eb100f`; campaign merge `688dff7`.
- A1 cache: builder `c94efc7`, ratio_adjust_daily `a0bfd47`, wiring `89282dd`, roll-volume `acd0db7` (A1 merge = `acd0db7`, pushed).
- B4 blend: dated returns `8e5e580`, blend module `1bf6a92`, runner `e4d8dbc`, dedupe fix `a88d5ba` (B4 merge = `a88d5ba`, pushed).
- Comprehensive review doc `60fb125` (HEAD, pushed).
- Feature branches (all merged to main, can be deleted): `feat/futures-sharpe-uplift`, `feat/futures-daily-panel-cache`, `feat/crypto-satellite-blend`.

## Files Touched (key code)
- `src/data/carry_calculator.py` -- crypto asset_class branch (roll yield); bond FRED carry.
- `src/data/futures/asset_class.py` -- CLUSTER/ASSET_CLASS incl. BTC/ETH crypto.
- `src/data/futures/paths.py` -- `daily_raw_dir()`, `roll_volume_dir()`.
- `src/data/continuous_contract_loader.py` -- `ratio_adjust_daily` + daily-raw cache wiring + roll-volume disk cache (the A1 core).
- `src/strategies/advanced/futures_value_strategy.py` -- FuturesValue (Asness reversal).
- `src/strategies/advanced/futures_carry_strategy.py` -- FuturesCarryXS (from de-concentration).
- `src/backtesting/utils/idm_weights.py` -- `compute_div_mult` + optional `per_instrument_cap`.
- `src/backtesting/blend/satellite_blend.py` -- `blend_books` (core-satellite blend + gate). NOTE 1.5x placeholder.
- `scripts/backtest_scripts/run_carver_walkforward.py` -- `idm`/`idm_cap`/`return_window_returns` threading, `_oos_returns_dated`.
- `scripts/backtest_scripts/run_satellite_blend.py` -- blend runner (has `__main__` guard).
- `scripts/data/build_daily_raw_cache.py`, `scripts/data/build_roll_volume_cache.py`.
- Configs: `config/backtesting/{carry_idm_broad,crypto_carry_broad,value_broad,carry_idm_crypto,carry_idm_cap15}.yaml`.
- Docs: `docs/strategies/research/20260705_FUTURES_STRATEGY_EXPLORATION_REVIEW.md` (the comprehensive review), campaign spec/plans, `docs/progress/20260704_{OVERNIGHT_RESULTS,SHARPE_UPLIFT_PHASE1_SUMMARY}.md`.

## Key Takeaways & Gotchas
- **`__main__` guard REQUIRED** for any direct multiprocessing script on Windows (ProcessPoolExecutor re-imports the main module; no guard -> spawn bomb -> `BrokenProcessPool`, easily mis-read as OOM). All shipped runners have it; my throwaway verify scripts initially did not.
- **A1 fixed the memory + speed**, so `--jobs 8` now works for crypto/35-root runs (was serial-only). The prior ~15min CLI kill was memory-driven, now resolved.
- **~60-min background-job cap** and **8-thread cap** still apply to any NEW heavy work. Prefix env vars; one long run per bg job.
- **Vol normalization in the blend is full-sample (disclosed simplification).** Causal/trailing-vol is a later refinement. The blend's boundary-date dedupe (`~index.duplicated(keep=first)`) is required (reindex rejects duplicate labels) and does not distort the apples-to-apples comparison.
- **Report all outcomes, no survivorship.** Every rejected trial (value, XS, trend, naive/capped crypto) is in the results log.
- **Memory reality:** machine 66 GB; pre-cache crypto window 5.59 GB (8-way ~45 GB, marginal OOM); post-cache 0.33 GB (8-way ~2.6 GB).

## References
- Comprehensive review: `docs/strategies/research/20260705_FUTURES_STRATEGY_EXPLORATION_REVIEW.md`
- Results log (all raw numbers): `docs/progress/20260704_OVERNIGHT_RESULTS.md`
- Untracked TODO (next steps): `docs/progress/20260704_NEXT_STEPS_TODO.md`
- Repo: https://github.com/shuyangw/Homeguard (main @ 60fb125)
- Memory: `feedback_background_job_60min_cap.md`, `feedback_parallel_thread_cap.md`
