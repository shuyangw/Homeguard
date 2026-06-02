# RAMP Wave-3 Signal-Construction Probe - 2026-06-01

## Summary

Ran the full 5-variant RAMP Wave-3 signal-construction probe (V31, V28, V26, V02+V05,
V33-core) end-to-end via a new G0.5-durable single-variant runner, then gated the family
together. **Three variants beat the incumbent V11 materially -- V28 multi-horizon ensemble
(+0.283 Sharpe), V31 beta-residual (+0.241), V02+V05 regime-free vanilla (+0.155) -- and all
three pass the 1.5x cost gate that V11 itself fails.** The family PBO = 0.503 (time-period
instability) makes a purged/embargoed walk-forward mandatory before any graduation. Verdict:
**V28 HOLD -> walk-forward** (V31 strong co-candidate); null option (ship V11) rejected.

Two data-integrity bugs were found and fixed mid-probe (see below); both invalidated the
first V31 result, which was re-run clean.

## Changes Made

- **`src/research/ramp_phase4/data.py`**: corrected stale `SIP_SPLIT_REL` path constant. The
  1-min SIP tree moved to `equities/sip_split/1min`; the stale constant made the loader
  silently fall through to the corrupt LEGACY daily cache (unadjusted NFLX 10:1 split,
  2025-11-17, -90% phantom return). Fix repoints at the live split-adjusted tree; the clean
  `daily_from_sip` cache (already on disk) is now used. V11's 0.528 baseline is unaffected
  (it ran 2026-05-23 before the tree moved). Closes G0.0 + G0.2.
- **`src/research/ramp_phase4/variants.py`**: added 5 Wave-3 variants + helpers --
  `_variant_v31` (beta-residual, 90d window), `_variant_v28` (multi-horizon 0.5/0.3/0.2 +
  0.1 reversal), `_variant_v26` (z-score winsorized, V27 bounded-penalty folded as a
  param/sensitivity), `_variant_v02_v05` (regime-free vanilla + min-hold), `_variant_v33_core`
  (abs-mom cash gate). Registered all in `REGISTRY`. Also fixed a dtype bug (`pct_change`
  on object-dtype columns -> `np.isnan` TypeError) that the corrupt cache had masked.
- **`scripts/backtest_scripts/ramp_phase4_wave3_readiness.py`** (NEW): G0.5-durable
  single-variant runner. Per-sub-backtest `append_run` to the experiment registry, resume-skip
  keyed on (variant, cost, timing, git SHA, snapshot), atomic `.tmp`+`os.replace` artifact
  writes. Closes G0.5. Also fixed the hardcoded misleading "turnover 39%" incumbent string.
- **`tests/research/ramp_phase4/test_variants.py`**: +27 TDD tests across the 5 variants
  (137 -> 159 passing... actual final count 159 with V33; suite green throughout).
- **Reports** (`docs/reports/ramp/20260601_wave3_*.md/.json`): per-variant readiness +
  `20260601_wave3_family_gate.md` (cross-section, PBO, DSR n_trials sensitivity, sub-window
  stability, cost gate, combined verdict).

## Key Findings

1. **Signal construction beats regime timing.** The 2026-05 detector campaign capped
   regime-timing gains at +0.08 Sharpe; signal construction delivered +0.28 (V28). The lens
   was right.
2. **Turnover "blowup" was a phantom.** Early per-variant verdicts flagged 5,000-10,000%
   turnover vs "V11's 39%". Running V11 through the SAME runner gave V11 AnnTO = 10,325% --
   the 39% used a different definition. All variants are the same order; V28 is actually the
   LOWEST (5,264%). Verdicts corrected.
3. **V28** (multi-horizon ensemble): Sharpe 0.811, PSR 0.993, lowest turnover, best cost
   profile (0.766 at 7.5bps), beats V11 in all 3 sub-windows. Top candidate.
4. **V31** (beta-residual): Sharpe 0.769, LOWEST max DD (-33.5%) -- directly attacks the
   H6/H8 high-beta-BEAR-loser root cause. Strong co-candidate.
5. **V02+V05** (regime-free vanilla): Sharpe 0.683, most sub-window-consistent. Beating V11
   while regime-free is direct support for H2 (regime apparatus may be net-negative).
6. **PBO = 0.503** -- family selection is time-period-unstable. Walk-forward mandatory.
7. **DSR** V28 passes at n_trials <= 12 (documented Wave-3 family-reset), fails at >= 36.

## Commits

- `1ea948d` feat(wave3): G0.5 durable single-variant Wave-3 readiness runner
- `7a07ef4`, `a88e762`, `44f9ffb`, `729f065` V31 (initial, later found contaminated)
- `429df47` fix(wave3): correct stale SIP_SPLIT_REL path (G0.0/G0.2 data-integrity fix)
- `dcdcb39`, `b29298e`, `a20ac57` V28 implementation + dtype fix + results
- `f852f5a`, `b1b6a4b` V26 implementation + results
- `55358ca` V02+V05 regime-free control + tests
- `3179f12` fix(wave3): correct phantom turnover comparator + clean V11 reference
- `4c83f3b`, `0f04be9` V33-core implementation + results
- `6ce1d1b` + V31 clean re-run + family gate report

## Known Issues / Remaining Work

- **NEXT PHASE (not started):** V28 (and V31) purged/embargoed walk-forward -- methodology
  Section 3, >= 3 rolling windows, purge 21d / embargo 2%, OOS/IS >= 0.70, every OOS
  sub-window must beat V11. PBO 0.503 makes this mandatory before graduation.
- Check V28/V31 correlation; if > 0.85 pick one via walk-forward OOS Sharpe.
- V11 remains the deployed paper incumbent until V28 clears the walk-forward.
- The corrupt LEGACY cache (`equities_daily_cache.parquet`) still exists as a silent
  fallback; consider making the loader fail-loud instead of falling through to stale data
  (hardening follow-up, not blocking).

## Validation

- 159 ramp_phase4 unit tests pass after every variant.
- V11 re-run through the new runner reproduced the 0.528 baseline exactly (clean-data + runner sanity check).
- Data fix verified end-to-end: `load_universe_panel` reports "FRESH SIP-aggregated panel",
  NFLX continuous (worst daily ret -3.9% vs the -90% phantom).
- All 6 family streams confirmed full-window (2355 days) with aligned date axes before PBO.
