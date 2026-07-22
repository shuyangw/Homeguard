# FX EM Daily Cache + Dukascopy Backfill - 2026-07-21

## Summary
Built an emerging-market FX daily cache (8 USD-EM pairs) from the on-disk Massive
minute tree using the existing symbol-generic `build_fx_daily_cache`, then brought
6 of them to G10-grade gap-free quality by backfilling two shared single-vendor
holes from Dukascopy (the same keyless procedure used for G10). This unblocks the
EM strategies (#18 EM carry, #55 USDCNH) on the data side. No strategy verdict was
run; this is data plumbing only.

## Context / decision
- User asked to test another wave; investigation showed the tracker's `DATA` tag
  was stale. EM spot, oil, equity indices, and the CB calendar are all already
  local or one keyless fetch away. The only real blockers are build steps.
- EM minute data (USDMXN/ZAR/CNH/TRY/BRL/PLN/HUF/INR, 2011-2026) was already on
  disk under `fx/massive/1min/` (symlinked as `fx_1min/`). Building the daily
  cache is a re-run of the existing parameterized builder, not new infra.
- User chose "backfill first" (gate-grade data before any EM strategy screen).

## Changes Made
- **src/data/acquisition/plugins/dukascopy_fx.py**: added 6 EM entries to
  `DUKA_MAP` (USDMXN/ZAR/CNH/TRY/PLN/HUF). Hardened `to_canonical`: empty input ->
  empty canonical frame; object-dtype volume coerced to numeric. Restricted EM
  instruments (BRL/INR) return empty frames for months Dukascopy lacks, which
  previously crashed `volume.round().astype(int)`.
- **tests/data/test_dukascopy_fx.py**: 2 new regression tests (empty-frame,
  object-volume). 7/7 pass.
- **config/universes/fx_em-2026.csv**: 8-pair EM universe list (new).
- **EM daily bars** written to `fx_daily/symbol=USD{MXN,ZAR,CNH,TRY,BRL,PLN,HUF,INR}/`
  (local, gitignored market data).

## Method / validation
- Cross-validated each EM Dukascopy mapping vs existing Massive data on a control
  month (2022-06): median |diff| 1.0-9.8 bps for all 6 -> mappings confirmed
  correct (driver aborts if any exceeds 50 bps).
- Detected two shared vendor holes programmatically (runs >= 5 missing business
  days): Q4-2020 (~2.5 months, extends into 2021-01) and Sep-2019 (~3 weeks);
  USDCNH also 2014-09..11; USDZAR also 2025-01. 38 pair-months backfilled
  (1,096,978 minute bars).
- Post-backfill quality (vs G10 baseline 99.2-99.6% coverage):
  - USDMXN 99.5%, USDZAR 99.3%, USDCNH 99.7%, USDTRY 99.6%, USDPLN 99.9%,
    USDHUF 100.0% -- all 0 significant gaps (maxgap 1-3 bdays). G10-grade.
  - USDBRL 99.2% and USDINR 99.0% retain smaller Massive-only holes (~23-bday
    max, 2 gaps each): Dukascopy carries neither (0 rows any period). Documented.
  - Spike-cleaner preserves real EM crisis moves (USDTRY maxret 19.9% = 2018/2021
    lira crises, USDZAR 12.8% = real selloffs); reverting artifacts nulled
    (ZAR 13, PLN 1).

## Commits
- `1ebf47e` feat(fx-data): EM daily cache via Dukascopy backfill (6 G10-grade pairs)

## Known Issues / Remaining Work
- USDBRL / USDINR are not gate-grade (Massive-only holes; no Dukascopy source).
  Fine to exclude from an EM wave, or source elsewhere if BRL/INR is needed.
- EM carry (#18) still needs an EM short-rate availability check in FRED: MXN is
  ready per the tracker; ZAR/TRY/PLN/HUF/CNH rate series unverified (some EM
  series may not exist keyless). Verify before running an EM carry basket.
- No strategy has been run on EM yet. Next step (deferred, awaits go-ahead):
  pre-register an EM screen (carry #18 lead + G10 mechanisms re-run on EM
  universe), route the verdict through strategy-lead.

## Validation
- tests/data/test_dukascopy_fx.py 7/7 pass (fintech env).
- py_compile clean on the plugin.
- Final gap-check report confirms 6 EM pairs at G10-grade coverage, 0 significant
  gaps. Merged via FF ref-update (no checkout); origin/main = 1ebf47e.
