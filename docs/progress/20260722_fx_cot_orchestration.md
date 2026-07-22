# FX COT Positioning Wave -- Orchestrator Log - 2026-07-22

## Summary
First test of a NON-price signal family against the daily-spot-taker engine, prompted
by the new North Star principle ("a negative bounds the specification tested, not the
asset class"). Fetched CFTC COT, built the loader + 3 strategies, pre-registered, and
ran the verdict via strategy-lead. **All 3 trials FAIL, scoped.**

## Why this wave (decision context)
After G10 + EM exhaustion of the retail daily/taker price/rate/carry slice, the user
challenged "exhausted the asset class" (profitable FX industries exist -> the test is
mis-specified, not the market dead). We encoded that as a principle, then picked the
mis-specification we can test HONESTLY now: SIGNAL FAMILY. The cost-side (maker) and
frequency (microstructure) axes are PARKED -- they need tick/L2 data to model adverse
selection, and a minute-bar version would be a fake PASS. CFTC COT positioning is a
genuinely new signal family, free, and honestly gate-able on the trusted daily engine.

## What was built (all merged + pushed)
- `scripts/data/fetch_cot_fx.py`: keyless CFTC Socrata fetch -> `alt_data/cot/cot_fx.parquet`
  (8 pairs, 2000-2026, signed bullish-the-pair net%OI).
- `src/data/cot.py`: publication-lagged (D+7) weekly panel + no-lookahead daily ffill.
- `fx_strategies.py`: FxCotContrarianTS / MomentumTS / ContrarianXS (signs fixed a
  priori); registered; 3 WF configs. 5/5 unit tests.
- Pre-registration LOCKED before running: `docs/strategies/research/20260722_fx_cot_positioning_preregistration.md`.

## Verdict (all 3 FAIL; cumulative DSR N 120 -> 123)
| Trial | Sharpe 1x | Sharpe 1.5x | PSR | DSR | PBO | S&P corr |
|---|---:|---:|---:|---:|---:|---:|
| COT-CONTRARIAN-TS | -0.129 | -0.165 | 0 | 0 | 0.47 | 0.03 |
| COT-MOMENTUM-TS | -0.104 | -0.197 | 0 | 0 | 0.48 | 0.01 |
| COT-CONTRARIAN-XS | -0.128 | -0.160 | 0 | 0 | 0.24 | -0.13 |

All fail on the primary clause (non-positive OOS Sharpe), and the deficit WIDENS at
1.5x cost (genuine friction drag, not a marginal edge tipped under). No near-miss.

## Verdict scoping (per the principle)
NOT "COT/positioning has no FX predictive value." Only: this weekly-net%OI z-score,
D+7-lagged, daily-spot-taker construction on COT8 does not clear the gate. Recorded
scoped in the tracker's COT WAVE RESOLUTION + SCOPE banner.

## Commits (main = origin = 7b4af2e)
- `8a7b6bb` LOCK COT pre-registration
- `761253a` COT build (loader, 3 strategies, fetch, configs, tests)
- `7da7f2c` COT verdict (strategy-lead) -- all 3 FAIL, scoped
- `7b4af2e` COT verdict session log (strategy-lead)

## Integrity
Fills verified non-empty (4941 OOS rows/trial + manifest). N counted honestly
(120->123, one registry row/trial). No lookahead (D+7 lag + backward rolling z).
Sentinel set/removed by strategy-lead. Signs fixed a priori (no post-hoc flip).

## Next (queued -- user's Tier B pick: more local signal families)
Cross-asset commodity-currency (oil->CAD/NOK via fetch_brent, gold->AUD from metals
cache), COT commercial/hedger positioning, risk-regime overlays (VIX/gold risk-on/off).
All honestly testable on the daily engine + taker costs. Scope the wave next.
