# FX EM Extension: Data -> Pre-Registration -> Build -> Verdict (orchestrator log) - 2026-07-21

## Summary
Extended the FX catalog to emerging markets end-to-end in one session: built + independently
validated an EM daily spot cache, pre-registered a 7-trial EM wave, built the prerequisites
(rate wiring, EM cost model, 2 strategies, rebalance threading), and ran the verdict through
`strategy-lead`. **All 7 EM trials FAIL the combined gate; EM catalog extension declared
exhausted per the pre-registered stopping rule.** This is a completed finding, not a dead end
to engineer around.

## Arc + key decisions (orchestrator level)
1. **Investigation** found the tracker's `DATA` tag was stale: EM spot minute data was already
   on disk; oil/equity/calendar are local or one keyless fetch away. The real blockers were
   build steps, not sourcing.
2. **Decision: backfill-first** (user). Built the EM daily cache via the existing symbol-generic
   `build_fx_daily_cache` (a re-run, not new infra), then closed two shared vendor holes
   (Q4-2020, Sep-2019) from Dukascopy for 6 pairs. BRL/INR not on Dukascopy -> kept as-is.
3. **Decision: validate independently (ultrathink).** Checked the CLEANED series vs yfinance +
   FRED H.10 (indep. of both feeds). Caught + fixed 25 sprinkled bad USDZAR closes (2023-2025)
   the spike-cleaner missed; structural + one-month checks had NOT caught them. Lesson recorded.
4. **Pre-registration LOCKED before any run** (North Star): 7 pairs, 6 specs / 7 trials, gate +
   EM cost model + 1.5x sensitivity + S&P bar, pre-committed PASS/FAIL + stopping rule. User
   chose universe (Tier1+TRY+INR) and wave size (6 specs) via explicit decision.
5. **Build prerequisites** (subagent-driven pattern, direct + tested): wired 5 EM FRED rate
   series; added an EM per-pair bps cost model that FIXED a p-hacking trap (`_tier_for_pair`
   priced every USD pair as "major", under-costing EM 3-14x); added FxCarryMom + FxMeanRev;
   threaded rebalance cadence through the WF runner (to honor the carry-daily trial).
6. **Verdict delegated to `strategy-lead`** (build-vs-verdict boundary + hook sentinel). It ran
   all 7 trials, gated, and produced the results docs.

## Verdict (all 7 FAIL; cumulative DSR N=120)
| Trial | Sharpe 1x | Sharpe 1.5x | PSR | DSR | PBO | Verdict |
|---|---:|---:|---:|---:|---:|---|
| EM-CARRY-weekly | +0.024 | -0.077 | 0.916 | 0 | 0.136 | FAIL |
| EM-CARRY-daily | +0.059 | -0.099 | 1.000 | 0 | 0.101 | FAIL |
| EM-CARRY-SEATBELT | +0.078 | -0.003 | 1.000 | 0 | 0.563 | FAIL (see caveat) |
| EM-TSMOM | -0.308 | -0.517 | 0 | 0 | 0.517 | FAIL |
| EM-XSMOM | -1.116 | -1.480 | 0 | 0 | 0.544 | FAIL |
| EM-CARRY-MOM | +0.042 | -0.097 | 0.991 | 0 | 0.552 | FAIL |
| EM-MEANREV | -0.691 | -1.013 | 0 | 0 | 0.479 | FAIL |

Carry legs are the pre-registered story: thin positive gross edge, sign-flips negative at 1.5x
EM cost, DSR=0 deflated for N=120. Trend/reversion negative even at 1x. EM's larger carry
differentials and different dynamics do NOT survive realistic EM costs or the deflation.

## Integrity findings surfaced by strategy-lead (not hidden)
1. **Gate-runner rebalance bug**: `run_fx_wave2_gate.py`'s primary gated call didn't thread the
   config `rebalance`, so a first EM-CARRY-daily attempt silently re-ran weekly. Fixed + re-run;
   the buggy duplicate registry row is RETAINED (N never shrinks) and documented.
2. **EM-CARRY-SEATBELT ran with an INERT crash filter** (CAVEAT): `compute_unwind_score`'s
   JPY/CHF/AUDJPY/XAUUSD terms are all absent from EM7, so the score is identically 0 across
   2011-2026. The seatbelt trial ran as a degenerate long-only carry+momentum book with a
   non-functioning veto -- it FAILED, but the pre-registered CRASH-FILTER mechanism was not
   actually tested on EM. Open item: generalize the unwind score to EM would be needed to test
   it genuinely (low prior probability of changing the verdict given the cost/DSR reality).

## Commits (all on main, pushed; origin/main = 0f6e855)
- `1ebf47e` EM daily cache via Dukascopy backfill (6 G10-grade pairs)
- `dd74728` EM cache session log + tracker unblock
- `5c5a33c` EM cache independent validation + ZAR artifact fix
- `6a3b32c` tracked EM wave pre-registration (locked)
- `346a2f8` EM wave build (rates, EM cost model, carry-mom + mean-rev, tests)
- `121b058` thread rebalance cadence through fx walk-forward runner
- `0f6e855` gate EM7 wave -- all 7 trials FAIL, catalog exhausted (strategy-lead)

## Deliverables
- Data: `docs/progress/20260721_fx_em_cache_backfill.md`
- Pre-reg: `docs/strategies/research/20260721_fx_em_wave_preregistration.md`
- Results: `docs/strategies/research/20260721_fx_em_wave_results.md`, `docs/reports/fx/em_wave_gate.md`
- Verdict session log: `docs/progress/20260721_fx_em_wave_gate.md`
- Tracker: `docs/strategies/FX_60_CATALOG_TRACKER.md` (EM WAVE RESOLUTION)

## Known issues / remaining work
- EM-CARRY-SEATBELT caveat above: crash-filter mechanism genuinely untested on EM (decide
  whether to generalize `compute_unwind_score` + re-run trial 3, or accept the caveat).
- USDBRL not gate-grade (holiday thin prints, not Dukascopy-backfillable); excluded from the wave.
- With EM exhausted alongside the G10 wave-2 exhaustion, the remaining untested FX mechanisms are
  the OHLC-daily group (#1/6/8/12/27/28/29/47, needs the trivial OHLC-into-forecast_panel wiring)
  and the intraday/bracket/ML/spread groups (larger builds).

## Validation
- EM data validated vs 2 independent sources (yfinance + FRED); ZAR fixed.
- Build: 14 new tests + 109 existing FX/cost/registry tests green.
- Verdict: strategy-lead verified each trial's fills artifact (manifest.csv + non-empty
  trades_oos.csv.gz) before accepting; sentinel removed; working tree clean.
