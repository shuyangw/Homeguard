# FX Emerging-Market Wave -- Verdict

**Date:** 2026-07-21/22 | **Pre-registration:** `docs/strategies/research/20260721_fx_em_wave_preregistration.md`
(LOCKED, no post-hoc edits) | **Runner:** `scripts/backtest_scripts/run_fx_wave2_gate.py`
(wraps `walk_forward_fx`, the honest growing-N gate mechanism already used and audited for
FxTSMOM/FxXSectMom/FxCarry/FxGoldSilver and the 2026-07-19 Wave 2 gate)

## Campaign verdict: ALL 7 PRE-REGISTERED TRIALS FAIL

Per the pre-registration's stopping rule (Section 6): **the EM carry/trend/mean-reversion catalog
extension is declared exhausted. STOP -- no wave-2 EM, no ML.** "EM carry/trend/mean-reversion
dies after realistic EM transaction costs and crash risk" is the completed finding, recorded here
as a success per the North Star (surfacing a robust failure is a completed objective, not a
problem to engineer around).

## Trial-by-trial results

Universe: EM7 = USDMXN, USDZAR, USDPLN, USDHUF, USDCNH, USDTRY, USDINR. Data frequency: daily spot
FX. Backtest window: 2011-01-01 to 2026-04-30 (OOS window 2014-01-01 to 2026-04-30 after the
36-month walk-forward train warm-up; CNH enters the panel from 2014). Walk-forward: anchored
36m-train / 12m-test / 12m-step, 13 OOS windows, n_oos_days=3211 for every trial (identical OOS
calendar across trials -- same universe/dates/rebalance-window scheme). EM-specific cost model
(`src/backtesting/costs/fx.py::_EM_HALF_SPREAD_BPS`): MXN 3, ZAR 6, PLN 4, HUF 5, CNH 5, TRY 15,
INR 8 bps half-spread; 1.5x-cost leg applies these x1.5.

| # | Trial | Strategy class | Rebal. | OOS Sharpe 1x | OOS Sharpe 1.5x | PSR | DSR | PBO | N (cumulative) | S&P corr | CAGR | MaxDD | MaxDD dur (d) | Calmar | Monthly win% | Fills (OOS) | Verdict (pre-reg Sec.5) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | EM-CARRY-weekly | `FxCarry` | weekly | 0.0245 | -0.0774 | 0.9164 | 0.0000 | 0.1357 | 112->113 | 0.4203 | -1.21% | -65.6% | 3065 | -0.018 | 57.4% | 4,259 | **FAIL** |
| 2 | EM-CARRY-daily | `FxCarry` | daily | 0.0586 | -0.0988 | 0.9995 | 0.0000 | 0.1012 | 114->115 | 0.4184 | -1.36% | -66.5% | 3065 | -0.020 | 57.4% | 21,282 | **FAIL** |
| 3 | EM-CARRY-SEATBELT | `FxCarrySeatbelt` | weekly | 0.0775 | -0.0025 | 1.0000 | 0.0000 | 0.5633 | 115->116 | -0.0338 | 0.09% | -2.5% | 1185 | 0.035 | 11.5% | 106 | **FAIL** |
| 4 | EM-TSMOM | `FxTSMOM` | weekly | -0.3078 | -0.5169 | 0.0000 | 0.0000 | 0.5171 | 116->117 | -0.1980 | -3.53% | -49.9% | 2005 | -0.071 | 43.2% | 3,277 | **FAIL** |
| 5 | EM-XSMOM | `FxXSectMom` | weekly | -1.1156 | -1.4797 | 0.0000 | 0.0000 | 0.5444 | 117->118 | 0.0151 | -6.86% | -61.6% | 2964 | -0.111 | 33.8% | 4,012 | **FAIL** |
| 6 | EM-CARRY-MOM | `FxCarryMom` | weekly | 0.0420 | -0.0970 | 0.9908 | 0.0000 | 0.5519 | 118->119 | 0.3502 | -0.06% | -37.5% | 3108 | -0.002 | 50.7% | 4,259 | **FAIL** |
| 7 | EM-MEANREV | `FxMeanRev` | weekly | -0.6908 | -1.0134 | 0.0000 | 0.0000 | 0.4793 | 119->120 | 0.1208 | -5.44% | -61.7% | 3164 | -0.088 | 45.3% | 4,048 | **FAIL** |

N column reads "before trial -> after trial" (this trial's own registered run is appended to the
registry immediately after its DSR is computed, growing N for the next trial per
`get_campaign_trial_distribution()`). See "Cumulative-N / DSR accounting" below for the one
apparatus-bug row folded into this chain.

### Per-trial FAIL reasons (pre-registration Section 5 criteria)

- **EM-CARRY-weekly**: OOS Sharpe positive (0.0245) but sign-flips negative at 1.5x cost
  (-0.0774) -- fails the mandatory 1.5x cost-sensitivity leg. PSR 0.916 < 0.95. DSR ~0.
- **EM-CARRY-daily**: same failure mode as weekly -- positive at 1x (0.0586), negative at 1.5x
  (-0.0988). PSR 0.9995 (passes) but DSR ~0 and the cost-sensitivity leg fails outright. Daily
  rebalance roughly doubled realized turnover (21,282 vs 4,259 fills) without buying enough extra
  Sharpe to survive the wider EM spreads at 1.5x.
- **EM-CARRY-SEATBELT**: OOS Sharpe 0.0775, PSR 1.0000 look strong in isolation, but this is an
  artifact of near-zero trading activity (106 fills over 13 years OOS, monthly win rate 11.5% --
  the book sits flat/in-cash most months). PBO 0.5633 (>0.5, fails outright) and 1.5x-cost Sharpe
  crosses to -0.0025. See "Seatbelt crash-filter did not generalize to EM7" below for why: this
  variant is genuinely NOT running the pre-registered crash-filter mechanism.
- **EM-TSMOM**: Non-positive OOS Sharpe (-0.3078). REJECT per the non-positive-Sharpe short
  circuit -- no edge to deflate.
- **EM-XSMOM**: Non-positive OOS Sharpe (-1.1156), the worst of the wave. REJECT.
- **EM-CARRY-MOM**: OOS Sharpe barely positive (0.0420), negative at 1.5x cost (-0.0970), PBO
  0.5519 (>0.5). The carry+momentum blend does not rescue either standalone leg (#1's carry FAIL
  or a hypothetical EM TSMOM leg -- #4's TSMOM FAILs even harder standalone).
- **EM-MEANREV**: Non-positive OOS Sharpe (-0.6908). REJECT. Close-only z-reversion on EM7 loses
  money even before considering costs.

None of the 7 comes close to a "near-miss needing one degree of freedom" -- every trial fails on
at least two independent legs of the combined gate (sign at 1.5x cost, PSR, DSR, or PBO), several
on three or more. This is a clean, unambiguous FAIL set, not a borderline call.

## Harness bug found and fixed (apparatus correction, not a new trial)

`scripts/backtest_scripts/run_fx_wave2_gate.py::run_gate()` read the config's `rebalance` field
into `kw["rebalance"]` but never threaded it into the PRIMARY gated `walk_forward_fx(...)` call --
only into the separate, non-gating S&P book-context helper (`_dated_oos_series`). Every prior use
of this script (Wave 2, 2026-07-19) used `rebalance: weekly` for all trials, so the bug was latent
and silent until this wave's EM-CARRY-daily trial (the first `rebalance: daily` trial run through
this script). The first attempt at EM-CARRY-daily silently re-ran the weekly gate (identical
metrics to EM-CARRY-weekly: oos_sharpe=0.0245, psr=0.9164, pbo=0.1357 -- byte-for-byte the same as
trial 1). Fixed with a one-line change (`rebalance=kw.get("rebalance", "weekly")` added to the
`walk_forward_fx(...)` call) and EM-CARRY-daily was re-run correctly (oos_sharpe=0.0586,
confirmed materially different from weekly, as expected for ~5x higher turnover). This is a bug
fix + re-run of the SAME pre-registered spec, i.e. an apparatus correction per the Phase 6.5
guardrail exception -- it does not consume the 15-iteration specification budget and is not a new
hypothesis.

The buggy first attempt DID already write a registry row (`walk_forward_fx` always calls
`append_run`; run_id `3f2e6d66-9c9d-4abd-b7e2-3f6e5fc21cb1`, oos_sharpe 0.0245, a duplicate of
trial 1's weekly result). Per the North Star ("never shrink N to make a gate easier"), this row
is retained in the registry and therefore in the DSR trial-Sharpe distribution used by every
subsequent trial and by future work -- it is a genuine (if redundant) evaluated backtest, not a
search trial to be swept under the rug. It is excluded from the 7-trial pre-registered count above
(it is not one of the 7 specs) but IS folded into the N=112->120 cumulative-N chain below, which
is the conservative (N-inflating) direction.

## EM-CARRY-SEATBELT crash-filter did not generalize to EM7 (integrity finding)

The pre-registration (Section 3.2) describes EM-CARRY-SEATBELT as "Spec 1 + crash filter... reusing
`src/backtesting/signals/carry_unwind.py` generalized to EM." Verified empirically before
accepting the trial's construction: `compute_unwind_score()` is built from four terms -- JPY
strength delta, CHF strength delta, AUDJPY short-horizon vol, and XAUUSD 3-day return -- none of
which are present in the EM7 universe (USDMXN/ZAR/PLN/HUF/CNH/TRY/INR has no JPY, CHF, AUDJPY, or
XAUUSD leg). Running `compute_unwind_score` on the EM7 close panel over the full 2011-2026 history
confirms the score is **identically 0.0 at every date** (`score.min() == score.max() == 0.0`,
3,993/3,993 observations). Consequently:

- The veto (flatten longs when score >= threshold) never engages.
- The offensive short leg (hardcoded to `AUDJPY`/`NZDJPY` in `fx_carry_seatbelt.py`) never fires
  -- neither pair is in the EM7 universe, so the `if pair in out.columns` guard silently no-ops.

This build gap was NOT listed in the pre-registration's Section 7 prerequisite build tasks (which
covered rate wiring, cost model, and universe wiring, but not generalizing `carry_unwind.py`'s
funding-currency terms to EM analogues). The strategy constructs and runs without error (per the
orchestration brief's instruction, a construction error would have stopped the wave; this is not
an error, it is a silent no-op), so the trial was run as registered under its exact name and is
reported honestly: **EM-CARRY-SEATBELT as actually tested is "carry + momentum-agreement long-only
gate, with a non-functioning crash veto,"** not the full pre-registered seatbelt mechanism. This
does not change the wave's verdict (the trial fails the gate regardless, on PBO and 1.5x-cost
grounds), but the pre-registration's economic rationale for this spec ("EM carry's dominant risk
is the crash... reduce/flatten a leg when its carry-unwind score fires") was never actually tested
against EM crash risk. If EM carry is ever revisited, a genuine EM crash-filter would need EM-native
funding/risk-off proxies (e.g. EM currency basket vol, DXY spike, or a CDX-EM-style credit proxy)
rather than the G10 JPY/CHF/AUDJPY/XAUUSD terms reused unchanged.

## INR tradeability caveat

INR is included in all 7 trials as a signal leg per the pre-registration (non-deliverable, spot
clean, retail-untradeable). None of the 7 trials PASS, so the "any INR-dependent PASS must be
re-checked NDF-tradeable" caveat from the pre-registration does not activate for this wave -- noted
for completeness only.

## Fills-level trade log verification (Section 12 / strategy-pipeline mandate)

Every trial's run-scoped `FillSink` output was verified present and non-empty before accepting its
verdict:

| Trial | `runs/<run_id>/manifest.csv` | `trades_oos.csv.gz` | Fill rows (OOS) |
|---|---|---:|---:|
| EM-CARRY-weekly | `output/backtests/FxCarry/runs/20260722T033525Z_832242/manifest.csv` | present | 4,259 |
| EM-CARRY-daily (corrected) | `output/backtests/FxCarry/runs/20260722T033657Z_796e67/manifest.csv` | present | 21,282 |
| EM-CARRY-SEATBELT | `output/backtests/FxCarrySeatbelt/runs/20260722T033722Z_b1d813/manifest.csv` | present | 106 |
| EM-TSMOM | `output/backtests/FxTSMOM/runs/20260722T033739Z_438fb4/manifest.csv` | present | 3,277 |
| EM-XSMOM | `output/backtests/FxXSectMom/runs/20260722T033757Z_856e47/manifest.csv` | present | 4,012 |
| EM-CARRY-MOM | `output/backtests/FxCarryMom/runs/20260722T033813Z_218c66/manifest.csv` | present | 4,259 |
| EM-MEANREV | `output/backtests/FxMeanRev/runs/20260722T033830Z_c122b0/manifest.csv` | present | 4,048 |

All well above the 30-trade OOS minimum (methodology Section 2.5). Fills are per-rebalance
position-change deltas (date, pair, units, cost), not matched round-trip entry/exit records, so
profit factor and average hold time (which require FIFO trade matching) were not computed for this
wave -- all 7 trials fail the primary gate decisively on independently-sufficient grounds (sign at
1.5x cost, PSR, DSR, and/or PBO), so this additional trade-matching effort would not be
decision-relevant. IS/OOS Sharpe ratio and per-regime breakdown were likewise not computed for the
same reason (proportional effort on a decisively-failed signal, per the Phase 6.5 guardrail
"do not spend the full [analysis] budget on a decisively-dead signal").

## Cumulative-N / DSR accounting

Starting cumulative trial count (before this wave, per
`src.backtesting.walkforward_common.get_campaign_trial_distribution()`, which combines the static
40-trial pre-registry baseline with every registry row carrying a numeric `oos_sharpe` metric):
**N = 112**.

Each `walk_forward_fx()` call computes DSR against the N in the registry AT THAT MOMENT, then
appends its own run (growing N for the next call). Running the 7 trials sequentially (never in
parallel across trials, so N grows honestly trial-to-trial) produced this chain:

| Step | N used for this trial's DSR | Registry run_id appended | New N |
|---|---:|---|---:|
| EM-CARRY-weekly | 112 | `e28fdd02-72d5-4962-be2f-bb98b27972ca` | 113 |
| [bug] first EM-CARRY-daily attempt (actually weekly, duplicate) | 113 | `3f2e6d66-9c9d-4abd-b7e2-3f6e5fc21cb1` | 114 |
| EM-CARRY-daily (corrected) | 114 | `42b51017-af67-4264-a84f-d3731cad4880` | 115 |
| EM-CARRY-SEATBELT | 115 | `baac0907-eac1-44a2-b3b2-c12c2aea948c` | 116 |
| EM-TSMOM | 116 | `d715cc83-19d7-4abf-931e-c291a462976b` | 117 |
| EM-XSMOM | 117 | `94e3a3dc-b2b5-4713-9b15-2e55dbca113b` | 118 |
| EM-CARRY-MOM | 118 | `e0008d00-2748-48dc-8cca-5149294ede27` | 119 |
| EM-MEANREV | 119 | `a823e81e-2288-4b1a-989a-4610a69cafcb` | 120 |

**Final cumulative N after this wave: 120** (112 baseline + 7 valid EM trials + 1 apparatus-bug
duplicate row, retained per the never-shrink-N principle). Every trial's reported DSR above used
the N in effect at the time it ran, not the final N -- this is the honest, sequential, growing-N
behavior the harness is designed to enforce (SR_zero rises with every trial anyone spends, so a
hypothetically-late 7th trial faces a strictly higher bar than the 1st, exactly as intended).

## Stopping rule outcome

Per pre-registration Section 6: **all 7 trials FAIL net of costs under honest walk-forward.** The
EM carry/trend/mean-reversion catalog extension is declared exhausted for the EM7 universe under
the pre-registered mechanisms. STOP -- no wave-2 EM, no ML meta-labeling harness build for this
line of inquiry. The finding is recorded as a completed objective: EM carry's higher rate
differentials and EM trend's structurally different dynamics do NOT survive realistic EM
transaction costs (MXN 3bp / ZAR 6bp / PLN 4bp / HUF 5bp / CNH 5bp / TRY 15bp / INR 8bp half-spread,
x1.5 sensitivity) and the crash/political/convertibility risk premium the carry trades are meant to
harvest.
