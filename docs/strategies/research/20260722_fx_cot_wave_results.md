# FX COT / Positioning Wave -- Verdict

**Date:** 2026-07-22 | **Pre-registration:** `docs/strategies/research/20260722_fx_cot_positioning_preregistration.md`
(LOCKED, no post-hoc edits) | **Runner:** `scripts/backtest_scripts/run_fx_wave2_gate.py`
(wraps `walk_forward_fx`, the honest growing-N gate mechanism already used and audited for
FxTSMOM/FxXSectMom/FxCarry/FxGoldSilver, the 2026-07-19 Wave 2 gate, and the 2026-07-21 EM wave)

## Campaign verdict: ALL 3 PRE-REGISTERED TRIALS FAIL

Per the pre-registration's stopping rule (Section 6): **the COT/positioning signal family is
unproductive for this daily-spot-taker construction. STOP this wave -- no further COT specs, no
parameter sweep, no ML variant.** Per the CLAUDE.md North Star and the SCOPE banner already
governing this tracker, this is a SCOPED finding: it does NOT establish that CFTC positioning data
carries no predictive value in FX -- only that a weekly-net%OI-z-score signal, publication-lagged
D+7, feeding a daily-spot-taker book on this 8-pair universe with these fixed forms and
parameters, does not clear the pre-registered gate.

## Trial-by-trial results

Universe: COT8 = EURUSD, USDJPY, GBPUSD, USDCAD, USDCHF, AUDUSD, NZDUSD, USDMXN. Data frequency:
daily spot FX with weekly COT signal (D+7 publication-lagged) forward-filled to daily. Backtest
window: 2011-01-01 to 2026-04-30. Walk-forward: anchored 36m-train / 12m-test / 12m-step, 13 OOS
windows, n_oos_days=3200 for every trial (identical OOS calendar across trials -- same
universe/dates/rebalance scheme). Cost model: standard taker (major tier + USDMXN EM bps,
`src/backtesting/costs/fx.py`); 1.5x-cost leg is the mandatory sensitivity gate. Rebalance: weekly
(matches COT publication frequency). IDM on, vol_target 0.03/instrument, leverage cap 10.

| # | Trial | Strategy class | Form / sign | OOS Sharpe 1x | OOS Sharpe 1.5x | PSR | DSR | PBO | N (cumulative, before -> after) | S&P corr | IR vs S&P | Verdict (pre-reg Sec.5) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | COT-CONTRARIAN-TS | `FxCotContrarianTS` | per-pair TS, sign fixed negative (fade crowded level) | -0.1292 | -0.1648 | 0.0000 | 0.0000 | 0.4732 | 120 -> 121 | 0.0286 | -0.5680 | **FAIL** |
| 2 | COT-MOMENTUM-TS | `FxCotMomentumTS` | per-pair TS, sign fixed positive (follow positioning flow) | -0.1040 | -0.1965 | 0.0000 | 0.0000 | 0.4771 | 121 -> 122 | 0.0064 | -0.6239 | **FAIL** |
| 3 | COT-CONTRARIAN-XS | `FxCotContrarianXS` | cross-sectional, sign fixed negative (fade most-crowded pair) | -0.1278 | -0.1601 | 0.0000 | 0.0000 | 0.2374 | 122 -> 123 | -0.1293 | -0.5332 | **FAIL** |

N column reads "before trial -> after trial" (this trial's own registered run is appended to the
registry immediately after its DSR is computed via `walk_forward_fx`, growing N for the next
trial). Baseline N=120 confirmed via `get_campaign_trial_distribution()` immediately before this
wave started, matching pre-registration Section 8's stated prior cumulative N. Final cumulative N
after the wave: **123**.

### Per-trial FAIL reasons (pre-registration Section 5 criteria)

- **COT-CONTRARIAN-TS**: Non-positive OOS Sharpe (-0.1292), and the edge WIDENS negative at 1.5x
  cost (-0.1648) -- a genuine cost-sensitivity failure, not a marginal edge nudged under by
  friction. REJECT per the non-positive-Sharpe short circuit; PSR/DSR are consequently 0 (no edge
  to deflate). PBO 0.4732 is elevated, consistent with an unstable/absent signal.
- **COT-MOMENTUM-TS**: Non-positive OOS Sharpe (-0.1040), widening to -0.1965 at 1.5x cost -- the
  largest cost-sensitivity degradation of the three (89% worse). REJECT. PBO 0.4771.
- **COT-CONTRARIAN-XS**: Non-positive OOS Sharpe (-0.1278), widening to -0.1601 at 1.5x cost.
  REJECT. PBO 0.2374 is the lowest of the three (below the live-deployment 0.25 bar) but this is
  moot -- PBO only matters once the Sharpe/cost legs clear, which they do not.

All three fail on the FIRST gate clause alone (non-positive OOS Sharpe), which the pre-registration
explicitly lists as an unconditional FAIL condition. No near-miss occurred and no post-hoc degree
of freedom (universe trim, window reselection, cost-model change, parameter retune) was invoked or
needed to reach any of these three verdicts -- all are decisively negative on the primary metric,
and all three degrade further (not converge toward zero) under the cost-sensitivity stress test,
which is the signature of a real (if small) friction drag rather than a borderline result.

### Both pre-registered mechanisms fail, in both directions and both constructions

The wave tested the two documented, OPPOSITE-signed COT mechanisms (contrarian-levels-mean-revert
vs. momentum-flow-trends) plus two distinct constructions of the contrarian mechanism
(per-pair time-series vs. cross-sectional rank). None produced a positive OOS edge. This rules out
"wrong sign" as the failure mode (both signs were tested, pre-registered, and both failed) and
rules out "wrong construction" for the contrarian leg specifically (both TS and XS forms failed).
The uniformly small-magnitude negative Sharpes (-0.10 to -0.13, all in a narrow band) combined with
near-zero S&P correlation (0.006 to 0.029 in magnitude for two of three; -0.13 for XS) suggest the
signal itself carries close to zero information content in this construction, rather than a real
edge overwhelmed by a large directional bet gone wrong -- consistent with a genuinely dead signal,
not an unlucky sample.

### Cumulative-N / DSR accounting

No apparatus bugs or re-runs occurred in this wave -- each of the 3 trials ran exactly once through
`walk_forward_fx`, appending exactly one registry row and consuming exactly one trial slot, matching
the pre-registration's Section 8 trial-accounting commitment (baseline 120 + 3 = 123, no
undercounting). The elevated PBO values (0.24-0.48) and DSR~0 across all three trials mean the
7-8% marginal increase in cumulative N this wave contributes had no bearing on the verdict -- these
specs are decisively rejected on Sharpe sign alone, well before deflation enters the calculation.

## Fills-level trade log verification (Section 12 / strategy-pipeline.md)

Every trial's walk-forward run persisted its fills via the mandatory run-scoped `FillSink`,
verified non-empty (manifest.csv + trades_oos.csv.gz) before accepting each verdict:

| Strategy | Run dir | trades_oos.csv.gz (uncompressed size) |
|---|---|---:|
| FxCotContrarianTS | `output/backtests/FxCotContrarianTS/runs/20260722T044732Z_b210e2/` | 272,688 bytes |
| FxCotMomentumTS | `output/backtests/FxCotMomentumTS/runs/20260722T044751Z_8924cd/` | 272,622 bytes |
| FxCotContrarianXS | `output/backtests/FxCotContrarianXS/runs/20260722T044808Z_5fc5b3/` | 272,892 bytes |

## Lookahead / integrity check

`src/data/cot.py::load_cot_weekly_panel` applies a fixed D+7 calendar-day publication lag to the
COT Tuesday snapshot (a conservative buffer past the actual D+3 Friday publication) and marks the
result "active" only from that date; `to_daily()` forward-fills each daily bar from the most recent
active-dated weekly value, so no bar can see a COT reading before it was public. The rolling
z-scores (`FxCotPositioningStrategy._weekly_forecast`, 156-week window) are computed entirely on
this already-lagged weekly series -- causal by construction, no additional shift needed. This
machinery is unchanged, previously-audited infra reused from the pre-registration's prerequisite
build phase; no changes were made during this verdict run.

## Stopping outcome

All 3 FAIL -> per pre-registration Section 6, this wave STOPS here. The COT/positioning signal
family, as specified (weekly net%OI level/flow z-scores, D+7 lag, daily-spot-taker execution,
COT8 universe), joins the retail daily/session taker-factor slice already declared exhausted by the
2026-07-19 Wave 2 resolution and the 2026-07-21 EM wave resolution. Per the SCOPE banner, order-flow
and options-implied signal families, and any maker/liquidity-provision or microstructure-frequency
construction of positioning data, remain untested and open.
