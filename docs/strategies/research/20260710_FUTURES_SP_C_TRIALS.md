# Futures SP-C: Family E Spread Trial Ledger - 2026-07-10

Sub-project C of the Futures Strategy Testability Campaign. Every spread x
segment/pair x weighting x sign is a pre-registered trial; verdicts below are the
REAL walk-forward gate output (train 36m / test 12m / step 12m), recorded verbatim.

**Gate (methodology Section 2.5):** PASS requires PSR >= 0.95 AND DSR >= 0.95 AND
PBO < 0.25. The return-stream path used here bakes realistic per-turnover cost
directly into the stream (it does not compute a separate 1.5x-cost OOS Sharpe
sub-gate); no trial reaches the cost question, since none clears PBO first.
**Benchmark:** carry_idm OOS Sharpe 0.765 (the incumbent best deployable futures book).

**Bottom line: NOTHING in Family E beats carry_idm, and nothing passes the gate.**
Every trial is ungradeable, a reject, or (for #33 crush) PBO-clean but with a
trivial Sharpe. The one nominal "beat" (#31 NG at 1.017) was pure roll-jump
contamination that vanished under masking. Surfacing these failures is the
completed objective.

## Continuous engine ([C] strategies)

| # | Trial | Weighting | Sign (pre-reg) | n_windows | OOS Sharpe | PSR | DSR | PBO | Verdict |
|---|-------|-----------|----------------|-----------|-----------|-----|-----|-----|---------|
| 35 | 2s10s steepener | DV01-neutral | long-10Y/short-2YY (CONFIRMED +0.0247 over 2023-11..2024-09) | 0 | NaN | NaN | NaN | NaN | UNGRADEABLE |
| 35 | 2s5s steepener | DV01-neutral | (same) | 0 | - | - | - | - | UNGRADEABLE (5YY degraded) |
| 35 | 5s30s steepener | DV01-neutral | (same) | 0 | - | - | - | - | UNGRADEABLE (5YY degraded) |
| 36 | NQ/ES RV | beta (notional) | 12-1 momentum | 12 | 0.329 | 1.0 | 1.0 | NaN | FAIL (< carry; PBO NaN shared-gate artifact) |
| 36 | RTY/ES RV | beta (notional) | 12-1 momentum | 4 | -0.280 | ~0 | ~0 | 0.913 | REJECT |

- **#35 ungradeable, not a bug:** CME Micro Yield futures are new (2YY launched
  ~2021-08). The pre-registered 3-year (756-day) policy-cycle z-window consumes
  most of the ~4.4-year history, leaving only ~1.5 years of usable return stream
  (2024-02..2025-08) -- below the 48 months the gate needs to open even one OOS
  window. 5YY has sparse Databento coverage (~440 rows, multi-month gaps from
  2023), so the 2s5s / 5s30s segments produce no gradeable stream at all. Sign was
  confirmed positive over a known bull-steepening episode and NOT flipped.
- **#36 does not beat carry.** NQ/ES OOS Sharpe 0.329 << 0.765; RTY/ES is negative
  and overfit (PBO 0.913). book_corr (correlation to the S&P equity-momentum
  sleeve) NOT RUN -- no readily-loadable RAMP daily-return series was found; it is
  a follow-up (the make-or-break re-expression check for NQ/ES specifically).
- **PBO=NaN is a shared-gate property** (affects VIX #26 too): `_compute_pbo`
  calls `pbo(matrix, s=16)`, which returns NaN when the shortest OOS window has
  < 16 rows; `_oos_windows` keeps windows with >= 10 rows, so one short
  data-end-truncated window NaNs the whole PBO. A methodology follow-up, not an
  SP-C defect.

## Convergence engine ([D] strategies, roll-masked where applicable)

| # | Trial | Anchor | n_windows | OOS Sharpe | PBO | skew | kurtosis | n_trades | Verdict |
|---|-------|--------|-----------|-----------|-----|------|----------|----------|---------|
| 31 | CL calendar | cash-and-carry | 13 | 0.394 | 0.631 | 1.09 | 163 | 84 | REJECT |
| 31 | NG calendar | cash-and-carry | 13 | -0.150 | 0.320 | -14.4 | 598 | 66 | REJECT (provisional*) |
| 31 | ZC calendar | cash-and-carry | 13 | 0.174 | 0.529 | 0.39 | 64 | 33 | REJECT |
| 31 | ZS calendar | cash-and-carry | 13 | 0.358 | 0.429 | 4.68 | 127 | 31 | REJECT |
| 31 | ZW calendar | cash-and-carry | 13 | 0.263 | 0.818 | 2.70 | 69 | 54 | REJECT |
| 32 | crack RB-CL | refining margin | 13 | -0.116 | 0.469 | - | 57 | ~40 | REJECT |
| 32 | crack HO-CL | refining margin | 13 | -0.215 | 0.704 | - | 129 | ~40 | REJECT |
| 33 | crush ZM+ZL-ZS | processing margin | 13 | 0.136 | 0.109 | - | 142 | ~40 | MARGINAL (PBO clean, Sharpe trivial) |
| 34 | GC/SI ratio | none (weak) | 13 | 0.269 | 0.674 | 3.05 | 109 | - | REJECT |

*NG REJECT is PROVISIONAL: PBO 0.320 sits near the 0.25 threshold, and
`front_next_history` selects F1/F2 by daily volume rank (which over-masks on
rank-noise symbol flips, deflating the Sharpe conservatively). A RollCalendar-based
F1/F2 (SP-C2 refinement) could move NG's verdict; the other four calendars reject
comfortably (PBO 0.43-0.82).

### The #31 roll-jump finding (headline)
The naive `(F2-F1).diff()` calendar returns were CONTAMINATED by contract-roll
jumps (a level discontinuity on every roll booked as a spurious return). Pre-fix
Sharpes looked strong -- CL 1.183, NG 1.017 (nominally > carry 0.765), ZW 1.019 --
with tell-tale kurtosis 327-556. Masking roll-day returns (front/second
symbol-change days, mirroring the VIX #26 fix) COLLAPSED them to 0.394 / -0.150 /
0.263. The apparent edge was almost entirely roll-jump artifact.

### Methodology caveats (SP-C2)
- #32/#33 additive spreads are built from independently ratio-adjusted continuous
  legs with a single fixed scale, inflating skew/kurtosis (57-142). Per-contract
  front-series (like #31) or return-space construction would be cleaner. Does not
  change the honest-negative verdicts (crack negative, crush trivial).
- #34 kurtosis 109 is genuine GC/SI fat tails (multiplicative pct_change is
  roll-clean), not an additive artifact.

## Trades / reproducibility
Every trial persisted returns.csv + gate.json (and trades.csv for the convergence
strategies) under `output/backtests/futures/sp_c_{steepener,intermarket,calendar,processing,ratio}/`.
