# Futures SP-B1 Trials -- Pre-Registration Ledger (2026-07-10)

Parameter-free, pre-registered [D]-type session trials (return-stream gated by
the shared PSR/DSR/PBO path). Signs long, fixed from the [A]-tier evidence; no
post-hoc flips. Window boundaries are fixed conventions.

| # | Strategy | Window (ET) | Roots | Expected sign | Trades | Real OOS Sharpe (1x / 1.5x) | PBO (1x / 1.5x) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 21/25 | Overnight drift | 16:00 -> 09:30 | ES,NQ | long | 8086 | 0.792 / 0.671 | 0.513 / 0.457 | WEAK -- positive premium but PBO >> 0.25 (window-unstable, fails the gate) |
| 21 | Overnight NY-Fed hour-slice | 02:00 -> 05:00 | ES,NQ | long | 8086 | -0.023 / -0.277 | 0.873 / 0.827 | REJECT -- negative; drift is NOT concentrated in this window; full overnight is stronger |
| 39 | Pre-FOMC drift | 14:00(F-1) -> 14:00(F) | ES,NQ | long | 252 | GATE n_windows=0 (all-NaN) | n/a | UNGRADEABLE -- sparse ~8 events/yr never fills a 12-month / 10-sample walk-forward window |

Gate = PSR>=0.95 AND DSR>=0.95 AND PBO<0.25 AND 1.5x cost.

## Caveats

- **#21 hour-slice**: the 02:00-05:00 ET window is an unverified APPROXIMATION
  of the NY-Fed SR-917 window, not a definitive refutation of SR-917. The
  negative result rejects this specific approximated window, not the SR-917
  finding itself.
- **#39 pre-FOMC**: PBO NaN / gate n_windows=0 on this single-config, sparse
  return stream is an inherited limitation (as with VIX) -- the return-stream
  walk-forward gate architecturally cannot produce an OOS verdict for a
  ~8-events/yr stream against a 12-month / 10-sample window requirement. This
  is not a code bug.
- **#39 decay subperiod split** (diagnostic only, NOT gate-validated): pre-2015
  Sharpe 0.25 (n=37), post-2015 Sharpe 6.54 (n=89). This is small-n NOISE, not
  a validated finding. Note also that sqrt(252) annualization overstates the
  Sharpe scale for a sparse ~8/yr event stream, so 6.54 is not a real
  annualized Sharpe -- it is an artifact of applying a daily-frequency
  annualization factor to a low-frequency stream.
- A best-of-N deflation across the full campaign (all trials, not just these
  three) is the honest multiple-testing adjustment and has not been applied
  here -- these are single confirmation smokes, not the campaign result.

## Honest bottom line

NONE of the three trials clears the combined statistical gate (PSR>=0.95,
DSR>=0.95, PBO<0.25, 1.5x cost). Overnight drift is the closest candidate --
positive OOS Sharpe at both cost levels -- but its PBO (0.51 / 0.46) is far
above the 0.25 threshold, meaning the edge is not stable across resampled
windows. This mirrors the SP-A/SP-E finding that most signals do not clear the
gate. The engine itself works correctly (return-stream construction is
DST-correct and weekend-overnight-correct); the strategies tested here are
honest negative/marginal results, not engine defects.
