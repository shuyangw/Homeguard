# #20 London Open Breakout Pre-Registration - 2026-07-19

Written and committed BEFORE any London Breakout walk-forward was run.

## Strategy
#20 London Open Breakout (Asian range break), intraday, pairs GBPUSD/EURUSD/
EURGBP/GBPJPY. Spec: docs/superpowers/specs/2026-07-19-fx-intraday-london-breakout-design.md.

## Success criterion (primary, relative)
Aggregate per-pair intraday P&L (a qty/pip-independent R-multiple series) into a
combined equal-risk daily return series; run the existing FX walk-forward
(36m/12m/12m); PASS if stitched OOS Sharpe exceeds the S&P 500 Sharpe over the
same OOS dates.

## Diagnostics (non-gating)
PSR, DSR (project-wide trial count), PBO, IS/OOS Sharpe ratio, correlation and IR
vs S&P, S&P aligned day count.

## No absolute kill threshold
A form that fails the S&P bar is a failed base form; one bounded improvement
round (a #20 modification a-d) may follow only if it lands marginal.

## Known limitations
Conservative 1m fills (worst-of trigger/open, adverse both-in-one-bar); half-
spread slippage is a floor; approximate tier-1 event dates (2a). The intraday
fills embed a single fixed round-trip spread (major tier); a separate 1.5x cost
leg is not re-run because the R-multiple nets that spread per trade and cannot be
cleanly rescaled post-hoc without a strategy-level cost multiplier -- a single
cost leg is reported honestly per this note rather than fabricating a 1.5x leg.
