# FxCarrySeatbelt Pre-Registration - 2026-07-06

Written and committed BEFORE any FxCarrySeatbelt backtest was run. Records the
success criterion so it cannot be moved after seeing results.

## Strategy
FxCarrySeatbelt (research #16 Carry-Momentum Double Filter + #19 Carry-Unwind
Detector). Spec: docs/superpowers/specs/2026-07-06-fx-carry-seatbelt-design.md.

## Success criterion (primary, relative)
Run the existing FX walk-forward (36m train / 12m test / 12m step, purge +
embargo, both 1.0x and 1.5x cost legs) on BOTH the daily and weekly rebalance
configs. The strategy PASSES if its stitched OOS Sharpe (1.0x cost) exceeds the
S&P 500 buy-and-hold annualized Sharpe computed over the exact same stitched OOS
dates (rf = 0, same convention), on at least one cadence.

## Diagnostics (reported, NOT gating)
PSR, DSR (using the cumulative project-wide trial count), PBO, trade count,
IS/OOS Sharpe ratio, OOS Sharpe under 1.5x cost, correlation to the S&P over the
OOS dates, information ratio vs the S&P. Plus per-episode P&L attribution for the
Aug 2024 yen-carry unwind and the Mar 2020 COVID unwind, reported as existence
proofs (N is too small to be statistics).

## No absolute kill threshold
There is no pre-committed DSR/Sharpe floor that abandons the carry idea. A form
that fails the S&P bar is a failed variant; whether to iterate (the one deferred
variant: #16 mod-a 12-month TSMOM momentum leg or mod-b graded sizing) or shelve
is decided after seeing the result and the diagnostics.

## Known limitations accepted going in
1. Swap = FRED policy-rate differential proxy (no broker swap tables); an
   optimism bias in the carry gate, reported not hidden.
2. Offensive short rests on ~4-6 unwind events; existence proof, not statistics.
