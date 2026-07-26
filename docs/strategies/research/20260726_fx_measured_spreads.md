# FX Measured Spreads vs the Assumed Cost Model - 2026-07-26

## Summary
The FX cost model had never been measured: it used a per-tier PIP constant plus a
SYNTHETIC hour-of-week shape. Sampling real Dukascopy bid/ask across 25 pairs
(Jun 2015 / 2020 / 2024, ~30k quotes per pair-month, 8,920 rows, zero fetch
failures) shows the model is wrong in BOTH directions, badly for some pairs.

## Measured round-trip cost vs model (2024 price levels, bps of notional)

| pair | px | measured RT | model RT | model/real | |
|---|---:|---:|---:|---:|---|
| EURNOK | 11.67 | 4.32 | 0.21 | **0.05x** | UNDER |
| USDNOK | 10.66 | 3.96 | 0.23 | **0.06x** | UNDER |
| EURSEK | 11.39 | 3.70 | 0.21 | **0.06x** | UNDER |
| USDSEK | 10.48 | 3.43 | 0.23 | **0.07x** | UNDER |
| XAGUSD | 29.02 | 10.41 | 4.00 | **0.38x** | UNDER |
| AUDNZD | 1.09 | 2.15 | 2.20 | 1.03x | ok |
| CHFJPY | 170.98 | 1.20 | 1.40 | 1.17x | ok |
| USDMXN | 18.15 | 4.32 | 6.00 | 1.39x | ok |
| NZDJPY | 90.92 | 1.79 | 2.64 | 1.48x | ok |
| USDPLN | 3.98 | 4.90 | 8.00 | 1.63x | ok |
| EURCHF | 0.95 | 1.53 | 2.53 | 1.65x | ok |
| USDHUF | 362.46 | 5.65 | 10.00 | 1.77x | ok |
| USDCAD | 1.37 | 0.95 | 1.76 | 1.84x | ok |
| NZDUSD | 0.61 | 2.05 | 3.95 | 1.92x | ok |
| USDCHF | 0.88 | 1.41 | 2.72 | 1.93x | ok |
| USDZAR | 18.32 | 5.54 | 12.00 | 2.17x | ok |
| AUDUSD | 0.66 | 1.62 | 3.64 | 2.25x | ok |
| GBPUSD | 1.27 | 0.80 | 1.89 | 2.36x | ok |
| EURJPY | 163.12 | 0.61 | 1.47 | 2.43x | ok |
| XAUUSD | 2380.65 | 1.63 | 4.00 | 2.45x | ok |
| AUDJPY | 98.77 | 0.98 | 2.43 | 2.48x | ok |
| USDJPY | 151.44 | 0.33 | 1.58 | 4.77x | over |
| EURUSD | 1.08 | 0.32 | 2.22 | 6.97x | over |
| USDCNH | 7.22 | 0.73 | 10.00 | 13.78x | over |
| USDTRY | 32.82 | 1.69 | 30.00 | 17.70x | over |

Median measured round trip across the universe: **1.69 bps**.

## Root cause

The tier model charges in PIPS. A pip is a different fraction of price at
different levels: 0.92 bps on EURUSD (px 1.08) but 0.086 bps on EURNOK (px
11.67). So every high-priced cross is systematically under-charged. The EM path
already prices in bps of notional, which is why MXN/ZAR/PLN/HUF land in a sane
1.4-2.2x band.

## Two things I got wrong earlier this session

1. **The EM bps model I added on 2026-07-21** -- presented as closing a
   p-hacking trap -- is itself badly miscalibrated for 2 of its 8 pairs. I
   assumed TRY 15 bps/side (30 RT) against a measured 1.69, and CNH 5 bps/side
   (10 RT) against 0.73. Conservative, so it cannot have manufactured a pass,
   but it was a guess presented alongside measured-sounding precision.
2. **XAGUSD at 0.38x** is under-costed by the flat 4 bps metals assumption.
   Silver's real spread is ~10.4 bps.

## Verdict impact

- The 5 UNDER-costed pairs (Nordic block + silver) were OPTIMISTIC in every
  G10-22 wave: Wave 1, Wave 2, and the OHLC wave. Those all FAILED, so no verdict
  flips -- an optimistic cost cannot rescue a failing result.
- Concretely relevant: **TOT-OIL** (the Tier B near-miss, +0.05 at 1x surviving
  1.5x) traded USDCAD and **USDNOK**, and USDNOK is 17x under-costed. Its small
  positive is therefore MORE optimistic than reported, not less. Correcting the
  model moves it further from passing.
- Majors were over-costed 2.4-7x. That is conservative, and the 2026-07-19
  cost-sensitivity re-gate already tested 0.5x costs and found 6/6 failures
  robust on PSR/DSR/PBO (not merely on Sharpe sign), so this is unlikely to flip
  anything either.

## Caveat that must travel with this data

Dukascopy quotes RAW ECN spreads. A retail account also pays commission
(IBKR-style FX commission is roughly 0.2 bps of notional per side, with a
minimum). These measurements are therefore a **LOWER BOUND** on realistic retail
cost, not a drop-in replacement for the taker model. Any recalibration must add
an explicit commission/slippage uplift, and that uplift is a modelling choice
that sets how hard every future gate is.

## Artifact

`<local_storage>/artifacts/fx/measured_spreads/table.parquet` -- per pair, sample
year/month, and hour-of-week: n, spread_p50, spread_p90. Rebuild with
`PYTHONPATH=$(pwd) python scripts/data/measure_fx_spreads.py`.
