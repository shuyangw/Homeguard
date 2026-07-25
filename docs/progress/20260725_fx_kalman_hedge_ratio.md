# Kalman Hedge-Ratio Scoping Diagnostic - 2026-07-25

## Summary
Ran the pre-registered scoping diagnostic: does a causal Kalman time-varying hedge
ratio change the pairs verdict, or did our Wave 2 negative bound only *static-OLS-
hedged* pairs trading? **Arm B FAILS the gate (registered prediction confirmed), but
for a reason that was NOT anticipated** -- it posted the campaign's first positive OOS
Sharpe (+0.42), and the honest reading is that this exposes a DEGENERATE BASELINE in
Arm A rather than a working Kalman mechanism. Several apparatus defects were found
that matter beyond this strategy.

## Result (both arms, per the pre-registered both-arms-always rule)

| Metric | Arm A (static 120d OLS) | Arm B (Kalman, corrected) |
|---|---|---|
| OOS Sharpe 1x | -0.2362 | **+0.4171** |
| OOS Sharpe 1.5x | -0.3016 | **+0.3466** |
| PSR | 0.0000 | 1.0000 |
| DSR | 0.0000 (N=109) | 9.96e-186 (N=133) |
| PBO | 0.8242 | **0.8931** |
| OOS fills | 286 | 260 |
| S&P corr | 0.0436 | 0.0261 |
| n_windows / n_oos_days | 13 / 3185 | 13 / 3185 |

**VERDICT: FAIL.** DSR is ~0 (deflated at N=133) and PBO 0.89 is far above the 0.5
ceiling. The positive Sharpe does not survive deflation for the search, and the high
PBO says the configuration is very likely overfit.

## Why the positive Sharpe is NOT a Kalman success (the load-bearing interpretation)

Arm A's rolling-OLS beta ranges over **[0.013, 1.656]**. A hedge ratio of 0.013 on two
cointegrated commodity currencies is not a pairs trade at all -- it is an outright long
AUDUSD wearing a pairs label, i.e. a near-collinear, barely-identified I(1) fit. At
delta=1e-4 the Kalman acts as **shrinkage, not time-variation** (beta range [0.499,
0.925], std 0.079, daily |d beta| 0.15x OLS).

So the 0.65 Sharpe swing measures *removing a broken baseline estimator*, not *adding
dynamic hedging*. Arm A's -0.24 was therefore partly an artifact of a degenerate
estimator, not a clean read on the mechanism. strategy-lead deliberately did not report
the sign flip as a Kalman benefit; that judgment is correct and is the main finding.

## Scoped conclusion

Hedge-ratio mis-specification is **eliminated** as the explanation for #35's Wave 2
failure: AUDUSD/NZDUSD pairs trading fails net of costs under BOTH a static rolling-OLS
AND a regularized dynamic hedge ratio. This does **not** generalize -- one pair, one
delta, daily bars, spread-TAKER costs, zero execution lag, and an event filter that is
inert in 12/13 windows. Not a lead; no further iterations, per the pre-commitment.

## Apparatus defects found (beyond this strategy -- these are the real deliverable)

1. **The spread walk-forward runner discarded its fills** -- kept `equity_curve`, threw
   away `res.trades`, violating the standing fills mandate. Now wired to the run-scoped
   FillSink (mirrors `run_fx_walkforward.py`). Verified: 260-row `trades_oos.csv.gz` +
   54-row `manifest.csv`.
2. **NaN R-seed silently zeroed two whole windows.** A literal `[:lookback]` seed slice
   went NaN on a leading non-finite block, poisoning the entire causal recursion and
   producing an empty book for the window. The two affected windows were Arm A's two
   WORST (-6.82%, -10.09%), so Arm B was sitting them out. Fixed via
   `select_warmup_indices` (first `warmup` FINITE aligned rows). strategy-lead first
   claimed this "substantially" explained the swing and then corrected itself: it
   accounts for only 10.7% (0.4955 -> 0.4171). Behaviour-neutral elsewhere (windows
   2-12 reproduce to 3 dp).
3. **PBO is computed on only the first ~65 of ~260 OOS days** (a 65-day stub window
   truncates all columns). PBO gates the entire Track B vertical -- so every Track B
   PBO figure, including #30/#35/#37, is suspect.
4. **Zero execution lag across the whole FX spread vertical** -- the signal from
   `close_i` fills at `close_i`. For a mean-reversion strategy this is a free lunch and
   is the most likely single explanation for Arm B's positive Sharpe.
5. RBA/RBNZ blackout is a no-op (calendar holds 2 dates, both 2025); no purge/embargo
   exists despite the pre-registration asserting it; plus a still-live silent-skip
   defect that flattens positions without the state machine knowing. Full list of 10 in
   the results doc.

All defects are symmetric or optimistic, so the FAIL verdicts stand a fortiori. But 3
and 4 mean any FUTURE positive from this vertical is untrustworthy until fixed.

## Pre-registration drafting flaw (mine, worth not repeating)

I wrote the gate as "DSR > 0". A DSR is a probability, so 9.96e-186 literally satisfies
it. The authoritative methodology (Section 2.5) requires **DSR >= 0.95**. Both readings
FAIL here (PBO fails independently), so no verdict changes -- but the phrasing was
sloppy and the same loose wording appears in earlier wave pre-registrations.

## Trial accounting

Pre-registered as 1 trial; **consumed 2** (N 131 -> 133) because the apparatus-fix rerun
registers honestly and the defective row is retained -- N is never reduced. Project-wide
cumulative N is now **133**.

## Commits (main = origin = 6508170)
- `de5ad1d` Arm B build (causal filter, no smoother) + pre-registered beta-path pre-check
- `fa8ccfd` NaN-robust Kalman R-seed + mandatory fill logging in the spread walk-forward
- `6508170` results + tracker rescope

## Known issues / remaining work
- **Fix PBO's 65-day truncation and add execution lag** to the FX spread vertical before
  trusting any positive result from it. These are the highest-value apparatus fixes now.
- Purge/embargo genuinely absent in the spread runner despite pre-reg language.
- Still-live silent-skip defect (positions flattened without state-machine knowledge).
- Two data-layer lookaheads remain open from the 2026-07-22 audit: spike cleaner (uses
  t+1) and the FRED monthly rate ffill (~1-2 months early, and it IS the carry signal).

## Validation
14 unit tests pass (incl. the no-lookahead proof and new NaN-seed regressions). Fills
artifact verified non-empty before the verdict was accepted. Adversarial review found no
lookahead in the Kalman path: single forward loop, `select_warmup_indices` is
prefix-determined, and the alpha term cancels algebraically so z-construction is provably
identical between arms apart from the beta source.
