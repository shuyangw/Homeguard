# Tier B Commodity Wave + PSR Unit Fix - 2026-07-25

## Summary
Ran the pre-registered Tier B commodity terms-of-trade wave: **all 3 trials FAIL**,
exactly as predicted. Also landed the Phase-2 OHLC unlock, and fixed a systemic
PSR/DSR unit bug the gate itself surfaced.

## Tier B verdict (N 134 -> 137)

| Trial | Sharpe 1x | Sharpe 1.5x | PSR (corrected) | DSR | PBO | S&P corr | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| TOT-OIL | **+0.0505** | **+0.0385** | 0.5712 | 0.0000 | 0.4927 | -0.150 | FAIL |
| TOT-GOLD | -0.4903 | -0.5238 | 0.0380 | 0.0000 | 0.1713 | +0.081 | FAIL |
| TOT-XS | -0.1702 | -0.1971 | 0.2722 | 0.0000 | 0.1097 | -0.029 | FAIL |

Deflated bar SR_zero = **1.126** (realized cross-trial spread v=0.4278 made the
actual bar HIGHER than the pre-registered ~1.05 estimate). The best observed
Sharpe is 1/22nd of the bar.

**Sign/alignment pre-check passed before gating:** corr(oil momentum, USDCAD
forecast) = -0.7937 (correctly negative); corr(gold momentum, AUDUSD forecast) =
+0.8114 (correctly positive); XS row-sum max 5.3e-15 (market-neutral).

**The interesting part:** TOT-OIL is POSITIVE and SURVIVES 1.5x cost stress. So
the economic channel is verifiably present and correctly signed, and still
produces no deflated edge. That makes this an *economically trivial* edge rather
than a cost-destroyed one, which is a different (and more precisely scoped)
finding than every prior wave, where costs did the killing.

**Scoped conclusion:** the commodity terms-of-trade family is unproductive FOR
THIS daily-momentum-taker EXPRESSION (63d momentum / 252d z / +-2 clip, fixed
signs, weekly rebalance, retail taker costs). NOT a claim that commodity
currencies lack edge -- the channel was verified present. Still live as separate
pre-registrations: other frequencies, maker execution, or a non-momentum
functional form (terms-of-trade LEVEL vs a fair-value anchor).

## PSR/DSR unit bug (systemic, found by the gate, fixed after adjudication)

The z-score scales by `sqrt(n-1)` where n counts RETURN OBSERVATIONS, so the
Sharpe must be PER-OBSERVATION. **Every** walk-forward runner passed an
ANNUALIZED Sharpe with a DAILY n, inflating z by ~sqrt(252) = 15.9. TOT-OIL's
headline PSR of 0.9979 -- a decisive-looking pass -- is actually **0.5712**.

strategy-lead reported both values rather than silently substituting, and
deliberately did NOT patch gate code mid-verdict (that would itself be a
researcher degree of freedom). Fixed here, after adjudication closed.

- `periods_per_year` added to `psr()` and `dsr()`, default 1.0 so per-period
  callers are bit-identical. `dsr()` passes it through, which matters because
  `sr_zero` inherits the units of `trial_sharpes` -- candidate and benchmark must
  be de-annualized together.
- All 13 live call sites migrated to 252: the shared `walkforward_common` gate,
  `session_walkforward`, `satellite_blend`, and the FX spot / spread /
  carry-seatbelt / london-breakout and Carver futures runners.
- Reproduced exactly: 0.9978 -> 0.5712 (matches strategy-lead's independent
  0.9979 -> 0.5712).

**No verdict changes.** The bug inflates PSR for a POSITIVE Sharpe, so it could
only manufacture a false PSR pass -- and nothing ever passed on PSR; every gated
strategy failed on DSR and/or a negative OOS Sharpe independently. Historical PSR
figures in prior wave reports are inflated and should not be re-quoted; the
verdicts they accompany stand.

## Phase 2: OHLC unlock

`run_fx_backtest` extracted `close` and discarded open/high/low before calling
the strategy -- the loader had always carried them. That single line is what made
every ATR / ADX / true-range / Parkinson signal unbuildable (catalog #1, 6, 8,
12, 27, 28, 29, 47). Strategies may now set `wants_ohlc = True` to receive the
full (pair, field) panel; the close-only path stays the default.

Scope honesty: 8 rows unblocked, but only ~5 are genuinely NEW mechanisms. #8
Bollinger's base form is a close-based z-score, i.e. essentially the EM-MEANREV
spec that already failed; only its ADX filter is new.

## Commits (main = origin = 0c6f053)
- `428cdac` Tier B build + LOCK pre-reg + Phase 2 OHLC unlock
- `bf5c38a` Tier B verdict, all 3 FAIL, scoped (strategy-lead)
- `0c6f053` PSR/DSR unit fix across all 13 call sites

## Known issues / remaining work
- `run_fx_walkforward.py` registers run rows but NOT return streams (strategy-lead
  appended 9,580 rows manually), and computes no S&P benchmark check. Both are
  gaps in the CLI runner worth closing.
- Full methodology 11.9 remains partial: exit reasons now land in the fill log,
  but entry/exit pairing and MAE/MFE are still not emitted.
- Next per the plan: the ~5 genuinely new OHLC mechanisms (#12 Keltner, #27
  squeeze, #28 ATR-regime, #6 ADX-gated, #1 ATR trail), then #36 Scandi (Brent is
  now cached) and #44.

## Validation
2351 pass across tests/backtesting + tests/strategies + tests/data (was 2346).
The 24 failures are pre-existing missing-data errors on this machine (futures
data absent, DGS10 not downloaded, sentiment), none in touched modules. Fills
verified non-empty before each Tier B verdict was accepted (1,220 / 1,247 / 2,457
fills + 53-row manifests). Registry rows 489 -> 492; the post-run recompute had
its registry append deliberately disabled so it could not inflate N to 140.
