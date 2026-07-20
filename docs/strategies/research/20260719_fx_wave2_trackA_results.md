# FX Catalog Wave 2, Track A -- Results (#33, #39, #42)

**Date:** 2026-07-19
**Pre-registration:** `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`
**Generator:** `scripts/backtest_scripts/run_fx_wave2_gate.py`
**Gate:** methodology Section 2.5 combined statistical gate (PSR>=0.95, DSR>=0.95 using
the honest project-wide growing trial count, PBO<0.25, 1.5x cost-sensitivity survival).
S&P correlation / IR / marginal contribution are book-level context, non-gating.

## Summary

| # | Strategy | OOS Sharpe (1x) | OOS Sharpe (1.5x) | PSR | DSR | PBO | Trials (N) | S&P corr | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 33 | Turn-of-Month USD | -0.2824 | -0.3555 | 0.0000 | 0.0000 | 0.8360 | 104 | 0.0301 | REJECT |
| 39 | PCA Dollar-Factor Residual | -0.1214 | -0.2222 | 0.0000 | 0.0000 | 0.3790 | 105 | 0.0198 | REJECT |
| 42 | RORO Regime Spread | 0.0578 | -0.0293 | 0.9993 | 0.0000 | 0.1733 | 106 | 0.0022 | WEAK |

**None of the 3 Track A strategies clears the combined gate.** Full reports:
`docs/reports/fx/fx_turn_of_month_wave2_gate.md`,
`docs/reports/fx/fx_pca_dollar_residual_wave2_gate.md`,
`docs/reports/fx/fx_roro_regime_spread_wave2_gate.md`.

## #33 Turn-of-Month USD -- REJECT (confirmed, unchanged from prior turn)

OOS Sharpe -0.28, PSR/DSR 0, PBO 0.84 (badly overfit -- most of the backtest performance
is not reproducible out-of-sample), S&P corr 0.03. Non-positive OOS Sharpe: no edge to
deflate or gate. Applying the pre-registered #33 discipline: gated once, FAIL, no
tweak-and-re-run.

## #39 PCA Dollar-Factor Residual -- fixed and re-gated, REJECT

**Bug (as reported):** `IndexError('index 0 is out of bounds for axis 0 with size 0')`
in `dollar_factor()` -- `np.linalg.svd` on a zero-row matrix when
`returns_df.dropna(how="any")` produced an empty frame, then `vt[0]` indexed into an
empty array. Root cause: several of the 22 pairs (the Nordic/exotic crosses --
EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY) have gaps or shorter history
within some trailing 250-day windows in the FX 60-catalog's early years; requiring
*every* pair to be simultaneously non-NaN emptied the window entirely for some
rebalance cycles.

**Fix** (`src/strategies/advanced/fx_pca_dollar_residual.py::forecast_panel`): instead
of dropping ROWS with any NaN (which zeroes the whole window if even one exotic pair
has a gap on a single day), drop COLUMNS that are not fully populated over the trailing
window, and require at least `2 * n_legs` complete-history columns to proceed
(otherwise flatten and skip that rebalance cycle, without touching the `pc1_jump`
tracking state so the next valid window's jump-check isn't corrupted by the gap). This
preserves more usable history than the naive fix and is closer to the pre-registered
"exotics inform the factor" design than simply shrinking the eligible universe.

**Re-gate result:** OOS Sharpe -0.12 (1.5x: -0.22), PSR 0, DSR 0 (N=105), PBO 0.38, S&P
corr 0.02. Non-positive OOS Sharpe -- REJECT.

**Data-quality caveat (non-blocking, noted for the record):** the re-gate run emitted
`RuntimeWarning: overflow/invalid value encountered in matmul` during a handful of the
early/thin-data windows, consistent with a column having (near-)zero return variance
inside its trailing window (e.g., a stale/flat-priced illiquid pair), which produces an
`inf` in the standardized `Z` array before the SVD. This did not crash the run and the
resulting monthly Sharpe/return series are all finite and in a sane range, so it does
not appear to have corrupted the reported verdict -- but it means a few early-history
PC1 estimates may be numerically degenerate rather than economically meaningful. Given
the strategy already fails the primary gate decisively (negative OOS Sharpe, DSR=0,
PBO=0.38), this was not pursued further; it would be an apparatus-hardening item (guard
zero-variance columns before standardizing) rather than something that could plausibly
flip the REJECT verdict.

## #42 RORO Regime Spread -- fixed and re-gated, WEAK (fails gate)

**Bug (as reported):** `ValueError('cannot convert JPY to USD: neither JPYUSD nor
USDJPY in panel')`. Root cause: `config/backtesting/fx_roro_regime_spread.yaml`'s
universe was `[AUDJPY, CHFJPY, XAUUSD]` -- both traded legs are JPY-quoted, so
`FxSpotPortfolioSimulator` / `build_quote_usd_panel` needs a JPY->USD conversion leg
(USDJPY) in the loaded panel to mark P&L to USD. XAUUSD is already USD-quoted, so no
conversion was needed there.

**Fix:**
- `config/backtesting/fx_roro_regime_spread.yaml`: added `USDJPY` to `universe`
  (now `[AUDJPY, CHFJPY, XAUUSD, USDJPY]`) so the panel loader has the conversion leg.
- `src/strategies/advanced/fx_roro_regime_spread.py::forecast_panel`: the strategy only
  ever traded AUDJPY/CHFJPY and hardcoded a 3-column output dict (`AUDJPY`, `CHFJPY`,
  `XAUUSD`), which would `KeyError` once a 4th universe member (USDJPY) was added to
  `cols`. Changed to default every column in `cols` to a 0.0 forecast, then override
  only `AUDJPY`/`CHFJPY` from the state machine -- so USDJPY (and XAUUSD) are present
  in the panel for currency conversion but never traded, matching the pre-registered
  design ("XAUUSD is a score-only input -- it is never traded").

**Re-gate result:** OOS Sharpe +0.06 (1x cost) but -0.03 at 1.5x cost -- **fails the
mandatory cost-sensitivity gate outright (goes negative under a 50% cost stress)**.
PSR is high (0.9993, i.e. against the null of SR=0 without trial-count deflation) but
DSR -- the same statistic penalized for the honest, growing 106-trial project-wide
search -- is 0.0000: no statistical evidence of genuine skill once the search cost is
priced in. PBO 0.17 (passes the <0.25 threshold in isolation, but is moot given DSR/cost
failure). S&P correlation 0.0022 -- genuinely market-neutral by construction, as
expected for a beta-weighted spread, but book-level diversification value requires a
real (cost-surviving, DSR-supported) edge first, which this does not have.

**Does this meet the pre-registered "genuinely close" bar (Section 6)?** No. The
stopping rule's "comes genuinely close" language means a positive DEFLATED Sharpe (i.e.
DSR meaningfully positive, short of but approaching 0.95) with low correlation. Here DSR
is exactly 0.0000 and the edge does not even survive a 1.5x cost stress -- the 1x
positive Sharpe is cost-fragile, not marginal-but-real. This is a clear FAIL, not a
close call.

## Trial-count integrity check

Verified against `output/experiments.duckdb`: exactly one registered run per strategy
(`FxTurnOfMonth`, `FxPcaDollarResidual`, `FxRoroRegimeSpread` -- one `run_id` each,
no duplicates from the earlier crashed attempts, which errored before reaching the
registry-append step inside `walk_forward_fx`). The honest project-wide trial count
grew monotonically and by exactly one per gate call: 104 -> 105 -> 106, consistent with
#33's already-reported N=104 from the prior turn. Per the North Star and the Wave 2
pre-registration (Section 5), this growing N is the load-bearing protection behind the
DSR gate -- every one of these 3 specifications, pass or fail, permanently raises the
bar for whatever runs next.

## Does Track A trigger the pre-registered stopping rule?

**Not yet decidable from Track A alone.** The Wave 2 pre-registration's stopping rule
(Section 6) is scoped to all 6 Wave 2 strategies: 3 Track A (#33, #39, #42, now
complete, all FAIL/WEAK) + 3 Track B (#35 AUD/NZD spread, #37 cointegration scanner,
#30 XAU/XAG relative-vol -- all `SPREAD`-blocked, pending the beta-weighted
spread-execution engine, not built in this session). Track A alone does not clear the
combined gate for any strategy, so it does NOT trigger the "defines Wave 3" branch on
its own. Whether the full Wave 2 campaign concludes "declare the finding and stop" (all
6 fail) or "Wave 3, scoped to the surviving mechanism" depends on Track B, which remains
outstanding.
