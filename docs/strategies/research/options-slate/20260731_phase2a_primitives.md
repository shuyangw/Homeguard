# Options Slate -- Phase 2a: Shared Primitives, Mark Convention, Cost Model (2026-07-31)

**Scope:** the ten shared primitives **P1-P10**, the mark convention **M2**, and the cost model
**M3**. A library build.

**No strategy backtest was run. No P&L, no positions, no equity curve, no strategy verdict of any
kind appears in this document or in the code it describes.**

**Deliberately NOT built** (Phase 2b): M1 early assignment/pin, M4 P&L decomposition, M5
regime-sliced attribution, M6 CPCV wrapper, the options walk-forward runner. They are designed
around, not stubbed -- there is no placeholder for any of them.

**Ground truth is code and disk.** Every measurement below was taken on the materialized stores at
the stated snapshot. Where this brief, the doc chain and disk disagreed, disk won and the
divergence is in Section 7.

---

## 1. What was built

| File | Contents |
|---|---|
| `src/backtesting/options/snapshot.py` | the ONE snapshot-symmetry guard (spec Section 1.3 rule 1) |
| `src/backtesting/options/marks.py` | **M2** -- the mark convention, as amended by A3 |
| `src/backtesting/options/cost_model.py` | **M3** -- `cost_model_v2`, parameterized from `spread_census` |
| `src/backtesting/options/primitives.py` | **P1-P10** |
| `tests/backtesting/options/` | 107 tests, including four negative controls |

Full paths:
`C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\backtesting\options\snapshot.py` |
`...\marks.py` | `...\cost_model.py` | `...\primitives.py` |
`C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\backtesting\options\`

Reuse, not forks: `src/features/volatility.py` (Yang-Zhang -> P9),
`src/backtesting/vol/har_rv.py` (-> P10), `src/backtesting/costs/options.py` (-> M3),
`src/data/options/canonical.py`, `iv_surface.py`, `derived_store.py`.

**Promotion completed.** The Wave-0 diagnostic forms of `trailing_percentile`, `trailing_rank`,
`greek_usable_mask`, `select_nearest_abs_delta`, `third_friday` / `monthly_expiry` /
`monthly_expiry_set` were **moved** into the registered primitives.
`src/backtesting/diagnostics/options_iv_state.py` now **re-exports** them. There is exactly one
implementation of each; two copies of a lookahead guard is precisely how two copies drift apart.

---

## 2. The primitives, registered spec vs what was built

| ID | Registered spec (spec v2 Section 3) | Built | Deviation |
|---|---|---|---|
| **P1** | `select_strike_by_delta(chain, right, target_delta, dte_window)` -- nearest \|delta\| to target in the window, ties to the more liquid strike; for \|target\| < 0.10 read delta/IV from the smoothed surface; return the strike **plus the realized delta** | as registered, plus `delta_source` and `tie_break` on every return | see 2.1 |
| **P2** | `select_expiry(chain, dte_min, dte_max, prefer)`, `prefer in {monthly, any}`; never straddle a distribution silently -- log the chosen expiry | as registered; returns exactly one expiry with its candidate count | **which** expiry inside the window is unregistered -- see 2.2 |
| **P3** | `standard_exit(position, profit_take, dte_exit)` -- profit-take fraction of credit **or** `dte_exit`, whichever first, evaluated once daily at snapshot. Defaults 0.50 / 21 | as registered | takes scalars, not a position object -- see 2.3 |
| **P4** | `size_by_vega_budget(structure, budget_vega, nav)`; the **single shared budget constant** across F1/F3/F6, fixed at spec time, never tuned. `size_by_debit` / `size_by_notional` also exist | as registered; all three sizers built | **the constant has no registered value** -- see 2.4 |
| **P5** | `regime_gate(state, allowed_states)` from the point-in-time classifier | as registered | same-session read is a partial leak -- see 2.5 |
| **P6** | `hedge_ledger(position, bars_1m, mode, band, cost_bps_schedule)`; `mode in {daily_1545, band}`; slippage **tiered by realized-vol percentile**; emits a ledger of every fill | as registered | **the bps values are unregistered** -- see 2.4 |
| **P7** | `roll(position, roll_trigger, new_selector)` -- logs each roll as a separate cost event | as registered; emits **two** cost events per roll (close + open), never netted | none |
| **P8** | `iv_rank` / `iv_percentile` from `iv_rank_daily`, strictly backward looking | as registered | rank is unclipped -- see 2.6 |
| **P9** | `yang_zhang_rv(bars_1m, window_days)`, reuse the toolbelt implementation, enforce the 15:45 truncation internally | as registered, wrapping `src/features/volatility.py` | its daily OHLC differs from Wave-0's -- see 2.7 |
| **P10** | `har_rv_forecast(bars_1m, spec=(1,5,22), horizon)`, spec **frozen**, expanding window, no forward data, 15:45 truncation, **emits its correlation to contemporaneous VIX** | as registered; `spec != (1,5,22)` and `horizon != 1` both raise | none |

### 2.1 P1 -- the delta source is exposed, per call

Four policies, one enum:

- `AUTO` -- **P1 as registered**: the smoothed surface below \|0.10\|, shipped greeks at or above it.
- `SHIPPED` / `SMOOTH` -- explicit overrides.
- `A3_SMOOTH_NONEXTRAP` -- **amendment A3 Section 3.2**: `delta_smooth` where the surface is
  non-extrapolated at the selection point, otherwise `delta_shipped` **with a flag**.

The A3 policy binds **only OPT-047 and OPT-030**. A3 Section 4 explicitly declines a slate-wide
relaxation of P1, so `AUTO` is unchanged for every other candidate and the measured
shipped-greek discrepancy finding (smallest at low delta, 0.0008; largest deep ITM, 0.0187) is
**recorded and not acted on**, exactly as A3 instructs.

Every `StrikeSelection` carries `delta_source` and `extrapolated_fallback`, so the split is
reportable rather than buried. `tie_break` records which rule decided a tie.

**Liquidity tie-break, and the leak it refuses.** Ties break on `oi_eod_lag1` (descending), then
tighter `spread_abs`, then the lower strike so the result is deterministic. The same-session
`oi_eod` is a hard lookahead at the snapshot minute (V6) and `select_strike_by_delta` **raises
`LeakageError`** if it is named. A chain that carries `oi_eod` but no `oi_eod_lag1` simply does
not get an OI tie-break -- it never silently reads the leaky column.

### 2.2 P2 -- a genuine gap in the registration, closed deterministically

The spec fixes the DTE window and the `prefer` flag but **never says which expiry inside the
window wins**. Registered here, before any test ran: **nearest to `target_dte`** (default = the
window midpoint), **ties to the shorter dte**. This is a specification choice, not a tuned
parameter; it is stated so it cannot be quietly re-chosen later.

### 2.3 P3 -- stateless by construction

The registered signature names a `position`. Phase 2a has no position object and must not invent
one, so P3 takes `(entry_credit, current_value, dte)`. Same rule, no state.

Two behaviours worth naming: it **requires a positive credit** (it is a credit rule; debit
structures name their own exit), and an **unmarked** position (`current_value` NaN, i.e. no valid
quote) **does not exit** -- it returns `unmarked=True` to be reported, per M2/A3.

### 2.4 P4 and P6 -- two registered-but-unvalued parameters. Both fail loud.

This is the one place the build stopped rather than substituting.

- **P4's shared vega budget.** The chain registers that the constant exists, is shared across
  F1/F3/F6, and is never tuned. It is referenced repeatedly (4 places each in `spec_v2` and
  `slate_v1_1`, plus the v1 documents) and **never given a value** -- including relative
  references that presuppose one ("**half** the OPT-015 vega budget", OPT-018). Note the contrast:
  OPT-047's *premium* budget **is** registered numerically (40 bps/month), so the omission is
  specific to the vega budget, not a general looseness. `shared_vega_budget()` raises
  `UnregisteredParameterError`.
- **P6's slippage schedule.** The chain registers that slippage is *tiered by realized-vol
  percentile* and never states the bps. `cost_bps_schedule` is a **required** argument with no
  default.

A silently substituted parameter is an extra trial. Both need a logged amendment before the
candidates that depend on them can run. This blocks OPT-015/016/018/044 and every delta-hedged
candidate at sizing/hedging time -- not at build time.

### 2.5 P5 -- a leak found in the regime table, reported not papered over

`regime_state_daily` builds session t's row from session t's **close**. Consuming it at the 15:45
snapshot on session t is therefore a **partial same-session look-forward**.

`regime_state_at(session, lag_sessions=0)` defaults to 0 because that is the house convention
(RAMP reads the same-day classifier in production), but the parameter is explicit at every call
site and `lag_sessions=1` removes the leak. **A candidate whose gate result changes between
lag 0 and lag 1 has a result that depends on the leak**, and that comparison should be run.

`regime_gate(None, ...)` returns **False**: an unknown regime is not a licence to trade.

### 2.6 P8 -- IV rank is not clipped, deliberately

The window is **strictly prior**, so a session that sets a new IV high ranks **above 1.0**.
Clipping would hide the exact state the rank is read for, and the materialized `iv_rank_daily`
already carries unclipped values. Percentile is bounded [0,1] by construction. Documented and
pinned by a test.

### 2.7 P9 -- its daily OHLC is not Wave-0's, by design

P9 enforces the 15:45 truncation, so its daily bar is 09:30-15:45. The Wave-0
`build_rv_daily` aggregates the **full** regular session (to 15:59). The two differ by
construction; snapshot symmetry requires the truncated one for anything a strategy reads.
Both remain; they answer different questions.

---

## 3. M2 -- the mark convention

- `mid = (bid + ask) / 2`, **valid quotes only**.
- **Invalid rows are FLAGGED, never dropped and never repaired.** Reason codes: `zero_bid`,
  `crossed`, `nonfinite_quote`, `zero_quote_open_bar`. Row count in equals row count out.
  `unmarked_census()` produces the by-reason count A3 requires for OPT-047/030.
- **No imputation, forward-fill, interpolation or smoothing anywhere.** A rejected row keeps
  `mark = NaN`.
- **Trade prints are refused loudly.** `assert_not_trade_print` raises on
  `open/high/low/close/vwap/volume/trade_count/day_volume`.
- **The 09:30 bar has its own reason code.** `bid == ask == 0` there is a structural artifact of
  the pre-open state, not a genuine zero bid; lumping them together would corrupt every quote
  census. Verified on disk (Section 5).
- **Surface cross-check, never substitution.** Below \|delta\| < 0.10, `|iv_shipped - iv_smooth|`
  in vol points is recorded and flagged above the registered **1.5 vol point** bound. The mark
  stays the quote mid.
- **A3 guard.** `assert_mark_source_allowed("OPT-047", "iv_smooth")` raises.
  `"OPT-027"` is unaffected -- it sits at 0.10 delta exactly and P1's `< 0.10` rule does not bind
  (A3 Section 3.3).
- **The NaN-vs-NULL trap is closed.** `greek_usable_mask` tests values with `np.isfinite`, never
  `null_count`, `is_null` or column presence.

---

## 4. M3 -- the cost model, and the unit reconciliation

### 4.1 The unit trap, with worked numbers

Two conventions differ by a factor of **two**:

| Convention | Where | Meaning |
|---|---|---|
| fraction of **FULL quoted width** beyond mid, per transit | the doc chain ("single leg, normal: 25%") | `slippage = f * (ask - bid)` |
| `alpha`, fraction of the **HALF-spread** | `src/backtesting/costs/options.py`, methodology Section 4.5 | `fill = mid +/- alpha * (1/2)(ask - bid)` |

**`alpha = 2 * width_fraction`. 25% of width == alpha 0.50.**

Worked, bid `1.00` / ask `1.20` (width `0.20`, half-spread `0.10`):

```
25% of width : 0.25 * 0.20 = $0.05 / share = $5.00 / contract
alpha 0.50   : 0.50 * 0.10 = $0.05 / share = $5.00 / contract   <- same number
```

`options_slippage_per_contract(1.00, 1.20, side="buy", override_alpha=0.50)` returns
`fill = 1.15`, `slippage = 0.05`. That equality is asserted in
`test_reconciliation_against_the_existing_alpha_module_with_worked_numbers`.

The existing alpha table translated into width fractions, so the two are never conflated again:

| tier | `alpha` | == width fraction |
|---|---|---|
| `very_liquid` | 0.4 | **0.20** |
| `liquid_etf` | 0.6 | **0.30** |
| `single_stock_atm` | 0.85 | **0.425** |
| `wings_illiquid` | 1.1 | **0.55** |

The pre-existing module is **extended, not forked**: `slippage_via_alpha_module()` is the single
bridge, so any drift surfaces as a test failure rather than a silently doubled cost.

### 4.2 Registered fill fractions

| Case | Fraction of width, per leg, per transit |
|---|---|
| single leg, normal | **0.25** |
| native combo, 2-4 legs, normal | **0.03-0.06**, midpoint 0.045 |
| stressed tier, combos at single-leg grade | **0.15-0.25**, midpoint 0.20 |
| 0DTE | always single-leg grade |
| 5+ legs | single-leg grade (native combo orders cover 2-4) |

Stressed triggers, as registered: event window, `regime_state in {UNPREDICTABLE, BEAR}`, or
realized-vol percentile **> 0.80**.

### 4.3 Fee stack, per contract, one way

| Component | Value |
|---|---|
| commission | $0.35-0.65, default **$0.50** |
| OCC clearing | $0.025 |
| ORF | $0.023 |
| CAT | $0.0003 |
| FINRA TAF | $0.00329, **sells only** |
| SEC Section 31 | $20.60 per $1M of **sale** proceeds, **from 2026-04-04** |

At the default commission: **buy $0.5483, sell $0.5516** -- inside the registered $0.42-0.72 band.
A divergence at the band edges is in Section 7.

### 4.4 `spread_census` coverage -- measured, per root and per operating bucket

Widths come from the census; the assumed table is the **recorded per-leg fallback**. Every call
returns `width_source_by_leg`. A cell with fewer than **100** sampled quotes falls back.

Coverage over each root's honest usable window:

| root | window | census cells | cells with n >= 100 | fraction |
|---|---|---|---|---|
| SPY | 2018-2025 | 4,668 | 3,865 | **0.828** |
| IWM | 2017-2025 | 5,041 | 4,051 | **0.804** |
| QQQ | 2012-2025 | 7,997 | 6,490 | **0.812** |

The ~19% shortfall is concentrated in structurally thin cells (`d_null_or_oob`, `m_undefined`,
`vol_unknown`). **At the buckets Wave-1 candidates actually operate in, coverage is complete:**

| Operating bucket | SPY | IWM | QQQ | census median width |
|---|---|---|---|---|
| ATM, 31-60 DTE, 0.35-0.65 delta (straddle / IC body) | 24/24 | 26/26 | 42/42 | 0.040 / 0.050 / 0.045 |
| 2-5% OTM put, 31-60 DTE, 0.15-0.35 delta | 24/24 | 26/26 | 42/42 | 0.030 / 0.040 / 0.030 |
| far-OTM put, 91-180 DTE, <0.05 delta (OPT-047) | 24/24 | 26/26 | 42/42 | 0.010 / 0.030 / 0.030 |
| 0DTE ATM | 24/24 | 26/26 | **41/42** | 0.0125 / 0.020 / 0.020 |
| 5-10% OTM call, 31-60 DTE, 0.05-0.15 delta | 24/24 | 26/26 | 42/42 | 0.015 / 0.020 / 0.020 |

(cells with n >= 100 / cells present; one QQQ 0DTE root-year is thin.)

**The measured widths are materially TIGHTER than the assumed table**, which matters:

| bucket | assumed midpoint | measured median | assumed / measured |
|---|---|---|---|
| index ATM 30-45 DTE | 0.0750 | 0.040 (SPY) | **1.88x** |
| index low delta | 0.0400 | 0.010 (SPY) | **4.0x** |
| index 0DTE ATM | 0.0150 | 0.0125 (SPY) | 1.2x |

Running Wave 1 on the assumed table would overstate index option costs by roughly **2x at ATM and
4x in the wings**. This is exactly why V11 was made a Wave-0 deliverable, and it is the single
most consequential number in this document.

### 4.5 Smoke on a real chain -- SPY 2024-06-10, 30-delta put, 32 DTE, 10 contracts

| width source | width | entry | exit | fees | total |
|---|---|---|---|---|---|
| `quote` (this contract's own quote) | 0.0200 | $5.00 | $5.00 | $11.00 | $21.00 |
| `census` (registered default) | 0.0400 | $10.00 | $10.00 | $11.00 | $31.00 |
| `assumed` (fallback) | 0.0750 | $18.75 | $18.75 | $11.00 | $48.50 |

Mandatory sensitivity band on the census figure: **0.5x $15.50 | 1.0x $31.00 | 1.5x $46.50**.

Same session, M2 and P1 on all 3,083 contracts: **100.00% marked**, zero unmarked rows, **128
low-delta surface-divergence flags**. P2 chose one expiry (2024-07-12, 32 DTE) from 3 candidates.
P1 at -0.30 selected K=527, realized delta -0.2999, source `shipped`. P1 at -0.05 selected K=489,
realized delta -0.0500, source `smooth`, and the A3 policy selected **the same strike from the
same source** -- the surface was not extrapolated at that operating point, matching A3's finding
of 0.0% extrapolation at OPT-047's operating point.

---

## 5. Verification and negative controls

107 tests. Four are **negative controls** -- without them the corresponding positive tests would
pass vacuously:

1. **Lookahead-injecting truncation.** A truncation admitting one extra minute (<= 15:46) must be
   REJECTED by `verify_snapshot_truncation`. It is.
2. **Padded early-close bar.** Truncating a half day at the registered 15:45 instead of the real
   close must be REJECTED. It is.
3. **Full-sample percentile.** `assert_backward_looking(lambda x: x.rank(pct=True), s)` must
   raise `LeakageError`. It does; `trailing_percentile` / `trailing_rank` pass.
4. **Post-snapshot tampering.** Multiplying every post-15:45 bar by 5 must not move P9's output by
   one bit, and multiplying the tail of the RV series must not move P10's earlier forecasts.
   Neither does.

Disk-backed regression test (skips where the store is absent): the 391-label, zero-quoted-09:30,
zero-volume-16:00 structure of the options store, which is what makes the two stores' clamps
differ.

Regression run over every touched area -- `tests/backtesting/options`,
`tests/backtesting/test_diagnostics`, `tests/data/test_options`, `tests/backtesting/costs`,
`tests/backtesting/vol`: **381 passed, 0 failed**.

---

## 6. What Phase 2b may assume from this

1. The snapshot is truncated in **one** function for underlying bars
   (`snapshot.truncate_to_snapshot`) and in **one** place for options (`canonical`). Nothing else
   slices bars.
2. Every mark carries a validity flag and a reason; unmarked is a reportable state, never a gap
   to fill.
3. Every cost call returns its tier, its width fraction, its alpha equivalent, and its per-leg
   width provenance, and is reportable at 1.0x and +/-50%.
4. Every P1 selection reports which delta source chose it.
5. **P4 and P6 will raise until their constants are registered by amendment.** This is intended.

---

## 7. Divergences -- brief / doc chain vs disk and code

Reported because reality wins.

| # | Asserted | Found | Consequence |
|---|---|---|---|
| 1 | (this brief) the worktree is ready to build in | The supplied worktree was branched from a stale point **17 commits off `origin/main`** and contained **none** of the Phase-1 options data layer -- no `canonical.py`, no `iv_surface.py`, no `derived_store.py`, no `options_iv_state.py`. The Phase-1 work lives on **local `main` only** (51 commits ahead of `origin/main`, unpushed) | Realigned the worktree to local `main` (`26409b5`) before building. Nothing was discarded: the 17 stale commits are on `origin/main` and on two other worktree branches. **Phase 1 is not on the remote** -- if this host is lost, so is the data layer |
| 2 | (my own working note) `canonical._snapshot_cutoff_expr` admits one padded bar on early closes | **FALSE, and I retracted it.** The options store's quote fields are the quote AS OF the labelled minute (measured: 391 labels 09:30..16:00; the 09:30 label is zero-quoted in **100.00%** of rows yet carries volume in **45%**; the 16:00 label has live quotes and **zero** volume in 100%). The row at an early close is that session's CLOSING quote. canonical is correct | No change to canonical. My guard covers the equity 1m store, which IS bar-START (measured: 04:00..19:59) |
| 3 | half days are padded to 16:00 with stale quotes | **CONFIRMED.** SPY 2024-07-03 (13:00 close) runs to 16:00 with volume 0 and a frozen quote: mean mid **6.8692** from ~13:1x onward against **6.9018** at the 13:00 bar | The clamp is load-bearing. Reading 15:45 blindly marks half days off a stale quote |
| 4 | the 09:30 bar has `bid == ask == 0` universally | **CONFIRMED at 1.0000**, and it also carries **volume in 45%** of rows -- it is a real trade bar with no quote, not an empty row | Given its own reject reason (`zero_quote_open_bar`) rather than being counted as a zero bid |
| 5 | P4's vega budget is "fixed at spec time" | **No value is stated anywhere in the chain.** Every reference names the constant; none sets it -- and OPT-018 sizes at "**half** the OPT-015 vega budget", a relative reference to a number that does not exist. OPT-047's premium budget, by contrast, IS registered (40 bps/month) | `shared_vega_budget()` raises. Needs a logged amendment before F1/F3/F6 sizing |
| 6 | P6's slippage is "tiered by realized-vol percentile" | The **tiering** is registered; **no bps values are** | `cost_bps_schedule` is required, no default |
| 7 | the fee stack sums to $0.42-0.72 one way | The stated components do not reproduce the stated band. Ex-commission: buy **$0.0483**, sell **$0.05159**. With commission at the quoted **$0.35** floor, buy = **$0.3983** (below $0.42); at the **$0.65** ceiling, sell = **$0.7016** (below $0.72). The band implies a commission floor near $0.372 | ~2 cents per contract at each edge; immaterial at these premia but the arithmetic is stated so it is not rediscovered. Defaults sit comfortably inside the band |
| 8 | P2's expiry rule is registered | The DTE window and `prefer` are; **which expiry inside the window wins is not** | Registered here as nearest-to-target, ties to shorter. A specification choice, stated before any test ran |
| 9 | `regime_state_daily` is point-in-time | Session t's row is built from session t's **close**, so a 15:45 read is a partial same-session look-forward | `lag_sessions` exposed, default 0 (house convention), leak stated. Candidates should compare lag 0 vs lag 1 |
| 10 | (spec Section 2) the assumed width table parameterizes costs | It is **~2x too wide at index ATM and ~4x too wide in the index wings** against the measured census | `cost_model_v2` reads the census by default. Any result computed off the assumed table would materially understate index option strategies |
| 11 | "the stressed tier treats combos at single-leg grade **15-25%**" while "single leg, normal" is a point **25%** | The two statements are not consistent -- "single-leg grade" is quoted as a band whose top equals the single-leg point | Implemented as stated: 0.25 point for single legs, 0.15-0.25 band (midpoint 0.20) for stressed combos. Flagged rather than reconciled unilaterally |
| 12 | `src/backtesting/diagnostics/options_iv_state.py` resolves paths via `get_local_storage_dir()` | It **hardcodes** `Path("H:/Stock_Data/options/options_chain_eod")` (line 40) | Pre-existing, out of scope for this pass, not touched. Worth closing -- it will break on any other host |

---

## 8. Reproduction

```bash
# tests (107 in this package; 381 across every touched area)
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/backtesting/options/ -q

POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/backtesting/options/ tests/backtesting/test_diagnostics/ \
    tests/data/test_options/ tests/backtesting/costs/ tests/backtesting/vol/ -q
```

Commits (branch `worktree-agent-a6a6e96aace594c60`, **not pushed**):

- `73cca62` feat(options): Phase 2a -- P1-P10 primitives, M2 marks, M3 cost model
- `3d419fe` fix(options): correct a FALSE divergence claim about the snapshot clamp

---

## 9. Open items for Phase 2b

1. **P4's vega budget and P6's bps schedule need a logged amendment.** Both currently raise.
2. **The `regime_state_daily` same-session read (divergence 9)** should be decided explicitly
   before any regime-gated candidate runs.
3. **Phase 1 is unpushed** (divergence 1). The entire options data layer exists on one host only.
4. The one thin QQQ 0DTE census cell, and the ~19% of structurally thin census cells, will fall
   back to the assumed table -- which Section 4.4 shows is ~2-4x too wide. Any candidate whose
   costs land materially on fallback cells should say so.
5. `options_iv_state.py`'s hardcoded storage path (divergence 12).

*No strategy backtest has been run. No P&L has been observed.*
