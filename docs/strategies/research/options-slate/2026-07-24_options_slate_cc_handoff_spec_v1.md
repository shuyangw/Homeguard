# Equity Options Slate — CC Implementation Handoff Spec (v1)

**Date:** 2026-07-24
**Author context:** Homeguard, spec-first workflow. Claude = spec producer / reviewer. CC = execution agent.
**Chain:** `2026-07-24_options_strategy_slate_v1.md` (candidate definitions, priors) → `2026-07-24_options_slate_feasibility_screen_v1.md` (data/cost screens, verdicts, waves) → **this document** (buildable spec).
**Scope:** the **46 candidates still in play** — 21 GO, 18 CONDITIONAL, 7 DEFER — with full implementation detail. OPT-023 (NO-GO) is excluded. The 3 DROP candidates (010, 012, 022) appear in Appendix C as a *do-not-resurrect* record only.
**Status:** Pre-registration of record. Blind. No backtest return has been observed by any stage of this chain.

---

## TL;DR

- **This document is the buildable form of the slate.** It defines the ORATS data layer, a canonical option-chain schema, ten shared strategy primitives (P1–P10), a two-tier cost model, six harness modules, then a per-candidate spec for all 46 live candidates organized in build order (Wave 0 → Wave 4), plus the ledger schema and a five-phase CC implementation plan.
- **CC must not begin Phase 2 (strategy specs) before Phase 0 (repo reconciliation) is complete and reported.** Five integration points are asserted from memory and are *unverified*: the research-harness registry format, the data-provider abstraction's fitness for chain-shaped data, the `strategy_toggle.yaml` conventions, the multiple-testing ledger's location, and whether `TransformedDataProvider` exists yet. Code is ground truth. Do not build against this document's assumptions where the repo disagrees — report the divergence and stop.
- **One leakage trap is specific to this build and easy to miss:** ORATS near-EOD snapshots the chain at approximately 15:46 ET, so every signal computed from 1-minute underlying bars must be truncated at **15:45 ET inclusive**. Pairing a full-session realized-volatility estimate with a 15:46 option snapshot embeds fourteen minutes of lookahead into every Yang-Zhang- and HAR-gated candidate. This is enforced centrally in P9/P10, not per-strategy.
- **Sequencing is the whole design.** Wave 0 costs zero trials and can kill or promote roughly a dozen candidates before any strategy backtest runs. Wave 1 is nine trials. Nothing beyond Wave 1 is authorized by this document — later waves are specified so CC can build reusable machinery once, not so they can be run early.

---

## What this document does NOT do

- It does **not** authorize backtests beyond Wave 0 diagnostics and the Wave 1 slate. Waves 2–4 are specified for build reuse; each requires its named gate to open first.
- It does **not** permit parameter tuning. Every parameter here is pre-registered and fixed. If a value proves unimplementable as written, CC **stops and reports**; it does not substitute a nearby value. A substituted parameter is a silent extra trial.
- It does **not** define GO decisions on strategies. It defines pre-registered *criteria*; the harness computes; STOP/GO on capital remains Shuyang's call.
- It does **not** replace the repo. Where this spec and the code disagree, the code is right and this spec is stale.
- It does **not** treat any performance number as existing. None do.

---

## Phase 0 — Repo reconciliation (mandatory first task, blocking)

CC has previously propagated false claims about repo state. Every item below must be verified by **direct inspection** (`cat`, `ls`, `git log`, `rg`) and reported with the evidence inline — file paths, line numbers, commit SHAs. "I checked and it exists" is not a report. If an item does not exist, say it does not exist; do not describe the version you expected.

| # | Assertion to verify | How | If divergent |
|---|---|---|---|
| 0.1 | Research-harness registry ingest format (fields a strategy spec must expose) | Locate the registry module; print the schema/dataclass | Reconcile §7 ledger + §5 spec fields to the real format; report deltas before building |
| 0.2 | Data-provider abstraction assumes OHLCV-shaped bars | Read the provider base class | Chain data is strike × expiry × date — likely needs a **new provider type**, as prediction markets did. Do not force it into the OHLCV interface |
| 0.3 | Multiple-testing ledger: does it exist, where, what schema | `rg` for the ledger; check research docs dir | If absent, create per §7 and report the chosen path for approval |
| 0.4 | `config/trading/strategy_toggle.yaml` conventions + the known dead `ramp.variant: v11` key | `cat` the file | Do not add options entries until the dead-config question is settled |
| 0.5 | `TransformedDataProvider`, robust z-score, Yang-Zhang, log/rank transforms (Tier-1 toolbelt) | `rg` for each | If Yang-Zhang exists, **reuse it** — P9 must not fork a second implementation |
| 0.6 | S3 Hive-partition conventions + DuckDB attach patterns actually in use | Inspect an existing dataset path and loader | Match existing convention exactly; §2 paths are proposals, not decrees |
| 0.7 | Existing cost-model / slippage module (from RAMP work) | `rg` | Extend rather than fork; RAMP's cost floor lesson lives there |
| 0.8 | Regime classifier interface + historical state log availability | Inspect `MarketRegimeDetector` and any persisted state history | OPT-050 and every regime gate depend on a point-in-time state log. If states were only computed live and never persisted, say so — several gates become unbacktestable as specified |

**Phase 0 output:** a written reconciliation report. Phases 1+ are blocked until it is reviewed.

---

# 1. Data layer

## 1.1 Rung ladder (what is authorized to be purchased, and when)

| Rung | Product | Coverage | Cost | Authorization |
|---|---|---|---|---|
| 0 | Owned: Alpaca SIP underlying 1m OHLCV (2016+); CBOE VIX term structure (free, 2010+); Homeguard classifier history (internal, pending 0.8) | — | $0 | Active now |
| 1 | **ORATS near-EOD historical** — full chains, NBBO-derived bid/ask, smoothed (SMV) greeks + IV, snapshot ≈15:46 ET, 2007–present | ~19.5 y | ~$399 one-time | **Authorized. Buy first — it gates everything.** Re-verify price/tier at checkout |
| 2a | ThetaData Value — 1-minute **quote (NBBO)** + OHLC + open-interest endpoints; **no greeks at this tier** | 2020-01+ (~6.5 y) | ~$40/mo | **Not authorized.** Requires a Wave 0–2 result creating a live intraday question |
| 2b | Databento OPRA CBBO-1m — consolidated minute NBBO, no greeks | 2013+ (~13 y) | usage-based | Alternative to 2a; longer history, DIY implied volatility (IV) |
| 3 | Earnings-calendar API (approximate dates + before/after-market-open flags) | varies | low | Deferred until OPT-020/021 reach the front of the queue |
| — | True point-in-time (PIT) analyst consensus | — | institutional | **Never at retail.** This is why OPT-023 is NO-GO |

Open-interest history for OPT-043 is available from either ORATS or ThetaData's OI endpoint — resolve during Phase 1 and prefer whichever avoids a new subscription.

## 1.2 Canonical internal chain schema

Normalize the vendor payload into one table at ingest. Do not let vendor column names leak into strategy code.

```
options_chain_eod:
  ticker                str      # underlying root
  trade_date            date     # session of the snapshot
  snapshot_ts           timestamp# actual snapshot time (≈15:46 ET) — carried, never assumed
  expiry                date
  dte                   int32    # calendar days; also store dte_trading
  strike                float64
  right                 char     # 'C' | 'P'
  bid, ask              float64  # NBBO-derived at snapshot
  mid                   float64  # (bid+ask)/2, materialized
  smv_value             float64  # ORATS smoothed theoretical value
  iv_bid, iv_ask, iv_mid float64
  iv_smv                float64  # smoothed surface IV — REQUIRED source for |delta| < 0.10
  delta, gamma, theta, vega, rho float64
  spot                  float64  # underlying at snapshot
  open_interest         int64    # T-1 reported; see PIT rule below
  volume                int64
  div_assumption        float64
  residual_yield        float64
  ingest_version        int16
  source                str
```

**Partitioning:** `s3://<bucket>/options/orats_eod/ticker=<T>/year=<YYYY>/month=<MM>/*.parquet`, matching whatever convention Phase 0.6 finds. Query via DuckDB. Expect a large row count (full chains × 19.5 y × universe) — restrict the initial pull to the Wave-1 universe (§1.4) rather than backfilling 5,000 symbols.

**Derived tables to materialize once** (recomputing these per-strategy is both slow and a divergence risk):
- `atm_iv_daily(ticker, trade_date, dte_bucket, atm_iv)` — dte buckets {7, 14, 30, 45, 60, 90}, interpolated on the smoothed surface.
- `skew_daily(ticker, trade_date, dte_bucket, iv_25d_put, iv_25d_call, skew_25d = iv_25d_put − iv_25d_call, curvature)`.
- `term_slope_daily(ticker, trade_date, m1_atm_iv, m2_atm_iv, slope = m2/m1)`.
- `iv_rank_daily(ticker, trade_date, iv_rank_1y, iv_pctile_1y, iv_pctile_2y)` — trailing windows, strictly backward-looking.
- `rv_daily(ticker, trade_date, yz_10d, yz_20d, har_forecast_22d)` — from P9/P10, **truncated at 15:45** (§1.3).

## 1.3 Point-in-time and leakage rules (binding, enforced centrally)

1. **The 15:45 truncation rule.** Signals derived from 1-minute underlying bars must use bars through **15:45 ET inclusive** when paired with a same-day ORATS snapshot. Implement as a single guarded function in the feature layer; strategies never slice bars themselves. Violating this embeds ~14 minutes of lookahead in every RV-gated candidate (OPT-019, 036, 048, 049 and the gates in 005/013/039/048).
2. **Open interest is next-morning data.** Any OI-conditioned selection (OPT-043) uses **T−1** OI. Same-session OI is unavailable at decision time and its use is a hard leak.
3. **Trailing windows are strictly backward-looking and expanding-safe.** IV rank/percentile at date *t* uses only data < *t*. No full-sample percentile computation anywhere.
4. **Earnings dates are an integrity item, not a convenience field.** Dates pulled today may be corrected relative to what was known then. Treat them with the same discipline as ALFRED vintages: store the retrieval timestamp, validate a sample against a second source, and mark affected results as lower-confidence. Applies to OPT-020/021/024/026/045.
5. **Regime-classifier state must be point-in-time.** If Phase 0.8 finds no persisted historical state log, regime gates must be recomputed causally (classifier fitted/applied with no forward information) and that recomputation reported as a build artifact — not silently assumed correct.
6. **No survivorship in the universe.** Single-name universes must be constructed from a point-in-time membership/liquidity snapshot, not today's list. If a PIT option-volume ranking is unavailable, fall back to index membership at date *t* plus a liquidity proxy, and record the compromise.
7. **Corporate actions.** Verify ORATS' adjustment handling explicitly in Phase 1 on a known split (a documented 2020–2024 mega-cap split). Some vendors deliver "as reported" with no adjustment; if so, build the adjustment layer before any single-name candidate runs.

## 1.4 Universe definitions (fixed)

- `U_INDEX` = {SPY, QQQ, IWM}. Wave-1 pull is **SPY + QQQ only** unless a candidate names IWM.
- `U_MEGA10` = the ten largest US-listed single names by option volume at date *t*, from a PIT-constructed ranking (rule 6).
- `U_MEGA20`, `U_TIER1_100` = same construction, 20 and 100 names; the 100-name universe carries the liquidity floor **≥ 1,000 contracts/day trailing 20-session average**, below which multi-leg retail execution degrades badly.
- `U_TOP6_SPY` = the six largest SPY constituents by index weight at date *t* (OPT-044).

**Wave-1 ingest scope:** SPY, QQQ + `U_MEGA20` for the overwrite candidates. Do not pull the full 5,000-symbol ORATS universe.

---

# 2. Cost model (pre-registered, versioned)

One module, `cost_model_v1`, used by every candidate. No strategy may define its own costs.

**Signature:** `cost(legs, regime_state, width_source, tier) -> (entry_cost, exit_cost, fees)`

**Spread convention** (fraction of quoted width paid beyond mid, per transit):
- Single leg, normal conditions: **25%**
- Native multi-leg combo (2–4 legs), normal: **3–6% per leg**
- **Stressed tier** — event windows, `regime_state ∈ {UNPREDICTABLE, BEAR}`, or realized-vol percentile > 80: treat combos at **single-leg grade (15–25%)**
- 0DTE / same-day: single-leg grade always

**Fee stack, per contract, one way:** commission $0.35–0.65 (tiered/Lite); Options Clearing Corporation clearing $0.025; Options Regulatory Fee ≈ $0.023; Consolidated Audit Trail $0.0003; FINRA Trading Activity Fee $0.00329 (sells only); SEC Section 31 at $20.60 per $1M of sale proceeds from **2026-04-04** (negligible at these premia but modeled for completeness). All-in ≈ **$0.42–0.72/contract one way**.

**Width assumptions (normal conditions), pending replacement by own fills:**

| Class | Quoted width |
|---|---|
| SPY/QQQ ATM, 30–45 DTE | $0.03–0.12 |
| SPY 0DTE ATM | $0.01–0.02 |
| SPY 0.05–0.10Δ (value $0.30–1.50) | $0.02–0.06 |
| Tier-1 single name ATM | $0.05–0.15 |
| Rank 50–100 ATM | $0.10–0.30 |
| Event window | 1.5–3× normal |
| LEAPS (12–18M) | $0.30–1.00 |

**Mandatory:** every backtest reports results at **1.0× and ±50%** cost multipliers. A candidate whose sign flips within that band is reported as cost-indeterminate, not as a result. **Replacement trigger:** once any options strategy trades live, own IBKR fill statistics supersede these assumptions and every prior result is re-run — the assumptions are the weakest quantitative input in this chain.

---

# 3. Shared primitives (P1–P10)

Build once, in a primitives module. Candidates reference these by name; duplicated logic is a correctness risk and a silent-divergence risk.

- **P1 `select_strike_by_delta(chain, right, target_delta, dte_window)`** — nearest available |delta| to target within the DTE window, ties to the more liquid strike. **Rule:** for |target_delta| < 0.10, read delta and IV from `iv_smv`/smoothed greeks, never from raw quotes (deep-OTM IV error). Return the strike plus the realized delta actually selected; strategies log both.
- **P2 `select_expiry(chain, dte_min, dte_max, prefer)`** — `prefer ∈ {monthly, any}`. Monthly-only where the spec says OpEx (OPT-043). Never straddle a distribution of expiries silently — log the chosen expiry.
- **P3 `standard_exit(position, profit_take, dte_exit)`** — close at `profit_take` fraction of credit captured **or** at `dte_exit`, whichever first; evaluated once daily at snapshot. Default (001, 002, 004, 016, 017, 018): `profit_take=0.50`, `dte_exit=21`.
- **P4 `size_by_vega_budget(structure, budget_vega, nav)`** — units such that |net vega| ≈ budget. The **single shared budget constant** across F1/F3/F6 makes the ablation ladders comparable; it is fixed at spec time and never tuned. Alternative sizers (`size_by_debit`, `size_by_notional`) exist for debit structures and are named per candidate.
- **P5 `regime_gate(state, allowed_states)`** — from the point-in-time classifier state (rule 5). Gate is evaluated at entry only unless the spec says otherwise; a spec that unwinds on regime change says so explicitly.
- **P6 `hedge_ledger(position, bars_1m, mode, band, cost_bps_schedule)`** — delta hedging in the underlying. `mode ∈ {daily_1550, band}`. Daily mode hedges at 15:50 using own bars. Band mode triggers on ±band σ moves. **Slippage schedule is tiered by realized-vol percentile** — hedge frequency spikes exactly when underlying spreads widen, and a flat bps assumption flatters every gamma strategy. Emits a ledger of every hedge fill; P&L attribution reads the ledger, not a reconstruction.
- **P7 `roll(position, roll_trigger, new_selector)`** — mechanical rolls (OPT-001/006/038/047). Logs each roll as a separate cost event.
- **P8 `iv_rank(ticker, date, window_years)` / `iv_percentile(...)`** — from `iv_rank_daily`, strictly backward-looking.
- **P9 `yang_zhang_rv(bars_1m, window_days)`** — Yang-Zhang realized-volatility estimator. **Reuse the Tier-1 toolbelt implementation if Phase 0.5 finds one.** Enforces the 15:45 truncation internally.
- **P10 `har_rv_forecast(bars_1m, spec=(1,5,22), horizon)`** — Heterogeneous Autoregressive Realized Volatility forecast on P9 inputs. **Spec frozen at (1,5,22); no order selection, no re-fitting schedule tuning.** Fit on an expanding window with no forward data. Enforces 15:45 truncation. Emits the forecast plus its correlation to contemporaneous VIX (reported alongside every result — see OPT-019's spurious-path check).

---

# 4. Harness modules (build once)

- **M1 — Early assignment & pin.** Short in-the-money calls: assign the session before ex-dividend when remaining extrinsic value < dividend. Short options in-the-money by ≥ $0.01 at expiry: auto-exercise per OCC. Pin handling at short strikes. **Required by every short-call/short-put structure** — roughly two-thirds of the slate. Needs a dividend calendar (ORATS `div_assumption` is a starting point; verify against a second source).
- **M2 — Mark convention.** All marks are the near-close snapshot. **Prohibited:** trade-print-derived option OHLC bars as marks for out-of-the-money or low-volume strikes, and raw session "close" prints. Marking uses `mid`, with `smv_value` as the cross-check; a mid that diverges materially from SMV on a low-delta strike is flagged, not silently used.
- **M3 — Cost model** (§2), with the sensitivity switch.
- **M4 — P&L decomposition.** Theta / vega / delta-residual / gamma attribution. Required by all of F4 (the calendar falsifiers are stated in these terms) and reused by OPT-038/044. Build it once with F4.
- **M5 — Regime-sliced attribution.** Every result reports P&L and Sharpe **per classifier state** in the standard output. This is not optional reporting polish: the OPT-015/039/048 falsifiers are stated per-regime, and aggregate numbers over 2007–2026 can hide a result manufactured entirely by the 2012–2019 short-vol era.
- **M6 — Validation wrapper.** Combinatorial Purged Cross-Validation (CPCV) with **embargo ≥ maximum holding period** of the candidate (45–60 days for most of this slate; longer for OPT-003/006's quarterly and LEAPS legs). Overlapping expiries make naive purging insufficient — the embargo must cover the full life of any position open at a block boundary. Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR) computed with `n_trials` drawn from the **lifetime ledger** (§7), never from the wave.

---

# 5. Wave 0 — Diagnostics (zero trials, run first)

Diagnostics measure **data properties**. They involve no strategy P&L and no selection on returns, so they do **not** increment `n_trials`. Each has a pre-registered gate whose failure drops the dependent candidate *before* it spends a draw from the finite sample. This is the cheapest prior-elevation available and the reason Wave 1 is only nine trials.

**Group A — runnable today, owned data only, $0:**

| ID | Measurement | Data | Gate |
|---|---|---|---|
| D-014/042 | Gap-continuation effect (open gap ≥ ±0.5%, 30-min confirmation → close) and first-hour-trend continuation (≥ ±0.35% → rest-of-day), SPY/QQQ 2016+; **post-2023 sub-sample reported separately** | 1m bars | Proceed toward Rung-2 only if underlying effect ≥ **15 bps/trade** net of a 2 bp slippage haircut **in the post-2023 slice**. Else DROP both |
| D-040a | Weekend realized-variance share: Fri-close→Mon-open variance vs 3 trading days' worth | 1m bars | Feeds D-040b |
| D-013/030 | Drawdown-shape census: conditional on regime-downgrade triggers, classify subsequent declines gap vs grind | 1m bars + VIX term | Descriptive; frames both candidates' valley-of-death falsifier |
| D-050a | Realized-vol timing around classifier transitions: RV in the 5 sessions before vs after each transition | 1m bars + classifier log | If RV peaks **before** transitions, the classifier lags and OPT-050 is dead pre-trial |

**Group B — after the ORATS pull:**

| ID | Measurement | Gate |
|---|---|---|
| D-003 | Financing ratio distribution: (1M 0.20Δ call premium × 3) ÷ (3M 0.15Δ put cost), monthly, 2007+ | Proceed only if ratio ≥ **0.7 in ≥ 60% of months**; else DROP 003 |
| D-005 | Post-pullback put VRP vs unconditional put VRP at 0.30Δ | Proceed only if conditional > unconditional; else DROP 005 |
| D-011 | Entry-day put-IV percentile census at breakdown triggers | Descriptive; if entries systematically land at IV pctile > 80, expect the drift edge to be consumed |
| D-018 | IV-rank state-transition matrix (does IVR>50 persist / lead rising vol?) | Descriptive; frames the conditioning trap |
| D-027 | Skew-percentile state persistence + realized crash frequency conditional on steepness | Descriptive |
| D-029 | Jade-lizard starvation census: frequency the credit>width constraint binds | Proceed only if ≥ **6 qualifying entries/yr** average; else DROP 029 |
| D-033 | Non-overlapping M2/M1 < 0.97 episode count, 2007–2026 | **≥ 12 episodes** → Wave 3 backtest; **< 12** → route to forward paper validation, do not spend the historical sample |
| D-037 | Broken-wing-fly starvation census (no-cost entry constraint) | Same 6/yr rule as D-029 |
| D-040b | Friday vs Thursday term-adjusted ATM IV discount | Proceed only if measured Friday discount < **50%** of calendar-day theta differential; else DROP 040 (expected) |
| D-043 | Pin distance: |spot − max-OI strike| behavior, monthly OpEx week vs control weeks (needs OI) | Proceed only if OpEx-week distance is materially smaller; else DROP 043 |
| D-047/030 | Deep-OTM integrity: sample cross-check of smoothed curvature/low-delta marks against raw NBBO | Integrity gate — if the surface diverges materially at 0.05Δ, both candidates need a different mark source |
| D-048 | IV percentile in compressed-RV states (is IV already at its own floor?) | If IV sits at its floor, sellers are not extrapolating → DROP 048 pre-trial |
| D-049 | Non-overlapping episodes where HAR-RV forecast > ORATS IV, aligned window | **≥ 15 episodes** → Wave 2; fewer → forward paper route |
| D-050b | Entry-day IV percentile at classifier transitions | If mean pctile > **80**, classifier lags the vol → DROP 050 |

---

# 6. Per-candidate specifications

Field key: `structure` / `entry` / `exit` / `sizing` / `gate` / `primitives` / `data` / `cost` / `criteria` (pre-registered GO/NO-GO) / `falsifier` / `notes`. Every candidate contributes **`n_trials = 1`** on execution unless stated. Parameters are FIXED — see §"What this document does NOT do."

**Universal pre-registered criteria** (apply to every strategy backtest unless overridden): report DSR against lifetime N; PBO; regime-sliced attribution (M5); results at 1.0×/±50× cost. A candidate is **not** a survivor if its sign flips inside the cost band, if PBO > 0.5, or if DSR fails at lifetime N. These are necessary conditions, not a GO — capital decisions remain Shuyang's.

## 6.1 Wave 1 — nine trials, authorized after Phase 0 + ORATS ingest

### OPT-015 — Index Delta-Hedged Short Straddle · F1 · GO · Prior A
- `structure`: short 1 ATM call + 1 ATM put, same expiry, delta-hedged in shares.
- `entry`: P2 select expiry at **30 DTE**; ATM = nearest strike to spot at snapshot.
- `exit`: hold to **7 DTE**; no profit-take, no stop (deliberate — the anchor must be unmanaged to serve as the ladder baseline).
- `sizing`: P4, shared vega budget.
- `gate`: P5 — suspend new entries when state = UNPREDICTABLE.
- `primitives`: P1(ATM), P2, P4, P5, P6(`mode=daily_1550`).
- `data`: chains + smoothed greeks; own 1m bars for hedges.
- `cost`: 2-leg combo, monthly; hedge slippage from P6's tiered schedule. Modeled drag ~1–2% of premium annually — PASS with margin. Stressed-width exits are the real risk and must use the stressed tier.
- `criteria`: **the anchor.** Report VRP realized per regime state; the ladder comparisons (018, 019) are only interpretable against this. Explicit sub-criterion: report 2007–2011 / 2012–2019 / 2020–2026 sub-period Sharpes separately.
- `falsifier`: index VRP net of hedge slippage and costs ≤ 0 in the current regime.
- `notes`: hedge marks at 15:50 vs chain snapshot at ~15:46 — a 4-minute basis. Document it; do not "fix" it by moving the hedge to the snapshot minute without registering the change. M1 required (assignment on the short legs).

### OPT-019 — Forecast-Throttled Short Vol · F1 · GO · Prior A− · **flagship 1m candidate**
- `structure`: OPT-015's structure; units scaled by the VRP estimate.
- `entry`: as OPT-015, with `size_multiplier = clip((IV_30 − HAR_forecast) / IV_30, 0, cap)`, **cap = 2× base vega**. Zero size when the spread ≤ 0.
- `exit` / `gate`: as OPT-015.
- `primitives`: P1, P2, P4, P5, P6(daily), **P9, P10**.
- `data`: chains + greeks + own 1m bars. **Signal window is bounded by own-bar history (2016+), not by ORATS' 2007+.** Report the aligned window honestly; do **not** backfill the estimator with daily-bar approximations to extend it.
- `cost`: as OPT-015.
- `criteria`: throttled net Sharpe **>** unthrottled OPT-015 on the aligned window, at matched average vega. Also report throttle-vs-VIX correlation (P10 emits it): if the throttle is ~a VIX-level proxy, say so — the "forecast edge" claim fails even if P&L is positive.
- `falsifier`: throttled ≤ unthrottled on the aligned window.
- `notes`: 15:45 truncation is load-bearing here. Frozen HAR spec — no order selection.

### OPT-016 — Iron Condor, Delta-Managed · F1 · GO · Prior A−
- `structure`: short 0.20Δ strangle + long 0.05Δ wings, one expiry.
- `entry`: P2 **30–45 DTE**; P1 for all four strikes.
- `exit`: P3 (50% / 21 DTE) **or** mechanical close on short-strike touch. **No adjustment, ever** — adjustment rules generate researcher degrees of freedom.
- `sizing`: P4. `gate`: none. `primitives`: P1, P2, P3, P4.
- `data`: chains only (cheapest footprint on the slate — no greeks strictly needed beyond strike selection).
- `cost`: 4-leg combo monthly → ~4–8% of collected premium. PASS.
- `criteria`: compare to OPT-015 at matched vega; the wings' cost must be justified by tail reduction, reported as drawdown/CVaR improvement, not Sharpe alone.
- `falsifier`: long-run P&L < OPT-015 at matched vega **with** worse tail statistics.
- `notes`: **known measurement bias — state it with results.** Touch detection at daily snapshots misses intraday touches, so realized touch frequency is understated and this backtest is optimistic on exactly the dimension that matters. If 016 survives, an intraday-touch re-measurement is a legitimate later use of Rung-2 data.

### OPT-001 — Delta-Targeted Covered Call Overwrite · F7 · GO · Prior A
- `structure`: long 100 shares/unit + short 1 call at **0.25Δ**.
- `entry`: P2 **30–45 DTE**; P1(C, 0.25).
- `exit`: P3 (50% / 21 DTE), then P7 roll.
- `sizing`: fixed share lots per name. `gate`: P5 — **skip overwrite** when state ∈ {STRONG_BULL, UNPREDICTABLE}.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + deltas.
- `cost`: 1 leg, ~12 round trips/yr/name → <1% of premium. PASS.
- `criteria`: total return and risk vs buy-and-hold on the same shares; premium captured must exceed upside forfeited over the full regime cycle.
- `falsifier`: realized vol persistently ≥ implied at 0.25Δ strikes; or forfeited upside > premium captured across the cycle.
- `notes`: **M1 is mandatory** — dividend-driven early assignment on short ITM calls is the classic covered-call backtest error. Universe `U_INDEX` + `U_MEGA20`. Book note: overlaps RAMP's long-equity capital; that is a portfolio question, not a research blocker.

### OPT-002 — Cash-Secured Put Wheel · F7 · GO · Prior A
- `structure`: short 1 cash-secured put at **0.30Δ**; on assignment, hold shares and overwrite per OPT-001 (0.25Δ call) until called away; repeat.
- `entry`: P2 30–45 DTE. `exit`: P3 (50% / 21 DTE).
- `sizing`: one unit per name, no averaging down. `gate`: P5 — suspend new put sales in BEAR.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + deltas.
- `cost`: as OPT-001 plus share transitions. PASS.
- `criteria`: **P&L must be accounted net of the risk-free yield on the secured cash.** The researchable object is premium *in excess of* T-bill yield; an accounting that ignores it flatters the strategy in a positive-rate regime. Report both gross and excess.
- `falsifier`: put-side VRP at 0.30Δ ≤ 0 net of costs and cash yield across a full cycle.
- `notes`: assignment clustering at regime breaks is the risk; the BEAR gate carries the strategy — one registered spec only, no post-hoc with/without comparison.

### OPT-027 — Steep-Skew Put-Spread Harvest · F3 · GO · Prior A−
- `structure`: short 0.25Δ put / long 0.10Δ put, same expiry.
- `entry`: P2 **45 DTE**; only when `skew_25d` > trailing-**2y 70th percentile** (P8-style, backward-looking).
- `exit`: P3 (50% / 21 DTE). `sizing`: P4. `primitives`: P1, P2, P3, P4, P8.
- `data`: chains + `skew_daily`. `cost`: 2-leg index monthly → PASS.
- `criteria`: conditional premium **>** unconditional (same structure, no skew gate) — the gate must earn its place.
- `falsifier`: conditional ≤ unconditional.
- `notes`: D-027 frames this. Skew premium is among the less-decayed premia found in research; the conditioning trap (steep skew = regime label, not mispricing) is the live risk.

### OPT-032 — Contango Calendar Carry · F4 · GO · Prior B+
- `structure`: short 30 DTE ATM straddle / long 60 DTE ATM straddle.
- `entry`: when `term_slope_daily.slope` (M2/M1 ATM IV) > **1.05**.
- `exit`: at front expiry **−7 days**. `sizing`: P4. `primitives`: P1(ATM), P2, P4.
- `data`: chains + greeks, two expiries. `cost`: 4 contract-transits/cycle, index → PASS.
- `criteria`: **requires M4.** Slope-conditional carry must be positive **after** vega mark-to-market — decompose P&L into theta-differential vs vega components and report both. A positive total that is entirely vega is not carry.
- `falsifier`: slope-conditional carry ≤ 0 post-vega-MTM (i.e., slope is compensation, not mispricing).
- `notes`: build M4 here; OPT-038 and the F4 DEFERs reuse it.

### OPT-047 — Rolling Far-OTM Put Ladder · F10 · GO · Prior B+ (as infrastructure)
- `structure`: always-on ladder of three rungs, 3-month **0.05Δ** puts, one new rung per month.
- `entry`: monthly, budget-capped at **40 bps/month** of allocated capital. `exit`/monetization: sell a rung when VIX > **40** or rung delta > **0.30**, then immediately re-strike.
- `sizing`: budget-constrained, not vega-targeted. `primitives`: P1(0.05Δ — **smoothed greeks mandatory**), P2, P7.
- `data`: chains + smoothed low-delta IV. `cost`: ~2.5–4% of premium per rung-roll — small relative to a deliberately budgeted carry cost. PASS.
- `criteria`: **evaluated jointly with the Wave-1 short-vol book, never standalone.** Metric: portfolio Conditional Value-at-Risk improvement per basis point of drag versus the unhedged book. A standalone Sharpe on a tail hedge is a machine for concluding insurance loses money.
- `falsifier`: drag exceeds budget, or monetization triggers never fire across a full cycle containing a crash.
- `notes`: D-047/030 integrity gate applies — verify smoothed 0.05Δ marks against raw NBBO on a sample before trusting them. The compressed-VRP finding cuts *for* this candidate: cheaper insurance regime.

### OPT-050 — Regime-Transition Long Vol · F10 · GO · Prior B · **cheapest build on the slate**
- `structure`: long ATM straddle, **21–30 DTE**.
- `entry`: for **5 sessions** following any classifier transition *into* UNPREDICTABLE or *out of* STRONG_BULL.
- `exit`: at 5 sessions or 10 DTE, whichever first. `sizing`: P4, small fixed vega budget.
- `primitives`: P1(ATM), P2, P4, P5. `data`: chains + classifier state log.
- `cost`: episodic 2-leg index → PASS.
- `criteria`: **gated by D-050b** — if mean entry-day IV percentile at transitions > 80, the classifier lags the vol and this is dead pre-trial.
- `falsifier`: as above; or straddle P&L ≤ 0 across the transition census.
- `notes`: this is substantively a test of the classifier's **lead time**, so both outcomes pay: a failure is diagnostic information about RAMP's regime layer. Depends entirely on Phase 0.8 (persisted PIT state log).

## 6.2 Wave 2 — ten candidates, each gated

**Wave 2 opens only if Wave 1 leaves a live short-vol / carry question.** If OPT-015 and OPT-019 jointly establish that index VRP is dead net of costs in the current regime, the F1/F3/F6 block loses its foundation and Wave 2 is cancelled rather than run (see §8).

### OPT-018 — IV-Rank-Gated Short Strangle · F1 · GO · Prior B+ · ladder step 2
- `structure`: short **0.16Δ** strangle, undefined risk. `entry`: P2 **45 DTE**; only when `iv_rank_1y` > **50**. `exit`: P3 (50%/21 DTE).
- `sizing`: **half** the OPT-015 vega budget (undefined risk). `primitives`: P1, P2, P3, P4, P8. `cost`: 2-leg index → PASS.
- `criteria`: gated premium **>** ungated (OPT-015) at matched vega. Clean pre-registered ablation; must run *after* 015.
- `falsifier`: gated ≤ ungated. `notes`: D-018 frames the conditioning trap. Popular practitioner lore with thin rigorous support — informative either way.

### OPT-036 — Range-Gated Iron Condor · F6 · GO · Prior B+ · **flagship 1m candidate**
- `structure`: OPT-016's structure. `entry`: only when (a) state = SIDEWAYS **and** (b) `yz_10d` < 0.8 × ATM IV_30 **and** (c) **≥ 5 sessions from a scheduled CPI/FOMC date** (registered as part of the spec, closing the "IV correctly prices a known event" channel; macro calendar is free).
- `exit`/`sizing`: as OPT-016. `primitives`: P1–P5, **P9**. `data`: chains + own 1m bars + macro calendar.
- `criteria`: gated **>** ungated OPT-016 on the aligned (2016+) window. `falsifier`: gated ≤ ungated.
- `notes`: 15:45 truncation load-bearing. Reuse the Tier-1 Yang-Zhang implementation if it exists.

### OPT-005 — Laddered CSP Pullback Grid · F7 · GO · Prior B · gated by D-005
- `structure`: 3-rung put ladder at **0.30 / 0.20 / 0.10Δ**, 21–35 DTE. `entry`: sell the next rung only after a ≥ **2%** underlying pullback (daily close basis from 1m bars); unwind the ladder in BEAR.
- `exit`: P3. `sizing`: one unit per rung. `primitives`: P1, P2, P3, P5. `cost`: index widths → PASS.
- `criteria`: post-trigger put VRP > unconditional (that is D-005; if the diagnostic fails, this never runs).
- `falsifier`: the trigger is a regime-transition detector, not a mean-reversion signal — visible as ladder deployment into 2008/2020/2022-style declines.

### OPT-006 — Poor Man's Covered Call (LEAPS diagonal) · F7 · GO · Prior B · cost-sensitivity mandatory
- `structure`: long **12–18M 0.80Δ** call + short **30–45 DTE 0.25Δ** call. `entry`/`exit`: short leg per OPT-001; long leg rolled at 6M remaining (P7). `gate`: P5 as OPT-001.
- `primitives`: P1, P2, P3, P5, P7. `data`: chains + greeks — **ORATS smoothed, not ThetaData** (deep-ITM IV error + dividend handling).
- `cost`: LEAPS round trip ~1.5–3% of option value + monthly short legs → annualized drag ~3–5% worst case. **MARGINAL-PASS — the ±50% sensitivity run is the actual test.**
- `criteria`: net carry **>** OPT-001 on matched notional. `falsifier`: it is not.
- `notes`: vega is the bill for the capital efficiency — a vol crush marks the long leg down exactly when the overwrite is working. Must run alongside OPT-001 (its benchmark).

### OPT-013 — Put Ratio Backspread on Deterioration · F5→F10 · GO · Prior B
- `structure`: short 1 ATM put + long 2 **−1σ** puts, ~zero cost, **60 DTE**. `entry`: on regime downgrade (WEAK_BULL→SIDEWAYS→BEAR transitions) **with** VIX term flattening. `exit`: 21 DTE or regime recovery.
- `sizing`: zero-cost constraint. `primitives`: P1, P2, P5. `data`: chains + greeks + free VIX term structure.
- `criteria`: positive expectancy conditional on the trigger, with the D-013/030 shape census reported alongside.
- `falsifier`: declines are predominantly slow grinds settling in the valley between strikes (the live 2022 pattern).

### OPT-021 — Earnings Vol-Crush Short Iron Fly · F2 · CONDITIONAL · **spec v1.1**
- **Registered amendment (made before any data contact; v1.0 retired, not overwritten):** exit moves from T+1 **open** to T+1 **near-close snapshot**. The open mark is structurally unreliable at Rung 1 (widest spreads, sparsest prints). v1.1 therefore carries day-1 post-earnings drift exposure v1.0 did not — **it is a different strategy, not a proxy.** The T+1-open version waits for Rung-2 quote data, if ever.
- `structure`: short ATM straddle + wings at the **priced expected move**. Defined risk always.
- `entry`: T−1 near-close, nearest expiry after the print; universe = liquid names with weeklies and priced expected move ≥ **4%**. `exit`: T+1 near-close.
- `sizing`: fixed risk per event. `primitives`: P1, P2. `data`: chains + Rung-3 earnings dates.
- `cost`: 4 legs × 2 transits, single-name, **stressed tier mandatory** → 10–25% of collected premium. MARGINAL; sensitivity run required.
- `criteria`: **universe filter must be ex-ante** (liquidity, market cap, expected-move threshold). Selecting names by historical beat-the-implied-move rate imports the outcome into the universe and is prohibited.
- `falsifier`: realized moves ≥ implied on average, net of stressed-tier spreads.
- `notes`: research found beat-rates ranging ~25–63% **across names** — the premium is name-dependent and crowded, and published beat-rate statistics are themselves in-sample folklore. Earnings dates are a leakage item (rule 4). Requires M1.

### OPT-044 — Dispersion-Lite · F8 · GO · Prior B+ · cost-sensitivity mandatory
- `structure`: short SPY ATM straddle vs long weight-proportional ATM straddles on `U_TOP6_SPY`, **vega-neutral** at entry.
- `entry`: P2 **30–45 DTE**. `exit`: 21 DTE. `hedging`: P6 `mode=daily_1550` but executed **weekly** per spec. `primitives`: P1, P2, P4, P6.
- `data`: chains + greeks across 7 roots (single ORATS dataset).
- `cost`: 14 legs/cycle; 6 tier-1 single-name legs dominate → ~10–20% of premium capture. **MARGINAL-PASS.**
- `criteria`: **report the decomposition** — index-leg P&L vs component-leg P&L separately. With six names this is substantially a bet on six idiosyncratic vol processes plus index concentration; the "dispersion" label must not launder the exposure.
- `falsifier`: implied-realized correlation gap ≤ costs of the 14-leg structure.
- `notes`: research found this among the **least-decayed** premia (long-run gap ≈ 6.9 correlation points) — but correlation → 1 on crash days; the April 2025 near-total-co-movement session is the shape of the tail. Report worst-day P&L prominently.

### OPT-048 — Vol-Compression Breakout Long Straddle · F10 · GO · Prior B · gated by D-048
- `structure`: long ATM straddle, **30–45 DTE**. `entry`: when `yz_10d` < **20th** trailing-2y percentile **and** 20-day price range < **15th** percentile. `exit`: **+75% / −40%** of debit, or 10 DTE.
- `sizing`: `size_by_debit`. `primitives`: P1, P2, P9. `cost`: 2-leg index, episodic → PASS.
- `criteria`: gated by D-048 — if IV in compressed states already sits at its own percentile floor, sellers are not extrapolating and the thesis dies pre-trial.
- `falsifier`: as above; or long-vol bleed exceeds expansion capture (2017-style persistence).
- `notes`: M5 regime slicing mandatory — a long calm regime is the falsifier's natural habitat.

### OPT-049 — Gamma Scalping when Forecast RV > IV · F10 · CONDITIONAL · gated by D-049 (≥15 episodes)
- `structure`: long **30 DTE** ATM straddle, band-hedged. `entry`: when P10's HAR-RV forecast > ORATS IV_30 (sign flip of OPT-019's estimator). `exit`: on sign flip back or 10 DTE.
- `hedging`: P6 `mode=band`, band = **±0.25σ** underlying moves, off 1m bars. `primitives`: P1, P2, P9, P10, P6(band).
- `data`: **best structural fit on the slate** — hedge P&L accrues from own 1m bars via the hedge ledger; option marks needed only at entry/exit/terminal, so Rung 1 suffices.
- `cost`: share hedges cheap, **but** hedge frequency spikes exactly when underlying spreads widen — the tiered slippage schedule in P6 is what makes this honest. A flat bps assumption flatters it materially.
- `criteria`: hedge-ledger P&L > theta paid, across the episode census. `falsifier`: it is not; or D-049 finds < 15 episodes → forward-paper route, do not spend the historical sample.
- `notes`: shares P9/P10 machinery with OPT-019 — one build, two candidates.

### OPT-003 — Financed Collar Carry · F7 · CONDITIONAL · gated by D-003 (financing ratio)
- `structure`: long SPY + long 3M **0.15Δ** put + short monthly **0.20Δ** calls. `entry`/`roll`: static calendar, no discretionary adjustment. `primitives`: P1, P2, P7.
- `cost`: ~2 legs/mo + quarterly put roll → PASS. `criteria`: gated by D-003 (ratio ≥ 0.7 in ≥ 60% of months).
- `falsifier`: collar drag > unhedged drawdown improvement across a bear regime. `notes`: research reinforced drafting-time skepticism — financing ratios frequently fail. Expect D-003 to be the end of this candidate.

## 6.3 Wave 3 — twelve candidates (build machinery now, run only on gates)

- **OPT-011 — Bear Put Debit Spread on Breakdown** · F5 · GO · B. Short: mirror of OPT-007 — buy ATM put / sell −1σ put, 45–60 DTE, on state ∈ {BEAR} + 20-day-low breakdown; exits at 21 DTE or +100%/−50% of debit; universe = bottom-decile momentum, max 10. **Design constraint: the signal layer must NOT reuse RAMP's selection machinery** — RAMP's BEAR failure was long-side *selection*, and importing it re-imports the failure. `criteria`: entry-IV richness (D-011) must not consume the drift edge. Genuine diversifier vs the F7/F1 book.
- **OPT-020 — Single-Name VRP Basket** · F1 · CONDITIONAL (parent 015 + earnings-date QA). OPT-015 replicated per name on `U_MEGA10`, equal vega, earnings windows excluded. `cost`: MARGINAL (8–15% of premium on wider single-name spreads). `criteria`: basket VRP net of spreads **>** index VRP, else OPT-015 dominates and this dies. **QA condition:** validate exclusion dates against a second source on a sample; a single mis-dated print inside an "excluded" window contaminates the ambient-premium claim.
- **OPT-028 — Skew-Extreme Risk-Reversal RV** · F3 · GO · B. 60 DTE; skew > 90th pctile → sell put/buy call at 0.25Δ; < 10th → reverse; P6 daily hedge. `criteria`: **verify realized residual delta stays small** — otherwise the result is a directional trade mislabeled as a skew trade. Needs the hedge-ledger module from Wave 1.
- **OPT-030 — Crash-Zone Broken-Wing Put Butterfly** · F3 · GO · B. 60–90 DTE; long 1×−1σ, short 2×−1.5σ, long 1×−2.5σ, ~zero cost, rolled monthly. **Smoothed low-delta greeks mandatory (D-047/030 integrity gate).** `cost`: MARGINAL — 4 OTM legs, exit friction in stressed tapes is the bill. `falsifier`: terminal distributions concentrate at −1.5σ (2022-style grind) rather than bimodal.
- **OPT-033 — Backwardation Reverse Calendar** · F4 · CONDITIONAL · gated by D-033 (≥12 episodes). Long 30 DTE / short 60 DTE ATM straddles when slope < **0.97**; exit on re-normalization or front −7d. If < 12 episodes → **forward paper**, per the saturation rule.
- **OPT-038 — Double Diagonal Income** · F6 · GO · B−. Short 30 DTE 0.25Δ strangle / long 60 DTE 0.15Δ strangle; short legs rolled monthly, long legs re-struck quarterly. Reuses M4. `criteria`: must beat OPT-016 at matched risk.
- **OPT-039 — Band-Recentered Short Straddle** · F6 · GO · B−. 30 DTE short ATM straddle; recenter when the underlying exits a **±0.75σ** band on a **daily close** basis (deliberately not intraday — keeps it at Rung 1); **max 1 recenter**, then hard stop. `criteria`: must beat OPT-015 **within SIDEWAYS-classified periods specifically** — M5 delivers this directly. The recenter cap is the discipline separating this from martingale adjustment folklore.
- **OPT-043 — Expiration-Week Pin/Charm Short Straddle** · F9 · CONDITIONAL · gated by D-043. Monthly OpEx only (weekly pinning is diluted); short straddle at the **T−1** max-OI strike Wednesday, exit Friday 15:00. **OI is next-morning data — same-session OI is a hard leak.** Requires the OI pull.
- **OPT-041 — 0DTE Post-Opening-Range Iron Condor** · F9 · CONDITIONAL · **requires Rung 2; never justifies the purchase alone.** Enter 10:30 after the opening range, wings beyond ±1× opening range, hold to 15:55 or short-strike touch. `cost`: worst profile on the slate — 8 contract-transactions/day, modeled drag ~10–15% of daily credit, every day. **MARGINAL-to-FAIL.** Explicitly a **research-to-disprove** candidate: the valuable output is a clean measurement of how dead intraday VRP is after costs. Any positive result on a 2020–2026 window is suspect of being three good years with no 2018-style vol event in sample. Short-history significance penalty applies (§7).
- **OPT-007 / OPT-008 / OPT-009 — the long-momentum trio** · F5 · CONDITIONAL · **gated on the PR6 plain-momentum re-baseline** (already RAMP's registered Phase-4C go/no-go). If PR6 fails at matched cadence, all three DROP together with zero trials spent. Specs: 007 = bull call debit spread, ATM/+1σ, 45–60 DTE, top-decile 12-1 momentum + 20-day-high breakout, max 10 positions, exit 21 DTE or +100%/−50%; 008 = ZEBRA, buy 2× 0.70Δ calls / sell 1 ATM call, 60–90 DTE, exit on regime downgrade; 009 = risk reversal, sell 0.25Δ put / buy 0.25Δ call, 45 DTE, max 5 names. `falsifier` (shared): spread P&L < delta-equivalent stock P&L on matched signals — i.e. the wrapper only added cost. 009 is the undefined-risk member and the first to cut if only some survive.

## 6.4 Wave 4 / shelf — eight candidates (specified, not queued)

Each has a named unlock; **none justifies spend on its own.**

- **OPT-014 — Opening-Range Gap Same-Day Verticals** · unlock: D-014/042 post-2023 slice ≥ 15 bps **then** Rung 2. Enter ~10:00 on gap > ±0.5% with 30-min confirmation, 1-wide debit vertical, hard stop 15:45. Quote bars only.
- **OPT-042 — 0DTE First-Hour-Trend Debit Spread** · unlock: same shared diagnostic. First-hour return ≥ |0.35%| → 1-wide vertical at 10:30, exit 15:45. Expected per research: post-2023 decay shows in the slice → DROP.
- **OPT-040 — Weekend Theta Capture** · unlock: D-040b (Friday discount < 50% of calendar-day theta differential). **Expected DROP** — research indicates Friday IV generally pre-discounts the weekend. Note the Monday 09:45 exit is not Rung-1-compatible; a Monday near-close amendment would dilute the thesis and must be registered if chosen.
- **OPT-024 — Post-Crush Overshoot Short Strangle** · unlock: parents 020 ∧ 021 both clear. T+1 close entry, 21–30 DTE 0.16Δ strangle. `falsifier`: post-print VRP ≤ ambient single-name VRP (then 020 dominates) — uninterpretable before 020 exists.
- **OPT-025 — Macro-Event Index Straddle** · **registered amendment:** exit T+0 **near-close** (CPI 08:30 / FOMC 14:00 both resolve intraday; near-close is well-defined at Rung 1). Demoted to null-anchor — research reiterates implied event moves ≥ realized on average. Regime-conditional slicing of the result is reporting, not a new trial.
- **OPT-026 — Sector-Sympathy IV Fade** · **research-to-disprove.** The falsifier is currently live: 2023–26 AI-capex tapes show peer moves on leader prints are frequently real information transfer, not sympathy overpricing. A confirming result would be surprising and must be checked hardest.
- **OPT-031 — Inverted-Call-Skew Fade** · unlock: **live IBKR fill data** establishing achievable spreads in squeeze names. The mechanism (lottery-preference overpricing of OTM single-name calls) is documented; the cost model, not the mechanism, is the blocker — modeled drag frequently exceeds the 0.15Δ spread's premium.
- **OPT-045 — Same-Sector IV-Percentile Pairs** · unlock: OPT-044 clears. 80/50 IV-percentile gates, delta-hedged both legs, no leg within 10 days of either print. If the clean index-vs-component version of the correlation premium fails, the noisier pairwise version is not the rescue.

## 6.5 DEFER — seven candidates (sequential testing under pre-registered rules)

These were flagged as near-duplicates in slate v1 and are **not** independent hypotheses. Each runs **only** if its parent(s) clear validation; on execution each increments `n_trials` normally. The rules are fixed now, before any result — that is what makes this legitimate sequential testing rather than results-conditioned selection.

| ID | Parent(s) | Marginal question it answers | Extra gate |
|---|---|---|---|
| OPT-004 Covered Strangle | 001 ∧ 002 | Does the joint structure with its own regime kill-rules beat the sum of the parents? | — |
| OPT-017 Iron Butterfly | 016 | Profit-zone width vs touch frequency at max theta | — |
| OPT-029 Jade Lizard | 002 ∧ 027 | Does the credit>width constraint create a genuinely better skew+carry blend? | D-029 starvation (≥6 entries/yr) |
| OPT-034 Slope-Switched Program | 032 ∧ 033 | Is term slope a tradable factor rather than two episodic trades? | Defers with 033 if 033 routes to paper |
| OPT-035 Double Calendar | 032 | Do extra legs buy real path tolerance, or just cost? (strictly cost-worse: 8 transits/cycle) | — |
| OPT-037 Directional-Tilt BWB | 016/017 | Does the drift tilt add anything vs a symmetric fly? | D-037 starvation |
| OPT-046 Index-vs-Single-Name VRP Switch | 015 ∧ 020 | Does the hysteresis-banded switch beat static 50/50? | Dies automatically if either parent dies |

---

# 7. Ledger & DSR protocol

## 7.1 Schema

```
strategy_trial_ledger:
  id                     str      # OPT-0XX
  spec_version           str      # e.g. "v1", "v1.1" (021, 025 are amended)
  mechanism_family       str      # F1..F10
  emitted_at             date     # 2026-07-24 for all 50
  status                 enum     # emitted | queued | diagnostic_only | tested | deferred | dropped | no_go | paper_forward
  wave                   int
  data_window_start      date     # actual window the trial consumed
  data_window_end        date
  data_rung              str      # rung_1 | rung_2a | ...
  n_trials_contribution  int      # 0 until tested; 1 on execution
  tested_at              date
  parent_ids             list     # DEFER lineage
  gate_ids               list     # D-0XX diagnostics that gated it
  amendment_log          list     # registered spec changes with timestamps and rationale
  notes                  str
```

**Rules:** (1) `emitted ≠ tested`. All 50 are emitted today; DSR's N counts trials **evaluated against the overlapping data window**, so generated-but-never-tested candidates are not draws from the sample. (2) `n_trials_contribution` flips 0→1 **on test execution**, not on queueing. (3) Diagnostics (`diagnostic_only`) contribute **0** — they measure data properties, not strategy returns, and select nothing on P&L. (4) Any later parameter sweep on a candidate multiplies its contribution by the sweep cardinality and is re-appended. (5) Amendments are logged before the affected test runs; the superseded version is retired with its own row, never overwritten in place.

## 7.2 Null best-of-N arithmetic (Bailey–López de Prado)

σ of an annualized Sharpe estimate ≈ 1/√years. On the ORATS window (19.5 y), σ_SR ≈ **0.23**; expected maximum in-sample Sharpe under zero edge:

| Stage | N (this slate) | E[max Sharpe | null] |
|---|---|---|
| Wave 1 | 9 | ≈ **0.34** |
| + Wave 2 | ≈ 19 | ≈ **0.42** |
| + Waves 3–4 | ≈ 31–38 | ≈ **0.47–0.49** |

On a **6.5-year** intraday window (ThetaData Value), σ_SR ≈ **0.39** — so **N = 5 intraday trials alone put the null max at ≈ 0.47**, matching what ~30 EOD trials cost on the long window. This is the quantitative statement of why intraday candidates rank last, and it should be quoted in any conversation about buying Rung 2.

**Lifetime reconciliation (CC task, Phase 4):** true N adds Homeguard's prior ledger history (RAMP Phase-3/4 variants and everything else) **against overlapping windows** — the equities-daily trials overlap the 2015+ portion of the options window, and honest DSR counts shared-window trials, not calendar-disjoint ones. Illustrative only: lifetime N = 150 on the long window pushes the null max to ≈ 0.60. **The real count comes from the repo ledger, not from this document and not from memory.**

**This slate's delta:** +50 emitted; ≤ 9 tested at Wave 1; 1 NO-GO and 3 DROP recorded as emitted-not-tested.

---

# 8. Kill switches and the reiterate decision

Registered now, before results exist:

1. **Wave-1 anchor failure.** If OPT-015 and OPT-019 jointly establish that index VRP is dead net of costs in the current regime, **cancel Wave 2's short-vol block** (018, 036, 005, 003, and by extension much of F3/F6). Do not generate more short-vol variants — that is searching, not researching. Reiteration means *new mechanism families* (financing/rates-adjacent structures, cross-asset vol) or new data.
2. **PR6 gate failure.** 007/008/009 drop silently, zero trials spent. Already wired.
3. **Diagnostic failures** (D-003, D-005, D-029, D-033, D-037, D-040b, D-043, D-048, D-049, D-050b) drop or reroute their dependents pre-trial. This is the cheapest budget protection in the chain.
4. **Deep-OTM integrity failure** (D-047/030): if smoothed marks diverge materially from raw NBBO at 0.05Δ, both 047 and 030 need a different mark source before running — a data problem, not a strategy verdict.
5. **Zero survivors after CPCV/PBO/DSR.** A valid, pre-registered outcome. It does **not** trigger re-running anything with adjusted parameters. STOP remains Shuyang's call, not the analysis's.

**Prohibited responses to disappointing results** (stated explicitly because they are the natural next move and they are the overfit): re-running a candidate with adjusted parameters; adding "just one more" variant of a failed mechanism; loosening a pre-registered criterion after seeing the number; reclassifying a tested candidate as a diagnostic to avoid the trial count; extending a window to capture a better period.

---

# 9. CC implementation phases

**Phase 0 — Repo reconciliation.** §0 table. Blocking. Output: written report with paths, line numbers, SHAs. **Do not proceed without review.**

**Phase 1 — Data layer.** Purchase and ingest ORATS near-EOD (Wave-1 universe only: SPY, QQQ, `U_MEGA20`). Build the canonical schema (§1.2), partitioning, DuckDB access, and the derived tables. Verify: corporate-action handling on a known split; snapshot-time consistency; smoothed-vs-raw agreement at low delta (this doubles as D-047/030). Implement the 15:45 truncation guard and the PIT rules (§1.3) **centrally** — not in strategies. Output: ingest spec + data-quality report.

**Phase 2 — Primitives and harness.** P1–P10, M1–M6, `cost_model_v1`. Unit-test M1 (early assignment) against hand-worked dividend cases and M2 (marks) against known wide-spread days. **The hedge ledger's tiered slippage schedule must be built from own Alpaca/IBKR execution data, not assumed.**

**Phase 3 — Wave 0 diagnostics.** Group A today on owned data; Group B after Phase 1. Report every gate outcome explicitly, including the ones that drop candidates — a diagnostic that kills a candidate is the highest-value output in this chain, not a failure.

**Phase 4 — Ledger + validation wrapper.** Ledger per §7 at the location Phase 0.3 identifies; reconcile lifetime N against the repo's prior history. Wire CPCV embargo ≥ max holding period, PBO, DSR.

**Phase 5 — Wave 1 execution.** Nine candidates, in the §6.1 order (015 and 019 first — they anchor everything). Standard output per candidate: regime-sliced attribution, cost sensitivity at 1.0×/±50%, DSR at lifetime N, PBO, and the candidate's own pre-registered criterion evaluated as stated. **Stop at Wave 1.** Later waves require their gates and a review.

**Reporting discipline throughout:** report what the code does, not what the spec says it should do; if an implementation diverges from this document for any reason, surface the divergence in the results header rather than in a footnote. If a parameter proves unimplementable as written, stop and report — do not substitute.

---

# Appendix A — Candidate index by verdict

**GO (21):** 001, 002, 005, 006, 011, 013, 015, 016, 018, 019, 027, 028, 030, 032, 036, 038, 039, 044, 047, 048, 050
**CONDITIONAL (18):** 003, 007, 008, 009, 014, 020, 021, 024, 025, 026, 031, 033, 040, 041, 042, 043, 045, 049
**DEFER (7):** 004, 017, 029, 034, 035, 037, 046
**Specified here: 46.** DROP (3): 010, 012, 022 — Appendix C. NO-GO (1): 023 — excluded.

# Appendix B — Data-requirement matrix (condensed)

| Requirement | Candidates |
|---|---|
| Rung 1 only (chains + smoothed greeks) | 001, 002, 003, 005, 006, 011, 013, 015, 016, 017, 018, 027, 028, 029, 030, 032, 033, 034, 035, 037, 038, 039, 044, 046, 047, 048, 050, 004 |
| Rung 1 + own 1m bars (signal/hedge critical) | 005, 013, 019, 036, 039, 048, 049 |
| Rung 1 + free calendars (VIX term / macro) | 013, 025, 036 |
| Rung 1 + classifier state log | 001, 002, 005, 011, 013, 015, 036, 050 (+ all P5 gates) |
| Rung 2 required (NBBO quote bars) | 014, 041, 042 (+ 040's Monday-open exit, + 021's T+1-open v1.0) |
| Rung 3 required (earnings dates) | 020, 021, 024, 026, 045 |
| Open-interest history | 043 |
| Live IBKR fill data | 031 |
| External workstream gate (PR6) | 007, 008, 009 |

# Appendix C — DROP record (do not resurrect without a new mechanism argument)

- **OPT-010 Call Ratio Backspread on Breakout** — no evidence breakout right-tails are fat enough to pay the valley-of-death; the edge claim was payoff-shape cleverness, not mechanism.
- **OPT-012 Bearish Risk Reversal** — dominated ex ante by OPT-011's defined-risk form; bear-market rallies (the most violent) are precisely this structure's short leg.
- **OPT-022 Pre-Earnings IV Run-Up (long)** — run-up exists, capture net of theta and single-name spreads is unreliable; bounding the family is not worth a draw from the finite sample.

Ledger status for all three: `dropped`, `n_trials_contribution = 0`. If a genuinely new mechanism argument appears, that is a **new** candidate with a new ID, not a resurrection.

---

# Honesty block

Nothing in this chain has observed a backtest return. Every verdict, wave assignment, and prior tier is a feasibility-and-mechanism judgment made blind. "GO" means *worth one draw from a finite sample*, nothing more, and it is not a prediction.

The **cost model is the weakest quantitative input** in the entire chain: vendor-cited fill conventions plus width assumptions, mandatory ±50% sensitivity, to be replaced by own IBKR fill statistics the moment any options strategy trades live — at which point every prior result is re-run. Vendor facts (ORATS ~$399/2007+, ThetaData tiering, greeks availability) were verified in the research pass but partly through secondary comparisons; re-verify at checkout. Decay evidence is mixed in quality: the correlation-premium, index-performance, and 0DTE-share figures trace to named studies and index providers, while VRP-compression and weekend-effect readings lean on practitioner sources — where evidence was thin (026, 040), verdicts already discount in the conservative direction.

Two specs (021, 025) are **amended** versions of their v1.0 forms, registered before any data contact and logged as different strategies. Five integration points remain **unverified against the repo** and are the reason Phase 0 blocks everything.

Zero survivors after validation is a legitimate, pre-registered outcome of this entire slate.

*End of CC handoff spec v1. This document, plus slate v1 and the feasibility screen, is the pre-registration of record. Deviations require a logged amendment before the affected test runs.*
