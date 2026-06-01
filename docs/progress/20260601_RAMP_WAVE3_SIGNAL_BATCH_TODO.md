# Strategy Pipeline TODO -- RAMP Equity-Momentum (Wave-3 Signal Construction)

> Status: `[ ]` pending - `[~]` in progress - `[x]` done - `[!]` failed - `[-]` skipped
>
> Run: `claude --agent strategy-lead`
> Resume: `claude --agent strategy-lead --continue`
>
> Orchestrator: read this file FIRST on every session start.
> Mark `[~]` BEFORE starting a phase. Mark `[x]` AFTER verifying output exists.

---

## Active blockers (carried forward from the shelved options pipeline -- still live)

**Section 11.5 stop-slippage multiplier wiring (as of 2026-05-13).**
`portfolio_simulator.py` and its numba kernel apply uniform slippage to all fills
regardless of exit reason; `CostsSettings.stop_slippage_multiplier` is defined but
inert. Any variant using a stop exit (`fixed_pct_stop`, `vol_scaled_stop`,
`trailing_stop`, `time_stop_with_pct_stop`, `scale_out`) CANNOT graduate to
Phase 9 live until the wiring PR lands -- it would promote on optimistic metrics
(1.5x-3.0x stop-slippage reality). Most Wave-3 variants below are signal/selection
changes with NO stops, so they are unaffected for backtest readiness; the gate
binds only at live promotion and only for any stop-based exit. Tracked at
methodology Section 11.5 ("WIRING IN FLIGHT").

## Previous pipeline (shelved 2026-04-02, NOT discarded)

The RAMP **options-overlay** pipeline (31 candidates, 16 tested, verdict: no
options strategy deployment-ready -- the overlay destroys the momentum edge) is
archived (committed) at `docs/progress/20260402_RAMP_OPTIONS_PIPELINE_ARCHIVE.md`.
Its structural findings and per-strategy phase template remain a reference.

---

# RAMP Wave-3 Signal-Construction Batch

**Created:** 2026-06-01
**Branch (work-from):** `archive/regime-detector-campaign-2026-05`
**Research package:** `src/research/ramp_phase4/` (a consolidation rename to
`regime_momentum_lab` is in flight on another branch -- reconcile before merging)
**Incumbent to beat:** V11 (Sharpe 0.528 full-window, EXT-OOS +0.527, turnover 39%,
passes 7.5 bps cost gate; paper-deployed 2026-05-23)

---

## Why this batch exists (the lens)

The 2026-05-24 regime-detector campaign closed the **regime-TIMING** line with a
definitive negative: the detector fires ~3.4 days after the SPY trough, the
consumer-layer ceiling is **+0.08 Sharpe over V11**, and every BEAR-response
variant (V12/V12c/V13/V14a/b/c) landed TIER 3/4. See
`docs/progress/20260524_RAMP_REGIME_DETECTOR_CAMPAIGN_CLOSURE.md`.

But the root-cause investigation (`docs/reports/ramp/20260505_root_cause_investigation.md`)
pointed somewhere the campaign never tested:

- **H2 (dominant):** regime gating *actively harms* -- vanilla momentum beat
  production RAMP in EXT-OOS (0.314 vs 0.070).
- **H5 (refuted):** the momentum factor itself is intact.
- **H6/H8 (supported):** in BEAR, RAMP picks the *wrong stocks* -- high-beta
  lagged winners (SMCI/ENPH/MU) that crater; 48% of BEAR selections had negative
  next-day returns.

This batch attacks **signal construction and stock selection** -- a different,
never-tested mechanism not bounded by the +0.08 timing ceiling. It is option
value, not a known edge.

---

## Acceptance bar (read before running anything)

1. **Beat V11, not V01.** V11 is the incumbent. The relevant deltas are vs V11's
   0.528 full / +0.527 EXT-OOS at 5 bps.
2. **DSR multi-trial discipline.** Run the batch as ONE readiness gate with a
   shared trial budget so the DSR penalty is counted honestly. The v0 family is
   at n_trials=36; a clean trial-chain reset for a *signal-construction* family
   is justifiable (new signal math, not a detector tweak) -- document the
   justification in the gate output. The +0.10 TIER 1 lift bar is tight.
3. **Net, not gross.** Optimize EXT-OOS net Sharpe at 5 bps; verify survival at
   7.5 bps (1.5x cost-sensitivity gate, methodology Section 4).
4. **Both timing modes.** A candidate must not disappear under `one_day_lag`.
5. **Honesty discipline.** Sensitivity panels are informational and do NOT update
   defaults (V14 spec rev2 rule).

Authoritative methodology: `docs/methodology/backtesting.md` (Sections 1-5, 9, 12).

---

## Gate 0: shared prep -- data + run-durability (do once)

G0.1-G0.4 unblock the data-gated variants (V30, V33); the 9 close-only variants
need none of them. **G0.5 is run-durability infra that applies to the WHOLE gate
-- do it before any long run.**

- [ ] **G0.1 Validate the parquet read defect.** Full-table reads of the daily
  cache and the 1-min SIP files throw `OSError: Repetition level histogram size
  mismatch`. Column-subset and footer reads work. Confirm a column-subset read of
  `volume` succeeds; if not, resolve (likely a pyarrow up/downgrade). BLOCKING for
  V33 volume work.
  - Files: `H:/Stock_Data/cache/ramp_phase4/equities_daily_from_sip.parquet`,
    `H:/Stock_Data/equities/sip_split/1min/symbol=<SYM>/year=<Y>/month=<M>/data.parquet`
- [ ] **G0.2 Fix the stale SIP path constant.** `src/research/ramp_phase4/data.py`
  has `SIP_SPLIT_REL = 'equities_1min_sip_split'` (old top-level path). The tree
  moved to `equities/sip_split/1min/`. Reuse of the existing close-only cache
  masks this today, but any rebuild or new-column pull needs the correct path.
- [ ] **G0.3 Plumb a volume panel (for V33).** Extend `data.py` to aggregate
  `volume` from `sip_split` (split-adjusted -> dollar volume is continuous and
  correct) into a parallel volume cache. Schema confirmed: 8-col OHLCV with
  `volume`, `vwap`, `trade_count`. 12,345 symbols, all complete, ~2016->2026.
- [ ] **G0.4 Fetch + cache sector map (for V30).** Use
  `src/data/yfinance/fundamentals.py` to pull `sector` for the sp500-2025
  universe (~503 symbols) into the persistent `fundamentals_cache.parquet`.
  LIMITATION to document: sector is a current snapshot, not point-in-time
  (same class as the survivorship caveat already in every Phase-4 report).

- [ ] **G0.5 Per-backtest durability: wire the experiment registry into the
  readiness orchestrator** *(run-durability -- applies to ALL variants, not data prep)*
  - **Audit finding (2026-06-01):** the readiness orchestrators are
    all-or-nothing -- v11/v12/v14 run 12-30 backtests in memory and write ONE
    report at the very end (`ramp_phase4_v11_readiness.py:477`, `v12:702`,
    `v14:848`); a crash at backtest 12/18 loses everything, no resume.
    `append_run` is called from `src/backtest_runner.py:664` ONLY -- the research
    orchestrators bypass it, so `output/experiments.duckdb` holds 4 stale rows
    (last write 2026-05-13) despite the entire V01->V14 + detector campaign.
  - **Fix (one change closes three gaps: checkpoint + resume + Section 9.3 mandate):**
    - [ ] After EACH sub-backtest in the gate, call `src.experiments.append_run(...)`
      with the full metrics row + return stream (methodology Section 9.3) -- this
      IS the per-step disk write.
    - [ ] Before running a sub-backtest, query the registry for a matching prior
      run (key on variant_id + timing + cost_bps + git SHA + data-snapshot date);
      if present, SKIP and reuse -- makes the gate resumable.
    - [ ] Where the gate uses `grid_search`, pass `on_trial_complete`
      (`make_trial_callback`) so per-trial rows persist too.
    - [ ] Atomic-write any non-registry artifact the orchestrator emits
      (write-to-`.tmp` + `os.replace`), matching `src/data/acquisition/base.py:161`.
  - **Not a hard blocker:** close-only variants CAN run without G0.5, but the
    gate would have NO step-level durability and would not populate the registry.
    Mandatory before any multi-hour optimizer run.

---

## The variants (tiered by evidence-alignment x readiness)

### Tier 1 -- ready now (close-only), highest evidence-alignment

- [ ] **V31 -- Beta-residual momentum** *(highest evidence-alignment)*
  - **Mechanism:** rank residual returns after removing trailing SPY beta
    (estimate beta over 60-126d; rank residual 21d return).
  - **Attacks:** H6/H8 directly -- the BEAR losers were high-beta names that only
    looked strong on market beta. Residualizing removes exactly those.
  - **Data:** close + SPY (in panel). READY.
  - [ ] Implement plan_fn in `variants.py`; register in `REGISTRY`
  - [ ] TDD: `tests/research/ramp_phase4/test_variants.py`
  - [ ] Run gate; record metrics row

- [ ] **V28 -- Multi-horizon momentum ensemble** *(highest base-rate)*
  - **Mechanism:** blend 21d/63d/126d relative returns with fixed weights
    (0.5/0.3/0.2) + small 5d reversal penalty. No grid search.
  - **Attacks:** signal instability (the H2 root). A more stable signal may
    reduce the need for regime gating entirely.
  - **Data:** close + >=126d lookback (cache spans 2017->2026). READY.
  - [ ] Implement / register / TDD / run / record

- [ ] **V26 -- Z-score normalized score** *(cheap fragility fix)*
  - **Mechanism:** score = z(21d return) - lambda * z(5d return); cross-sectional
    z each day, winsorized at 3 sigma.
  - **Attacks:** the current penalty-term scale can let the short-term penalty
    dominate. Z-scoring makes the terms comparable.
  - **Data:** close only. READY.
  - **Sub-refinement (fold in, do NOT spend a separate trial):**
    **V27 bounded penalty** = lambda * max(0, z_5d - threshold). Test as a
    parameter of V26, not a registry entry, to conserve the DSR budget.
  - [ ] Implement / register / TDD / run / record

- [ ] **V02+V05 -- Vanilla momentum + min-hold** *(regime-free diagnostic)*
  - **Mechanism:** single 21/5 score, fixed top-N, NO regime param switching,
    with V05 min-hold (the dominant turnover lever from Wave 1).
  - **Tests:** H2's implication -- can you drop the entire regime apparatus and
    keep the turnover rescue? If this matches V11, the regime machinery is dead
    weight.
  - **Data:** close only. READY. (A "plain" variant may already exist from the
    consolidation work -- verify before re-implementing.)
  - [ ] Implement / register / TDD / run / record

### Tier 2 -- ready now (close-only), secondary

- [ ] **V15 -- BEAR defensive basket**
  - **Mechanism:** in detector-BEAR, replace momentum selection with low-vol,
    low-beta, positive-absolute-momentum names.
  - **Attacks:** the untested half of the root-cause #1 recommendation (V12
    tested "BEAR->cash"; nobody tested "BEAR->defensive rotation"). More
    lag-forgiving than cash.
  - **Caveat:** still detector-dependent -- temper expectations vs the lag finding.
  - **Data:** vol + beta + abs-mom from close + SPY. READY.
  - [ ] Implement / register / TDD / run / record

- [ ] **V16 -- WEAK_BULL half exposure**
  - **Mechanism:** cap gross at 50% in WEAK_BULL; apply to all holdings.
  - **Attacks:** the largest time-weighted drag (WEAK_BULL = 43.6% of EXT-OOS,
    Sharpe -0.78). Persistent state, not an onset -- lag finding does NOT pre-empt.
  - **Caveat:** trusts the regime label (H2 cautions); may bake in over-labeling.
  - **Data:** close + VIX (detector). READY.
  - [ ] Implement / register / TDD / run / record

- [ ] **V08 -- Weekly rebalance**
  - **Mechanism:** rebalance once weekly near the close; keep crash exits daily.
  - **Tests:** is RAMP closer to weekly momentum than daily alpha? 21d signal
    barely decays over a week; cuts turnover hard.
  - **Data:** close only. READY. (`--rebalance-frequency` CLI flag exists.)
  - [ ] Implement / register / TDD / run / record

- [ ] **V29 -- Trend-quality score** *(test only if V28 shows promise)*
  - **Mechanism:** rank by log-price regression slope t-stat / R^2 over 21-63d,
    not raw endpoint return.
  - **Attacks:** same robustness goal as V28 (smooth trends > single gaps).
    Correlated with V28 -- gate it behind a V28 positive.
  - **Data:** close only. READY.
  - [ ] (conditional) Implement / register / TDD / run / record

### Tier 3 -- needs Gate 0 data prep

- [ ] **V33 -- Absolute-momentum + liquidity filter** *(detector-free crash protection)*
  - **Mechanism:** only buy names with positive absolute 21/63d return AND
    adequate dollar volume; otherwise cash.
  - **Attacks:** H6 from the signal side, WITHOUT the lagging detector -- a
    regime-free risk-off mechanism. Liquidity screen also fixes cost
    underestimation on names like SMCI.
  - **Data:** abs-mom = close only (CORE RUNS NOW); liquidity = volume (needs
    G0.1 + G0.2 + G0.3).
  - [ ] Run **abs-mom core** on close-only data first (no Gate 0)
  - [ ] Add liquidity screen after G0.3; re-run
  - [ ] Implement / register / TDD / run / record

- [ ] **V30 -- Sector-relative momentum** *(needs G0.4 sector fetch)*
  - **Mechanism:** rank stocks by excess return vs sector median (or sector ETF).
  - **Attacks:** the sector-concentration behind H6 (the tech/high-beta cluster).
  - **Data:** sector map (G0.4). Sector-ETF-relative alternative needs sector ETF
    OHLCV (also fetchable) -- prefer sector-median to avoid new downloads.
  - [ ] Implement / register / TDD / run / record

---

## Recommended run order

1. **Minimal probe (no data prep):** V31 + V28 + V26(+V27) + V02/V05 + V33-core.
   This is the highest-EV slice and needs nothing from Gate 0.
2. If the probe produces a candidate clearing V11: run **Gate 0**, then add
   **V30** and **V33-full-liquidity** to the same family gate.
3. **Tier 2** (V15/V16/V08, V29 conditional) only if the probe is inconclusive
   and you want broader coverage before deciding ship-vs-iterate.

---

## Metrics to record for EVERY run (strategy-pipeline.md)

One row per variant per timing mode. Append to the experiment registry
(`output/experiments.duckdb`, methodology Section 9.3).

| Variant | Timing | Cost | Sharpe | PSR | DSR | PBO | CAGR | MaxDD | DD dur | Calmar | Win% | PF | Trades | Avg hold | Turnover | IS/OOS | EXT-OOS Sharpe | vs V11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V31 | near_close | 5bps | | | | | | | | | | | | | | | | |
| V31 | one_day_lag | 5bps | | | | | | | | | | | | | | | | |
| V28 | ... | | | | | | | | | | | | | | | | | |
| V26 | | | | | | | | | | | | | | | | | | |
| V02+V05 | | | | | | | | | | | | | | | | | | |
| V33-core | | | | | | | | | | | | | | | | | | |
| V33-full | | | | | | | | | | | | | | | | | | |
| V30 | | | | | | | | | | | | | | | | | | |
| V15 | | | | | | | | | | | | | | | | | | |
| V16 | | | | | | | | | | | | | | | | | | |
| V08 | | | | | | | | | | | | | | | | | | |
| V29 | | | | | | | | | | | | | | | | | | |

Also required per methodology Section 12: per-regime + per-sector attribution,
gross-exposure/position-count over time, worst-drawdown periods, top
contributors/detractors, config dump.

---

## Explicitly EXCLUDED (deliberate -- do not re-litigate)

| Variant | Why excluded |
|---|---|
| V07 twice-weekly, V09 daily-exit/weekly-entry, V10 turnover budget | Redundant with V05 min-hold (the dominant Wave-1 lever) and V11's organic 39% turnover |
| V13-ladder (BEAR 0/25/50) | Interpolates tested endpoints (V12 cash Tier 3 + V11); adds overfit knobs |
| V14-hedge (SH/SPY short) | Hedges beta, but H6 says selection is the problem; new broker capability |
| V17 WEAK_BULL deconcentration | Second-order; fold into V16 if testing WEAK_BULL |
| V18 WEAK_BULL static params | Subsumed by V02 (vanilla removes all per-regime param surgery) |
| V19 three-state collapse, V20 two-state | Halfway points between tested endpoints (5-state vs V02 1-state) |
| V21 hysteresis, V22 confidence-gate | **Directly pre-empted** -- V12 debouncing tested this; lag is the binding constraint |
| V23 continuous exposure scaler | **H4 refuted** -- V4 vol overlay made EXT-OOS worse (-0.266) |
| V24 breadth override | H4 + leading-indicator findings (smooth, not threshold-firing); breadth lags rebounds |
| V32 risk-aware sizing | Inverse-vol sizing already refuted (V2: EXT-OOS -0.015); caps overlap V30/V31 |

**Hardening (not an alpha variant):** V25 / F4 fail-safe-on-missing-data is worth
doing as production robustness regardless, but is tracked separately, not in this
alpha gate.

---

## The null option (keep it on the table)

The campaign closure itself names **"close-and-ship-V11"** as a valid endpoint.
If the minimal probe produces nothing clearing V11 at honest DSR, the disciplined
move is to stop, ship V11, and not keep spending trials. This batch is justified
only because it attacks a mechanism the +0.08 ceiling does NOT bound -- it is not
a license to grid-search.

---

## File pointers

- Variant registry: `src/research/ramp_phase4/variants.py`
- Engine / state: `src/research/ramp_phase4/engine.py`
- Data loader (Gate 0 targets): `src/research/ramp_phase4/data.py`
- Target planner: `src/strategies/advanced/ramp_target_planner.py`
- Sector source: `src/data/yfinance/fundamentals.py`
- Readiness orchestrator pattern: `scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py`
- Variant glossary (update after each verdict): `docs/strategies/RAMP_VARIANTS.md`
- Methodology (authoritative): `docs/methodology/backtesting.md`
- Campaign closure (context): `docs/progress/20260524_RAMP_REGIME_DETECTOR_CAMPAIGN_CLOSURE.md`
- Root cause (the evidence this batch acts on): `docs/reports/ramp/20260505_root_cause_investigation.md`
- Wave 1 findings (V11 baseline): `docs/reports/ramp/20260522_phase4_wave1_findings.md`
