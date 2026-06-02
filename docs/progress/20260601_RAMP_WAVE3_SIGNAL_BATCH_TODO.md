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

## >>> FAMILY GATE VERDICT (2026-06-01) -- PROBE COMPLETE <<<

All 5 variants implemented, run on CLEAN split-adjusted data, gated together.
Report: `docs/reports/ramp/20260601_wave3_family_gate.md`.

**Clean cross-section (5 bps near_close, full window, 2355 days):**

| Variant | Sharpe | vs V11 | Max DD | AnnTO | 7.5bps | Cost gate |
|---|---:|---:|---:|---:|---:|:--:|
| **V28** multi-horizon ensemble | **0.811** | **+0.283** | -42.0% | 5,264% | 0.766 | PASS |
| **V31** beta-residual | **0.769** | **+0.241** | **-33.5%** | 7,217% | 0.702 | PASS |
| **V02+V05** vanilla (regime-free) | **0.683** | **+0.155** | -57.5% | 10,275% | 0.598 | PASS |
| V11 incumbent | 0.528 | -- | -66.2% | 10,325% | 0.452 | **FAIL** |
| V26 z-score | 0.533 | +0.005 | -42.7% | 9,492% | 0.438 | FAIL |
| V33-core abs-mom | 0.479 | -0.049 | -52.3% | 9,711% | 0.372 | FAIL |

**FOUR variants beat/tie V11; THREE beat it materially (V28, V31, V02+V05) and pass the
1.5x cost gate that V11 ITSELF FAILS.** Signal construction was the right lens -- this is a
real result, NOT bounded by the +0.08 detector-timing ceiling.

**Statistical gate:**
- **PSR:** V28 0.993, V31 0.990, V02+V05 0.980 (all pass vs SR=0).
- **DSR (n_trials sensitivity):** V28 PASSES at n<=12 (Wave-3 family-reset, documented),
  FAILS at n>=36 (if chained to the v0/detector campaign). V02+V05 fails at all n (kurt 25.5).
- **PBO = 0.503** -- crosses the 0.50 "strong overfitting" line. Family selection is
  time-period-unstable (sub-window orderings differ). **This is the binding gate: do NOT
  skip a formal walk-forward.**
- **Sub-window stability:** V28 beats V11 in ALL three eras (2017-21 +0.08, 2022-24 +0.17,
  2024-26 +0.81); the 2024-26 surge (1.429) may be tail-regime concentration. V02+V05 most
  consistent (graceful, not tail-driven).

**VERDICT: V28 = HOLD -> GRADUATE TO PURGED/EMBARGOED WALK-FORWARD** (methodology Section 3;
PBO failure means the walk-forward is mandatory, not optional). V31 is a strong co-candidate
(lowest DD, directly attacks H6/H8) -- check V28/V31 correlation; if >0.85 pick one via WF
OOS. V02+V05 = HOLD secondary (H2 mechanism confirmation: regime-free beats regime-aware).
**Null option (ship V11) is NOT the call** -- +0.28 Sharpe with PSR 0.993 across all sub-windows
is too strong to discard; but V11 stays the deployed paper incumbent until V28 clears the WF.

**WALK-FORWARD COMPLETE (2026-06-01): V28 + V31 both REJECT. Null option ACTIVE -- V11 stays.**
- 7 sequential OOS calendar-year windows. **ALL FIVE proposed variants were OOS-evaluated;
  ALL FIVE REJECT** (none beats V11 in every window; every worst window is 2022): V28 3/7
  (worst -0.496), V31 5/7 (-0.745), V02+V05 4/7 (-0.120 = best worst-case, beats V11 in BEAR),
  V26 3/7 (-0.589), V33-core 4/7 (-1.543, abs-mom gate whipsawed). Pooled OOS V28 0.889 /
  V31 0.910 > V11 0.647, but the edge is concentrated in 2023-2025 up-markets; the regime-free
  signals lose the 2020/2022 BEAR years where V11's regime-cash mode protects. PBO 0.503 was
  prescient. The "V11 stays" verdict is robust across the WHOLE family.
- Report: `docs/reports/ramp/20260601_wave3_walkforward.md`; session log
  `docs/progress/20260601_RAMP_WAVE3_WALKFORWARD.md`. 201 tests pass.
- **OPEN LEAD (untested, the one direction the data supports):** HYBRID = V28/V31 up-market
  signal + V11 regime-cash BEAR overlay -- keep the momentum edge, restore downside protection.

**(superseded) NEXT PHASE (was IN PROGRESS 2026-06-01):** V28 + V31 walk-forward / OOS-robustness validation.
- V28/V31 daily-return correlation = **0.801** (<= 0.85 -> INDEPENDENT, carry BOTH). V28-V11
  0.477, V31-V11 0.565 (both genuinely different from the incumbent).
- METHODOLOGY (D0): V28/V31 have FIXED a-priori parameters (no fitting), so this is NOT
  parameter-optimization walk-forward -- the overfitting vector is variant SELECTION (PBO
  0.503), not parameter fitting. Rigor applied: (a) sequential OOS robustness across >=5
  rolling windows -- V28 (and V31) must beat V11 in EVERY OOS window, no Sharpe collapse;
  (b) selection robustness (per-window family ranking); (c) NO weight optimization (keep
  fixed); light a-priori-neighborhood sensitivity optional + DSR-costed. Document which
  parts of methodology Section 3 bind given no fitting.

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

- [x] **G0.0 (RESOLVED 2026-06-01 -- it was a STALE PATH, not a missing rebuild)**
  **Root cause: `SIP_SPLIT_REL` pointed at the OLD tree location.** The 1-min SIP tree
  moved to `equities/sip_split/1min` (~2026-05); the stale constant made
  `_load_or_build_sip_daily_cache` return None and silently fall through to the corrupt
  LEGACY cache. **Fix (commit 429df47): one-line path correction + docstrings.** The clean
  `daily_from_sip` cache (2017..2026-05-15, NFLX continuous, no unadjusted splits) ALREADY
  existed and is now used. V11's 0.528 baseline is UNAFFECTED (V11 ran 2026-05-23 before the
  tree moved, on this same clean cache -- no re-validation needed). Verified end-to-end:
  loader reports "FRESH SIP-aggregated daily panel", NFLX worst daily ret -3.9%. 137 tests pass.
  This also closes **G0.2** (stale path constant). NOTE: V31's recorded Sharpe is contaminated
  (ran on legacy before the fix); re-run V31 clean before the family gate so its return stream
  is uncorrupted. Turnover verdict (7133%) stands regardless.
  - **(superseded) original finding -- kept for the record:** legacy daily cache
  `equities_daily_from_sip.parquet`... (the corrupt artifact was actually the separate
  `equities_daily_cache.parquet` LEGACY fallback, not the daily_from_sip cache).
  - **Finding:** `H:/Stock_Data/cache/ramp_phase4/equities_daily_from_sip.parquet`
    (the LEGACY cache `load_universe_panel` currently falls back to) carries the
    UNADJUSTED Netflix 10:1 split: NFLX 2025-11-17 prev=1116.975 -> cur=110.15
    (-90.1% phantom return). V31's equity went negative downstream (-102% on
    2025-12-05; NaN CAGR; MaxDD > -100%). If one split is unadjusted, others across
    2017-2026 likely are too.
  - **`sip_split` is CORRECT:** column-subset (`['timestamp','close']`) reads of
    `H:/Stock_Data/equities/sip_split/1min/symbol=NFLX/...` succeed AND show NFLX
    continuous (~110) across 2025-11-17. So the canonical split-adjusted source is
    clean; only the legacy daily roll-up is corrupt. Column-subset reads sidestep
    the G0.1 full-table defect.
  - **Fix:** rebuild the daily close panel by reading ONLY the `close` column from
    each `sip_split` monthly parquet (split-adjusted, continuous) and aggregating to
    daily last; point `load_universe_panel` at the rebuilt cache. (Same column-subset
    technique G0.3 uses for volume.)
  - **Contaminates:** ALL Wave-3 variants AND the V11 incumbent baseline (0.528) --
    V11's window ends 2026-05-16, so if V11 held NFLX on 2025-11-17 its baseline is
    also unreliable. Re-validate V11 on the clean cache before trusting any vs-V11 delta.
  - **Blocks:** V28, V26, V02+V05, V33-core (the rest of the probe) -- do NOT run
    them on the contaminated cache.

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

- [x] **G0.5 Per-backtest durability: wire the experiment registry into the
  readiness orchestrator** *(run-durability -- applies to ALL variants, not data prep)*
  *(DONE 2026-06-01: built `scripts/backtest_scripts/ramp_phase4_wave3_readiness.py`
  -- single-variant runner with per-step append_run, resume-skip on
  (strategy+cost+timing+git SHA+snapshot), atomic .tmp+os.replace artifact writes.
  Validated on V31: 4 full-window runs persisted, 10,928 return-stream rows,
  resume-skip confirmed.)*
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

- [x] **V31 -- Beta-residual momentum** -- **VERDICT (CORRECTED at family gate): BEATS V11
  -- Sharpe 0.769 (+0.241), LOWEST max DD -33.5%, cost gate PASS. 2nd-best variant.**
  *(dispatch 1 number 0.307 was WRONG -- TWO bugs: ran on the corrupt legacy cache AND a
  dtype bug (pct_change on object cols -> np.isnan TypeError) masked by that cache. Both
  fixed; re-run clean at the family gate.)* *(2026-06-01)*
  - **Mechanism:** rank residual returns after removing trailing SPY beta
    (estimate beta over 60-126d; rank residual 21d return).
  - **Attacks:** H6/H8 directly -- the BEAR losers were high-beta names that only
    looked strong on market beta. Residualizing removes exactly those.
  - **Result:** built as "V11 with the ranking metric swapped" (90d beta window).
    Sharpe 0.307 near_close / 0.314 one_day_lag at 5 bps (vs V11 0.528, **delta
    -0.221**). PSR 0.930 (FAILS 0.95). **Annualized turnover 7133%** vs V11's 39%
    -- the beta-residual score is wildly unstable day-to-day; rank_buffer + min_hold
    cannot hold it. ~7%/yr cost drag. Disqualified on turnover alone, independent of
    the data defect below. Report: `docs/reports/ramp/20260601_wave3_v31.md`.
  - **CAVEAT:** the Sharpe is contaminated by the legacy-cache split defect (see
    Gate-0 finding below); turnover is data-defect-independent so the verdict stands.
  - [x] Implement plan_fn in `variants.py`; register in `REGISTRY`
  - [x] TDD: `tests/research/ramp_phase4/test_variants.py` (137 pass)
  - [x] Run gate; record metrics row

- [x] **V28 -- Multi-horizon momentum ensemble** -- **VERDICT: BEATS V11 -- TOP CANDIDATE
  (+0.283 Sharpe at HALF the turnover; turnover "disqualification" was a phantom, see
  correction block below)** *(2026-06-01, dispatch 2)*
  - **Mechanism:** blend 21d/63d/126d relative returns with fixed weights
    (0.5/0.3/0.2) + 0.1 * 5d reversal penalty. No grid search.
  - **Attacks:** signal instability (the H2 root). A more stable signal may
    reduce the need for regime gating entirely.
  - **Result:** Sharpe **0.811** near_close / **0.851** one_day_lag at 5 bps
    (vs V11 0.528, **+0.283 BEAT**); PSR 0.9928 (PASSES); 7.5 bps 0.766 (cost gate
    passes on flat-bps). Monthly win 54%, PF 1.16. P&L from STRONG/WEAK_BULL;
    BEAR/UNPREDICTABLE drag (same shape as V11). **BUT annualized turnover 5264%
    vs V11's 39% (135x)** -- the multi-horizon score reranks the universe daily;
    delta_rebalance 0.02 + rank_buffer + min_hold can't hold it. The flat 5 bps
    model understates true market impact at this churn, so the 0.811 net is
    optimistic. Report: `docs/reports/ramp/20260601_wave3_v28.md`.
  - **Read:** the multi-horizon BLEND has genuine alpha (clean PSR, lag-robust) --
    the signal is good; the TURNOVER is the problem. Refinement axis: pair the V28
    signal with stronger turnover control (V08 weekly rebalance / wider rank_buffer /
    longer min_hold / higher delta threshold) and re-measure NET. Most promising
    lead so far, conditional on taming turnover.
  - [x] Implement / register / TDD (142 pass) / run / record

> **!!! CORRECTION (2026-06-01, after running V11 through the SAME runner): the
> "turnover blowup" was a PHANTOM -- a definitional mismatch, not a real defect.**
> The "V11 turnover 39%" baseline (in memory, the TODO, and HARDCODED in the runner's
> report template) used a DIFFERENT metric than the Wave-3 runner's annualized two-sided
> turnover. Measured through the SAME runner, **V11's AnnTO is 10,325%** at 5 bps
> near_close -- the SAME order as every Wave-3 variant. Corrected, internally-consistent
> turnover (runner AnnTO, near_close 5 bps):
>
> | Variant | Sharpe | AnnTO | Read (corrected) |
> |---|---:|---:|---|
> | V11 (incumbent) | 0.528 | 10,325% | baseline |
> | **V28** | **0.811** | **5,264%** | **BEAT: +0.283 Sharpe at ~HALF the turnover** |
> | **V02+V05** | **0.683** | 10,275% | **BEAT: +0.155 Sharpe at ~same turnover** |
> | V26 | 0.533 | 9,492% | tie |
> | V31 | 0.307 | 7,133% | worse Sharpe (NOT a turnover outlier) |
>
> **Earlier "turnover-disqualified" verdicts for V28/V26/V02 were WRONG.** Judge the
> family on Sharpe + PSR + the family DSR/PBO gate + EXT-OOS robustness. V28 (multi-horizon
> ensemble) is the standout; V02+V05 beating V11 while REGIME-FREE is direct support for
> H2 (the regime apparatus may be net-negative). Runner report string fixed in code.

- [x] **V26 -- Z-score normalized score** -- **VERDICT: TIE with V11 (Sharpe +0.005,
  similar turnover ~9,492%); not a clear advance. Cost gate fails at 7.5 bps** *(2026-06-01, dispatch 3)*
  - **Mechanism:** score = z(21d) - 1.0 * z(5d); cross-sectional z each day, winsorized at 3 sigma.
  - **Result:** near_close 0.533 / one_day_lag 0.664 at 5 bps (vs V11 0.528 -- **+0.005,
    a TIE not a beat**). PSR 0.9465 (nc) / 0.9769 (lag). **Turnover 9492%** (worst in the
    batch). **Cost gate FAILS at 7.5 bps (0.438 < 0.5).** Regime drag shape matches
    V11/V28/V31 (BEAR/UNPREDICTABLE negative). Report: `docs/reports/ramp/20260601_wave3_v26.md`.
  - **V27 bounded penalty (informational sensitivity, folded in -- no separate trial):**
    Sharpe 0.533, identical to default -- the bounded threshold (penalize only z5>1.0)
    has ZERO impact on this signal. V27 is dead.
  - **Read:** z-scoring did NOT fix fragility; it churns as hard as raw scores. The lag
    Sharpe 0.664/PSR 0.977 is the strongest raw number so far, but same "alpha exists,
    turnover kills it" story. Not advancing as-is.
  - [x] Implement / register / TDD (148 pass) / run / record

- [x] **V02+V05 -- Vanilla momentum + min-hold** -- **VERDICT: BEATS V11 (+0.155 Sharpe at
  ~same turnover); REGIME-FREE -> direct support for H2** *(2026-06-01, dispatch 4)*
  - **Mechanism:** RAMPSignals SIDEWAYS params (21/5, long_w 0.5, pen_w 2.0), fixed
    top_n=10, NO regime switching, NO rank_buffer (minimal control), + min_hold(5).
  - **Result:** near_close 0.683 / one_day_lag 0.844 at 5 bps (vs V11 0.528,
    **+0.155 raw BEAT**); PSR 0.9804/0.9949; 7.5 bps 0.598 (cost gate passes).
    Turnover 10,275%. All days labelled VANILLA (detector confirmed uninvolved).
    Report: `docs/reports/ramp/20260601_wave3_v02+v05.md`.
  - **H2 read:** regime apparatus NOT clearly dead weight -- V02 deliberately omits
    rank_buffer, so its turnover vs V11 mostly measures rank_buffer's effect, not the
    regime logic. The Sharpe BEAT is real at gross level.
  - **!! MEASUREMENT FLAG:** V02 has no rank_buffer AND blows up; but V31/V28/V26 DO
    use rank_buffer and ALSO blew up. That inconsistency means the "V11 = 39% turnover"
    baseline may be computed differently than the Wave-3 runner's annualized turnover.
    **Must run V11 THROUGH THE WAVE-3 RUNNER for an apples-to-apples turnover number
    before trusting any turnover-blowup narrative.** (Resolving next, pre-family-gate.)
  - [x] Implement / register / TDD (153 pass) / run / record

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

- [x] **V33-core -- Absolute-momentum cash gate (close-only)** -- **VERDICT: NOT ADVANCING
  (cuts drawdown but costs more CAGR than it saves)** *(2026-06-01, dispatch 5)*
  - **Mechanism:** regime-free; buy only names with ret_21d>0 AND ret_63d>0, top_n by
    momentum among survivors, cash residual when <top_n qualify, + min_hold(5).
  - **Result:** near_close 0.479 / one_day_lag 0.573 at 5 bps (vs V11 0.528, **-0.049**);
    AnnTO 9,711%; PSR 0.922; **cost gate FAILS at near_close 7.5 bps (0.372 < 0.5)**.
    **Max DD -52.4% vs V11 -66.2% (the cash gate DOES cut tail risk -13.8 pp)** but
    CAGR halves (8.4% vs V02's 16.8%) -- de-risks through recoveries too. Net-negative
    Sharpe trade. Report: `docs/reports/ramp/20260601_wave3_v33-core.md`.
  - **Read:** endogenous crash protection works for drawdown but the Sharpe cost is too
    high; an asymmetric (faster re-entry) gate might recover it, but not as-is.
  - **Liquidity screen (volume / Gate 0.3): still deferred -- not needed, V33-core already
    rejected on the close-only core.**
  - [x] Implement / register / TDD (159 pass) / run / record

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
