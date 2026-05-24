# RAMP Research State -- 2026-05-24

## Summary

Three concurrent RAMP research workstreams in flight as of 2026-05-24:

1. **V11 production paper** -- live since 2026-05-23. A7 paper-validation gate counting toward 5 clean sessions. No code action; calendar-driven.
2. **Regime detector diagnostic** -- complete. Local-only branch `regime-detector-diagnostic`. Five hypotheses tested; H4 (no hysteresis) and H5 (SMA lag) supported; produces a reusable evaluation harness.
3. **V12 BEAR-to-cash variant** -- complete, Tier 3 readiness verdict. Local-only branch `v12-bear-to-cash`. NOT deployed; activated WS-3 (detector improvement) as next research priority.

The connecting thread: the production detector is unchanged from December 2025, but we now have a rigorous diagnostic of its failure modes (lag + flicker) AND quantitative proof that those failure modes translate to lost P&L when used as a BEAR-to-cash signal. The detector is the bottleneck; next-session work targets that.

## Active branches

| Branch | Tip | Pushed? | Purpose |
|---|---|---|---|
| `main` | `d60686e` | yes | Stable; last commit Dec 2025 |
| `ramp-phase4-turnover-regime-research` | `fc7de60` | yes | V11 work; production paper deploy basis |
| `regime-detector-diagnostic` | `f05f388` | **no** | Diagnostic infrastructure + V12 spec history (rev1-rev4 + followup) |
| `v12-bear-to-cash` | `fd94deb` | **no** | V12 implementation + readiness + Tier 3 verdict |

Two of the four branches are local-only. No merges to `main` are pending; all of the post-V11 work waits on either V11's paper-validation outcome or downstream WS-3 work.

## Workstream 1: V11 Production Paper

**Status**: Running. A7 paper-validation timer fires Mon-Fri 16:05 ET. Gate = 5 clean sessions.

**Deployed**: 2026-05-23 04:30 UTC. Variant flipped via `config/trading/strategy_toggle.yaml` (`ramp.variant: v11`). EC2 service `homeguard-multi` running.

**V11 readiness verdict** (`docs/reports/ramp/20260523_phase4_v11_readiness.md`): PARTIAL.

| Gate | V11 | Threshold |
|---|---:|---:|
| PSR (vs SR=0) | 0.944 | > 0.95 (FAIL by 0.006) |
| DSR (n_trials=20) | 0.811 | > 0.95 (FAIL) |
| PBO across 5 variants | 0.126 | < 0.5 (PASS) |
| One-day-lag delta | +9.79% | within 20% (PASS) |

V11 absolute Sharpe 0.528 (9 years) is the binding constraint. Multi-trial correction makes it fail strict significance, but structural gates pass cleanly. Paper validation is the next OOS channel.

**No action this session** -- branch is stable on origin, EC2 deploy is autonomous, A7 timer ticks daily.

## Workstream 2: Regime Detector Diagnostic

**Status**: Complete; synthesis written; LOCAL-ONLY branch.

**Spec**: `docs/superpowers/specs/2026-05-23-regime-detector-diagnostic-design.md`
**Plan**: `docs/superpowers/plans/2026-05-23-regime-detector-diagnostic.md`
**Synthesis report**: `docs/reports/ramp/20260523_regime_detector_diagnostic.md`

Six-phase diagnostic over 2017-2026 sample, no production code changes. Produces a reusable harness (driver + ground-truth labelers + analysis notebook) suitable for future detector revisions.

### Hypothesis verdicts

| ID | Hypothesis | Verdict | Key evidence |
|---|---|---|---|
| H1 | BEAR conjunction structurally too restrictive | REFUTED (literal) / SUPPORTED (reframed) | BEAR fires 16.2% overall (predicted < 5%). Reframed: 53.9% of G1_BEAR (drawdown > 10%) days are MISSED by the detector. |
| H2 | UNPREDICTABLE has dead zones in uptrends | REFUTED | UNPREDICTABLE rare across all regimes (14 runs total, 1.7%); not a dead-zone artifact. |
| H3 | 252-day VIX percentile compresses adaptively | REFUTED | Lookback sensitivity 27-32% firing rate across {63, 126, 252, 504}d -- material but not dramatic. |
| H4 | No hysteresis -> label flicker | **SUPPORTED** | 4 of 5 regimes have median run length <= 2 days. Diagonal mass 0.761 on transition matrix. |
| H5 | SMA-based inputs lag regime onset | **SUPPORTED** | Median lag to first BEAR label = 14 days; max 36; 5/5 drawdown events eventually captured. |

### Phase 0 finding (critical for downstream interpretation)

The detector is a **score-based argmax** classifier (each regime gets a soft score in [0,1], winner is argmax), NOT a 5-AND hard conjunction as the OMR / RAMP docs implied. Uses raw SMAs (not Kalman, despite some legacy doc references).

### Remediation ranking

The diagnostic ranked five candidate detector revisions:

1. **Option B (hysteresis layer)** -- top, directly addresses H4
2. Option E (score-based reformulation) -- partially current architecture
3. Option C (VIX lookback adjustment) -- modest signal
4. Option A (threshold recalibration) -- doesn't address structural lag
5. Option D (leading indicators) -- would address H5 but bigger lift

### Phase 5 recommendation

Track (c): pursue **both** WS-2 (RAMP BEAR-day overlay) and WS-3 (detector improvement) in parallel, with WS-2 prioritized because it tests whether detector improvements would matter at all.

WS-2 has now happened as V12 (see below). It confirms WS-3 is the higher-leverage next step.

## Workstream 3: V12 BEAR-to-cash

**Status**: Complete; Tier 3 verdict; deployment deferred.

**Spec history**: 4 revisions tracked in-file
- `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (rev4 + rev4-followup)

**Plan**: `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md`
**Readiness report**: `docs/reports/ramp/20260524_phase4_v12_readiness.md`
**Session log**: `docs/progress/20260524_RAMP_V12_READINESS.md`

### Implementation (6 tasks across 8 commits on `v12-bear-to-cash`)

| Task | Commit | What |
|---|---|---|
| 1 | `9e5e211` | Engine state additions (`regime_streak`, `last_validated_regime`) + `_engine_pre_variant_update` helper |
| 2 | `f2bf56f` | Config schema (`regime_positions`, `min_regime_days`) + validation |
| 3 | `5876573` | `_variant_v12` + REGISTRY + canonical pinning test (the source of truth for symmetric debouncing semantics) |
| 4 | `96034b4` | Integration tests via detector monkeypatch |
| 5 | `1d01102` | Readiness orchestrator (17 backtests, gate vs sensitivity split) |
| 6 | `c4e4f0d` | RAMP_VARIANTS.md canonical glossary |

Plus 5 post-readiness commits (`5a88903`, `409c575`, `6efa6a5`, `fd28289`, `fd94deb`) for the Tier 3 verdict correction (Gate 4 bug + docs).

**Test totals**: 94 passing under `tests/research/ramp_phase4/` (baseline 69 + 25 new V12 tests). Zero regressions.

### Readiness verdict

18 backtests in 15.86 min on local fintech env.

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR (vs SR=0) | FAIL | 0.7881 | > 0.95 |
| 2. DSR (n_trials=22) | FAIL | 0.5418 | > 0.95 |
| 3. PBO across 6 variants | PASS | 0.3934 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc-lag=-0.397 | <= 0.100 |
| 5. Cost floor + no-regress | PASS | V12 lag@7.5bps = 0.608 | > 0.30 AND >= 0.9*V11 |

**Overall: PARTIAL / Tier 3** per spec rev4 success criteria.

### V12 vs V11 head-to-head (5 bps)

| Variant | Sharpe (near_close) | Sharpe (one_day_lag) | CAGR (near_close) |
|---|---:|---:|---:|
| V11 | 0.528 | 0.580 | 11.93% |
| V12 default | 0.268 | 0.665 | 3.52% |
| V12-up-cash (UNPREDICTABLE='cash') | **0.586** | (not measured) | 9.83% |
| V12-deb-2 (min_regime_days=2) | 0.130 | (not measured) | 0.07% |
| V12-deb-3 | 0.315 | (not measured) | 4.83% |
| V12-deb-5 | 0.437 | (not measured) | 8.09% |

**V12 default near_close is WORSE than V11 by 0.26 Sharpe.** BEAR-to-cash costs more in transaction friction than it gains in avoided drawdown. V12-up-cash, which holds cash on UNPREDICTABLE days (not BEAR), beats V11 by +0.06. This is the actionable surprise.

### Detector-onset alignment panel

59 BEAR onsets in 2017-2026:
- **Mean gap_days = -3.42** between detector BEAR onset and SPY drawdown trough
- Mean avoided_return per cash period = **+0.18%**

Translation: the detector fires ~3.4 trading days AFTER the SPY trough on average. V12 cash periods bracket the recovery, not the crash. H5 (SMA lag) becomes a concrete strategy P&L cost.

### Gate 4 bug discovered + fixed

The orchestrator initially implemented Gate 4 as `lag_pass = abs(delta) <= cap` (symmetric). Spec rev4 specifies `(nc - lag) <= cap` (directional; lag > nc is safe). The bug masked V12's Tier 3 verdict as Tier 4 (FAIL). Fixed in commit `5a88903`; report patched in `409c575`. No re-run needed.

## Cross-workstream synthesis

**The picture**:

1. The production detector is **unchanged**. It's a December 2025 deploy. Score-based argmax over 5 regimes with raw SMAs and 252d VIX percentile.

2. **We now know precisely where it fails**: H4 (flicker -- 4 of 5 regimes have median run length <= 2 days) and H5 (SMA lag -- 14d median onset lag, max 36d).

3. **V11 with V11's filter stack ships despite PARTIAL readiness** because (a) its structural gates pass, (b) absolute Sharpe of 0.528 over 9 years is decent but borderline-significant, (c) paper validation is the next OOS evidence channel.

4. **V12's failure pinpoints why detector improvements are necessary, not just nice-to-have**: BEAR-to-cash should have been a win per the May 2026 root-cause investigation's V8 finding. It WASN'T, because the detector fires after the trough, so cash periods sit out recovery instead of selloff.

5. **The V12-up-cash sensitivity finding is the curious data point**: UNPREDICTABLE-to-cash works (+0.06 over V11), while BEAR-to-cash doesn't (-0.26). UNPREDICTABLE rarely fires (1.7%) but when it does it's productive. BEAR fires 16.2% but the timing is wrong.

## Decisions made

- **V12 NOT deployed** to production paper. Tier 3 verdict per spec rule.
- **WS-3 activated**: regime detector improvement (Option B + possibly Option D) is the top research priority going forward.
- **V12c queued**: UNPREDICTABLE='cash' as a candidate variant for the next research cycle. Requires its own readiness gate per spec honesty discipline.
- **V11 paper validation continues** independently. Outcome of WS-3 / V12c does not change V11's path.
- **No branches merged to main**. All post-V11 work waits on WS-3.

## Quantitative summary (one table)

Aggregate across the three workstreams:

| Aspect | Status / Value |
|---|---|
| Production-paper-deployed strategy | V11 (since 2026-05-23) |
| V11 A7 counter | 0/5 (timer fires Mon-Fri 16:05 ET) |
| V11 Sharpe (5 bps, 9 years) | 0.528 |
| V11 readiness | PARTIAL (PSR 0.94 / DSR 0.81) |
| V12 readiness | PARTIAL / Tier 3 (PSR 0.79 / DSR 0.54) |
| V12 vs V11 Sharpe delta | -0.26 (near_close) / +0.085 (one_day_lag) |
| Detector lag (median) | 14 days from drawdown peak to first BEAR |
| Detector flicker | 4 of 5 regimes have median run <= 2 days |
| BEAR onsets 2017-2026 | 59 |
| Mean gap_days (onset vs trough) | -3.42 (detector fires after trough) |
| Best V12 sensitivity variant | V12-up-cash (UNPREDICTABLE='cash') Sharpe 0.586 |
| Diagnostic verdicts | H1 reframed-supported / H2 refuted / H3 refuted / H4 supported / H5 supported |
| Next research priority | WS-3 (detector improvement, Option B hysteresis) |

## What's next

In priority order:

1. **WS-3 brainstorm** -- regime detector v1 design. Top candidate from Phase 5 diagnostic synthesis + Tier 3 V12 escalation. Two concrete tracks within:
   - **WS-3a**: hysteresis layer at the detector (Option B per diagnostic). Symmetric debouncing on regime classifications. NOT the same as V12's `min_regime_days` (which was at the variant level and didn't help).
   - **WS-3b**: faster BEAR detection via leading indicators (Option D per diagnostic). VIX term structure (VIX/VIX3M), credit spreads, breadth (advance-decline) as augmentations or replacements for the 20/50/200-SMA inputs.
2. **V12c brainstorm** -- UNPREDICTABLE='cash' as v12.1.0 default. Smaller scope; could run in parallel with WS-3 because they touch different parts of the stack.
3. **Monitor V11 paper validation** -- A7 counter. Continues regardless of WS-3 / V12c outcome.

## Reference inventory (files)

All paths under `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\`.

**Specs** (`docs/superpowers/specs/`):
- `2026-05-23-regime-detector-diagnostic-design.md`
- `2026-05-23-v12-bear-to-cash-design.md` (rev4 + followup)

**Plans** (`docs/superpowers/plans/`):
- `2026-05-23-ramp-research-roadmap.md` (3-workstream roadmap; partially superseded by this doc)
- `2026-05-23-regime-detector-diagnostic.md`
- `2026-05-23-v12-bear-to-cash.md`

**Reports** (`docs/reports/ramp/`):
- `20260523_phase4_v11_readiness.md`
- `20260523_regime_detector_diagnostic.md`
- `20260524_phase4_v12_readiness.md` (patched; Tier 3 verdict)

**Strategy** (`docs/strategies/`):
- `production/RAMP_STRATEGY.md` (V11 changelog entry 2026-05-23)
- `RAMP_VARIANTS.md` (canonical glossary V01-V14 reserved)

**Code** (key modifications since 2025-12):
- `src/strategies/advanced/market_regime_detector.py` -- UNCHANGED (the bottleneck)
- `src/research/ramp_phase4/` -- engine + filters + variants extended for V11 and V12
- `src/trading/adapters/ramp_live_adapter.py` -- V11 live adapter with `position_open_dates`, `variant`, V11 filter stack
- `scripts/backtest_scripts/ramp_phase4_v11_readiness.py` -- V11 readiness orchestrator
- `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` -- V12 readiness orchestrator (with Gate 4 directional fix)
- `scripts/diagnostics/regime_detector_replay.py` -- diagnostic driver (on diagnostic branch)
- `scripts/diagnostics/ground_truth_labelers.py` -- G1-G4 labelers (on diagnostic branch)

**Session logs** (`docs/progress/`, recent):
- `20260519_SESSION_FINDINGS.md`
- `20260522_SESSION_LOG.md`
- `20260522_RAMP_PHASE4_WAVE1.md`
- `20260523_RAMP_PHASE4_V11_READINESS.md`
- `20260523_RAMP_PHASE4_V11_PRODUCTION_PAPER.md`
- `20260524_RAMP_V12_READINESS.md`
- `20260524_RAMP_RESEARCH_STATE.md` (this doc)
