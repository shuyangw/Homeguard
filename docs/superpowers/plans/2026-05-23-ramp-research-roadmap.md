# RAMP Research Roadmap - Sequenced Workstreams

> **For agentic workers:** this is a META-plan covering three workstreams that span multiple weeks. Each workstream has its own spec + implementation plan that gets created when its turn comes. The roadmap exists to orchestrate the sequence and document gating logic between workstreams.

**Date**: 2026-05-23
**Status**: Active
**Owner**: Shuyang

## Context

Three RAMP research workstreams are in flight or queued:

1. **WS-1: V11 paper validation** — already deployed, A7 counter running. Passive calendar-driven monitoring.
2. **WS-2: V12 BEAR-to-cash on V11 base** — diagnostic Phase 5 + May 2026 root-cause both recommended this as the highest-leverage RAMP intervention. Not yet started.
3. **WS-3: v1 regime detector with hysteresis (Option B)** — diagnostic Phase 5's secondary recommendation. Gated on WS-2's outcome.

This roadmap sequences them so the higher-EV work runs first and lower-EV work only proceeds if needed.

## Sequenced phase chart

```
Time ----------------------------------------------------------------->

WS-1 (V11 paper):  [now running...... gate at counter=5 .... Phase D decision]
WS-2 (V12):                [brainstorm][spec][plan][impl][readiness][...validate]
WS-3 (v1 detector):                                                              [gate: only if WS-2 doesn't beat strict gates]
```

WS-1 is calendar-passive (no implementation in this roadmap). WS-2 starts now and progresses in parallel with WS-1 monitoring. WS-3 is conditional on WS-2's outcome.

## WS-1: V11 paper validation (in flight)

**Current state**: V11 deployed to EC2 2026-05-23 04:30 UTC. A7 counter at 0. Timer queued for Mon 2026-05-25 16:05 ET.

**No implementation work for this roadmap.** Monitoring only.

**Daily checklist (~30 seconds)**:

```bash
ssh ec2-user@$EC2_IP 'cat /var/lib/homeguard/a7_clean_sessions; cat /var/lib/homeguard/a7_last_session_date'
```

Or via Grafana: `hg_a7_clean_sessions` gauge.

**Gate**: counter reaches 5.

**Outcomes**:
- **Clean (counter -> 5)**: WS-1 enters Phase D decision (production live yes/no). User-driven, out of scope here.
- **FAIL** (counter resets to 0): investigate via `trade-log-analyzer` agent. Possible causes: comparator over-strict, broker fill drift, position-state-ledger desync. Fix and re-arm.
- **Stall** (counter doesn't move for >7 trading days): system issue. Check `systemctl status homeguard-multi homeguard-ramp-paper-check.timer`.

**Branch**: `ramp-phase4-turnover-regime-research` (on origin at `fc7de60`).

**Estimated calendar**: ~7-10 trading days assuming no FAILs.

## WS-2: V12 BEAR-to-cash on V11 base (active next session)

**Rationale**: Diagnostic Phase 5 + May 2026 root-cause investigation both point at "what RAMP does on BEAR days" as the dominant lever for RAMP. V11's PARTIAL readiness specifically fails strict PSR/DSR because absolute Sharpe (0.528) isn't high enough; BEAR-to-cash could lift this materially.

**Hypothesis**: when the regime detector fires BEAR (which it does on 16.2% of days per the diagnostic), exposure should be 0% (cash) instead of V11's normal long-momentum. The Phase 4 readiness investigation V8 showed Sharpe lift of +0.26 over V1 from this single change.

**Workflow**:

### WS-2.1 — Brainstorm (next session, ~1 hour)

Invoke `superpowers:brainstorming`. Topics to nail:

- **Trigger**: detector says BEAR. Per the diagnostic, BEAR fires 16.2% overall (range 1.6% in 2021 to 54.2% in 2022). On days when detector says BEAR, V12 holds 0% equity.
- **Latency**: detector fires BEAR with median 14-day lag from drawdown peak (Phase 4 Analysis D). That's a long tail; some events get 36 days. Accept the lag for now -- a faster trigger is WS-3's domain.
- **Re-entry**: when detector flips to non-BEAR (most likely UNPREDICTABLE -> SIDEWAYS -> WEAK_BULL), V12 re-enters via V11's normal rank_buffer + min_hold logic. The min_hold filter naturally smooths re-entry.
- **Hysteresis on the BEAR boundary**: should V12 add a min-N-days-in-BEAR before going to cash? And min-N-days-out-of-BEAR before re-entering? This sneaks Option B (WS-3) into V12. Defer to WS-3 unless V12's first results show whipsaw.
- **Open question**: does cash position-size go to 0, or does V12 hold a defensive ETF (TLT, SHY, GLD)? Spec defaults to cash; brainstorm decides whether to extend.
- **Scope**: V12 IS V11 + the BEAR-day exposure switch. No other changes.

Spec output: `docs/superpowers/specs/YYYY-MM-DD-v12-bear-to-cash-design.md`.

### WS-2.2 — Writing plan (next session, ~30 min)

Invoke `superpowers:writing-plans`. The implementation is small:

- Add `_variant_v12` to `src/research/ramp_phase4/variants.py` (~30 LOC) that wraps `_variant_v11` and checks `plan.regime == 'BEAR'` -> returns empty targets.
- Add tests in `tests/research/ramp_phase4/test_variants.py`.
- No engine changes needed.

Plan output: `docs/superpowers/plans/YYYY-MM-DD-v12-bear-to-cash.md`.

### WS-2.3 — Implementation (next session, ~1 hour)

Per the plan. Likely 2-3 commits.

### WS-2.4 — Run V12 through readiness orchestrator (~15 min)

Re-run `scripts/backtest_scripts/ramp_phase4_v11_readiness.py` with `V11` -> `V12` substituted. Need to update CROSS_VARIANTS to include V12 in the PBO trial set.

Expected outputs:
- V12 5 bps near_close Sharpe (compared to V11's 0.528).
- V12 PSR / DSR / PBO / lag-robustness.
- Verdict: does V12 clear strict significance where V11 didn't?

### WS-2.5 — Decision gate

| V12 verdict | Action |
|---|---|
| Clears ALL 4 gates (PSR/DSR/PBO/lag) | V12 is the new Phase D candidate. Skip WS-3. Extend A7 comparator for V12 (same filter stack as V11), redeploy. |
| Passes structural gates only (PBO/lag) | V12 is structurally sound but Sharpe still insufficient. Compare V12 vs V11's PARTIAL state -- is V12 strictly better? If yes, V12 is the deploy candidate. If no, escalate to WS-3. |
| Fails structural gates | V12 has overfitting or lookahead. Investigate. Possibly: BEAR-to-cash worked in backtest because the detector lagged the drawdown -- selection effect. WS-3 (better detector) becomes relevant. |

**Branch**: new branch `v12-bear-to-cash` based on `ramp-phase4-turnover-regime-research`.

**Estimated calendar**: 1 session (~3 hours) for brainstorm + plan + impl + readiness. Decision gate is immediate after readiness.

## WS-3: v1 regime detector with hysteresis (Option B) — conditional

**Gate**: WS-2's V12 must NOT clear strict significance. If V12 lands cleanly, skip WS-3.

**Why this is lower priority**: the diagnostic explicitly carries the "detector != bottleneck" caveat. WS-2 likely subsumes much of WS-3's value because BEAR-to-cash exposure is the high-leverage fix regardless of which day the detector fires BEAR.

**If we DO need WS-3**:

### WS-3.1 — Brainstorm Option B design (~1 hour)

Hysteresis layer specifics:
- Minimum dwell time: a regime must persist N days before the live runner accepts the flip. N=3? N=5? Brainstorm decides based on Phase 4 Analysis B median run-lengths.
- Threshold band: alternatively, add a hysteresis band around regime boundaries -- e.g., once in BEAR, stay in BEAR until VIX percentile drops below 60 (instead of just 70 to enter). Asymmetric thresholds.
- Where to inject: at the detector level (modify `MarketRegimeDetector`) vs at the consumer level (V11/V12 ignore flips less than N days old).

Spec output: `docs/superpowers/specs/YYYY-MM-DD-v1-detector-hysteresis-design.md`.

### WS-3.2 — Plan + implement (~3 hours)

Plan output: `docs/superpowers/plans/YYYY-MM-DD-v1-detector-hysteresis.md`. Implementation is detector-level change with full backward-compat (gated by a constructor flag so OMR / RAMP / future strategies can opt in).

### WS-3.3 — Re-run diagnostic against v1 detector (~30 min)

Re-run `scripts/diagnostics/regime_detector_replay.py` with the v1 detector class injected. Compare regime distributions, run-length distributions, lag-to-event metrics against the v0 baseline from this session's Phase 4 output.

The diagnostic infrastructure is designed to be re-runnable; this should be a 30-minute task.

### WS-3.4 — Re-run V11 / V12 readiness with v1 detector

Replace `MarketRegimeDetector()` with `MarketRegimeDetector(hysteresis=True)` in the variants. Re-run readiness orchestrator. Compare to v0 baseline.

**Gate**: detector v1 must lift V11 (or V12, whichever is the current candidate) to clear strict PSR/DSR/PBO/lag. If not, the methodology Section 4 caveat applies: V1 (vanilla momentum, no regime overlay) beats V0 OOS, so a complex detector that doesn't add value should be abandoned.

**Branch**: new branch `v1-detector-hysteresis`.

**Estimated calendar**: 1 session (~4-5 hours).

## Cross-cutting decision tree

```
WS-1 monitoring -> counter hits 5 -> Phase D decision (out of scope)
                  -> FAIL -> investigate, fix, re-arm (back to monitoring)
WS-2 V12        -> all 4 gates clear -> V12 is Phase D candidate (skip WS-3)
                  -> structural only -> compare to V11; pick best; consider WS-3
                  -> structural fail -> investigate; WS-3 becomes likely
WS-3 v1 det.    -> beats V1 baseline -> deploy
                  -> doesn't beat V1 -> abandon, the detector wasn't the issue
```

## What to do in the very next session

Recommendation: start WS-2.1 (V12 brainstorm). It's the highest-EV active work, doesn't depend on WS-1's outcome, and can complete in a single session (brainstorm + spec + plan + implementation + readiness re-run).

While WS-2 is in flight, WS-1 monitoring continues passively. WS-3 stays queued.

## Branches summary at the start of the next session

- `main` at `d60686e` (unchanged)
- `ramp-phase4-turnover-regime-research` at `fc7de60` (V11 paper deploy; on origin)
- `regime-detector-diagnostic` at `81206f8` (diagnostic complete; LOCAL ONLY)
- New branches to create:
  - `v12-bear-to-cash` from `ramp-phase4-turnover-regime-research` (WS-2)
  - `v1-detector-hysteresis` from `regime-detector-diagnostic` (WS-3, conditional)

## What this roadmap does NOT do

- Replace the V11 paper validation (WS-1) -- that's running on its own schedule.
- Predict the diagnostic's findings (already done in Phase 5).
- Commit to merging any branch into main -- merges wait for Phase D / strategy promotion decisions.
- Schedule WS-3 work -- it's conditional and may never happen.
