# Post-Campaign Next Steps -- 2026-05-25

**Date**: 2026-05-25
**Status**: Proposed
**Owner**: Shuyang
**Type**: Operational + research planning
**Supersedes**: campaign-closure doc's "next research direction deferred" framing
**Builds on**:
- `docs/progress/[campaign-closure].md` (2026-05-24 to 2026-06-02 RAMP regime-detector campaign)
- `ramp-phase4-turnover-regime-research@fc7de60` (V11 production paper deploy, 2026-05-23)
- GitHub repo state as inspected 2026-05-25

## State of RAMP today (verified against GitHub, not from memory)

**Production paper**: V11 deployed 2026-05-23 04:30 UTC on EC2 via
`homeguard-multi`. `config/trading/strategy_toggle.yaml` on
`ramp-phase4-turnover-regime-research@fc7de60` shows `ramp.enabled: true`,
`ramp.variant: v11`. Today is the first V11 rebalance day. The A7 paper-
validation counter begins incrementing at 16:05 ET today, target 5 clean
sessions.

**GitHub state**:

| Branch | Tip | Date | Contents |
|---|---|---|---|
| `main` | `d60686e` | 2026-05-22 | V11 base merge + parallel cleanup track |
| `ramp-phase4-turnover-regime-research` | `fc7de60` | 2026-05-23 | V11 production paper deploy + V01-V11 reports |
| `v12-bear-to-cash` | (LOCAL ONLY) | — | 28 commits: V12/V12c/V13/V14a/b/c + WS-3d + E2-E8 + closure |

**Critical operational fact:** The entire 2026-05-24 to 2026-06-02 campaign
lives on a local-only branch. 28 commits, 12+ reports, 8+ session logs,
4 specs, the V20 LightGBM detector pipeline, 4 leading-indicator
acquirers, and the closure document itself -- none of it is pushed.
A laptop hardware failure today loses all of it.

**Strategy toggle anomaly**: The deployed toggle yaml shows `omr.enabled:
false` and `mp.enabled: true`. Memory state has OMR as an active
production strategy and MP as not. Either memory is stale or the deploy
swapped them. Worth confirming, but not a blocker for V11 monitoring.

## Three workstreams, sequenced

The closure document deferred "next research direction" to a fresh
session. That framing under-prioritizes the operational and archival
work that has hard deadlines. The right sequence is:

1. **Operational** (this week) -- protect V11's A7 validation; preserve
   campaign artifacts.
2. **Archival** (this week) -- push the local branch; reconcile state
   with what's on GitHub.
3. **Research direction** (after A7 outcome) -- pick the next line
   conditional on V11's paper-validation result, not in the abstract.

These are not parallel choices. The operational work is genuinely time-
critical; the research direction depends on V11's outcome.

## Workstream A: Operational (V11 paper validation)

V11 is the only RAMP variant currently in production. Its A7 outcome
shapes every downstream decision. Three concrete tasks:

### A1. Monitor A7 counter through the 5-session window

The A7 helper runs Mon-Fri 16:05 ET. Counter location:
`/var/lib/homeguard/a7_clean_sessions`. Target: 5 clean sessions.

Earliest pass date: 2026-05-30 (Friday) if every session is clean from
today through Friday. Realistic pass date: mid-June, allowing for
occasional unclean sessions.

Daily check after 16:05 ET:
- A7 counter value (incremented? reset?)
- Comparator output: live `compare_paper_vs_plan` script result, looking
  for `--variant v11` confirmation in logs
- Position-ledger state: `ramp_position_state.json` reflecting V11
  filter outputs (rank_buffer + min_hold)

If counter resets to 0: investigate the unclean session before continuing.
A reset is informative, not a failure -- it tells us the V11 filter
stack hit an edge case the comparator didn't expect.

### A2. Define the "what if A7 fails" failure path

The closure document and prior state docs do not specify what happens if
V11 fails A7 (e.g., counter resets to 0 multiple times without making
forward progress). Three plausible responses:

1. **Investigate and re-deploy**: diagnose the unclean-session cause,
   patch the variant or comparator, restart the A7 timer.
2. **Roll back to V01**: re-deploy the pre-V11 baseline as production
   paper while RAMP regime work is paused. Less ambitious but preserves
   a paper-validated baseline.
3. **Halt production paper entirely**: shut down RAMP's slot in
   `strategy_toggle.yaml`, redirect to non-RAMP work.

Pre-register the decision rule now, before A7 outcome is known. My
recommendation: 2 consecutive A7 resets with the same root-cause class
triggers Option 1 (investigate and re-deploy). 3+ resets across mixed
root causes triggers Option 2 (roll back). Option 3 only if structural
RAMP issues surface that V11 can't address.

### A3. Strategy-toggle state audit

The toggle yaml on `ramp-phase4-turnover-regime-research@fc7de60` shows:

- `cscm.enabled: false` (consistent with IBKR migration pause per memory)
- `mp.enabled: true` (memory does not flag MP as an active strategy)
- `omr.enabled: false` (memory flags OMR as active)
- `ramp.enabled: true, variant: v11` (the V11 deploy)

The OMR/MP situation is either a memory-state inconsistency or an
accidental deploy state. ~30 minutes to confirm on EC2 (`systemctl
status homeguard-omr`, check actual running services). Resolve before
the A7 window closes so RAMP's A7 outcome isn't confounded with
unrelated strategy-stack instability.

## Workstream B: Archival (push the local branch)

The local-only branch `v12-bear-to-cash` contains 28 commits of campaign
work. Three honest options for what to do with it:

### Option B1: Push the full branch to GitHub as-is

- Pro: zero risk of artifact loss; matches V12 spec discipline of
  preserving all variant code "for diagnostic continuity"
- Pro: future researchers can replay the full campaign
- Pro: ~5 minutes of work (`git push -u origin v12-bear-to-cash`)
- Con: 28 commits including failed-gate variants, intermediate states,
  and dead-end LightGBM training runs. Branch will sit on GitHub forever
  even though it's not on the main path.

### Option B2: Curated squash-merge into a single closure commit on `main`

- Pro: clean history on main; the V12-V14 variant code lands in
  REGISTRY for diagnostic continuity but the campaign noise is collapsed
- Pro: closure doc and key reports land on main where they're
  discoverable
- Con: ~half day of work to identify the right curation
- Con: per-commit attribution is lost; if any commit's lessons are later
  re-derived, the original analysis isn't visible
- Con: PRESERVES less than B1; the rejected paths (Gate 1 rounds 1-3,
  Amendment 6 reframing, V14 boundary tests) are valuable evidence
  against re-attempting them

### Option B3: Push as a long-lived "archive" branch (B1 + naming)

- Variation on B1. Push the branch but rename it from `v12-bear-to-cash`
  (which describes only the first variant) to something like
  `archive/regime-detector-campaign-2026-05` that signals its purpose.
- Standing convention: branches matching `archive/*` are read-only and
  never merged to main. Documentation lookup tool: future docs reference
  `archive/regime-detector-campaign-2026-05@<sha>` for specific evidence.

**Recommendation: B3.** The campaign's artifacts have genuine reuse
value for any future regime-aware work (the 8 variant codes, the
WS-3d pipeline that any LightGBM-on-leading-indicators future attempt
inherits, the 4 acquirers, the diagnostic infrastructure). Compressing
them into a squash loses information. But the campaign is also closed
TIER 4 and shouldn't muddy `main`'s linear history. An archive branch
captures both constraints.

### B4. Cherry-pick high-value infrastructure to main

Independent of B1/B2/B3, some infrastructure from the campaign has
value beyond the campaign itself and should land on main:

| File / module | Why land on main |
|---|---|
| `src/data/leading_indicators/` (4 acquirers) | VIX term, HY OAS proxy, breadth, SKEW are reusable for any future signal work. No regime-specific coupling. |
| `MarketRegimeDetector.last_classification_timestamp` field | Tiny addition; useful for any future detector consumer wanting freshness assertions. Non-breaking. |
| `scripts/diagnostics/regime_detector_replay.py` + `ground_truth_labelers.py` | Already exist on main? Confirm. These are the reusable harness behind H1-H5 and any future detector evaluation. |
| `src/backtesting/validation/cpcv.py` (already on main per closure doc) | No-op; already integrated. |
| WS-3d's V1 detector class (`market_regime_detector_v1.py`) | NOT recommended for main. It's a TIER 4 detector; landing it on main creates ambiguity about which detector is canonical. Keep in archive only. |

Cherry-pick the first three (~half day). Leave the V20+ detector class
in the archive only.

## Workstream C: Research direction (post-A7)

The closure document lists five candidate directions. I'll rank them
against three criteria: (1) does this require V11 to have shipped, (2)
is the multi-trial budget consideration material, (3) realistic time-
to-evidence.

### Option C1: Universe expansion (S&P 500 + NASDAQ-100 union, then Russell 1000)

- **Premise**: RAMP's current universe is S&P 500. Tier 1 (S&P 500 ∪
  NDX) is the lowest-risk expansion per the May 2026 root-cause
  investigation.
- Requires V11 shipped: not strictly. Could run on V11-paper-validated
  variant or on V01 baseline.
- Trial-chain: opens a new dimension (universe size) that DSR n_trials
  can interpret as a single new trial if disciplined, or many if loose.
- Time to evidence: ~1 week for Tier 1 backtest + readiness rerun.
- **Best if**: V11 ships A7 cleanly. Universe expansion is the natural
  next ambition for a working V11.

### Option C2: RAMP-OMR portfolio construction

- **Premise**: OMR has structural negative correlation with RAMP (per
  memory; one bull, one bear). Combining at portfolio level might
  produce a higher-Sharpe ensemble than either alone.
- Requires V11 shipped: yes if RAMP side is V11; alternatively can use
  V01 RAMP.
- Trial-chain: cleanly resets if structured as a portfolio-construction
  layer, not a strategy modification.
- Time to evidence: ~2 weeks for portfolio backtest infrastructure +
  walk-forward validation.
- **Best if**: V11 ships A7 cleanly AND OMR is healthy (per A3 audit).

### Option C3: Darwinex-inspired FX strategy (Phase A-E of FX expansion)

- **Premise**: 50-pair FX universe, regime-adaptive mean reversion,
  Sharpe-3.64-reported strategy from r/algotrading. Six-phase validation
  pipeline already designed per memory.
- Requires V11 shipped: no, fully independent.
- Trial-chain: fresh; new strategy family, new universe, new
  infrastructure.
- Time to evidence: ~4-6 weeks because data plane (Phase A breadth,
  Phase B equity ETFs, Phase C alt data, Phase D futures, Phase E quote
  data) is the bulk of the work.
- **Best if**: V11 fails A7 OR OMR is unhealthy, redirecting investment
  away from US equities entirely.

### Option C4: Alternative signal stacks (factor moderation, ML ensembling)

- **Premise**: The closure document offers this as a generic redirect.
  No concrete spec exists.
- Requires V11 shipped: independent.
- Trial-chain: depends entirely on what's specified.
- Time to evidence: indeterminate.
- **Best if**: V11 ships A7 cleanly AND universe expansion (C1) doesn't
  produce sufficient diversification. Lowest-priority of the four
  because it has the most undefined scope.

### Option C5: Close-and-ship V11

- **Premise**: V11 paper-validates, then is promoted to live trading.
  RAMP research is paused; effort redirects to non-RAMP work (FX,
  OMR cross-strategy, etc).
- Requires V11 shipped: yes, by definition.
- Trial-chain: irrelevant (no further RAMP variants).
- Time to evidence: A7 outcome is sufficient.
- **Best if**: V11 ships A7 cleanly AND the analyst judges the +0.08
  Sharpe lift over baseline is enough deployment value to lock in
  without further iteration.

### Recommendation

The recommendation is conditional on A7's outcome, not absolute:

**If V11 clears A7 (5 clean sessions, ~mid-June):**
- Primary: C1 (universe expansion to Tier 1: S&P 500 ∪ NDX).
- Secondary parallel: C2 (RAMP-OMR portfolio construction) once OMR's
  status is confirmed.
- Defer: C3, C4, C5.

**If V11 fails A7 (2+ consecutive resets or fundamental issue):**
- Primary: investigate-and-redeploy per A2 Option 1.
- If investigation reveals V11 cannot be cleanly fixed: roll back to
  V01 baseline (A2 Option 2), then prioritize C3 (FX strategy) which is
  fully independent of RAMP.
- C5 (close-and-ship) is NOT recommended -- a failed A7 is evidence
  against shipping V11 in any form.

The decision is locked in by what A7 actually does. This document
pre-registers the rule so the post-A7 decision isn't subject to
hindsight bias.

## What I am NOT recommending

The closure document and prior session work surfaces several
temptations worth explicitly declining:

- **Do not retry WS-3d with a different ML architecture (HMM, threshold
  ensemble) without first running C1 or C3.** The closure isolated the
  failure mechanism (supervised paradigm on confirmation labels at the
  consumer threshold). Trying a different supervised architecture is
  the same failure mode in different clothing. The campaign produced
  enough evidence to stop, not enough to retry incrementally.

- **Do not propose any new V11-family variant on the v0 detector**
  before A7 outcome. Trial-chain is at 36; further variants on the same
  detector inherit a DSR threshold that cannot be cleared without
  forward OOS. Any new variant work waits on either A7 outcome or a
  fresh strategy family.

- **Do not begin C2 (RAMP-OMR portfolio) until A3 confirms OMR is
  actually deployed.** The toggle anomaly is the load-bearing question.
  Building a portfolio construction layer on a strategy that's
  accidentally disabled is wasted effort.

- **Do not delete the local branch under any circumstances.** Whatever
  archival decision is made (B1/B2/B3), the local branch is the
  authoritative source until something is on GitHub.

## Concrete sequenced tasks (next 14 days)

### Today (2026-05-25)

1. Push `v12-bear-to-cash` to GitHub as `archive/regime-detector-campaign-2026-05` (B3). 5 min.
2. Monitor A7 counter at 16:05 ET today. Log result. 5 min.
3. SSH to EC2, run `systemctl status homeguard-omr homeguard-mp homeguard-cscm`. Cross-reference with toggle yaml. Resolve discrepancy. 30 min.

### This week (2026-05-25 to 2026-05-29)

4. Daily A7 monitoring at 16:05 ET. Log counter value and any anomalies.
5. Cherry-pick `src/data/leading_indicators/` and `last_classification_timestamp` field from archive branch to main via small PR. Half day.
6. Confirm whether `regime_detector_replay.py` and `ground_truth_labelers.py` are already on main; if not, cherry-pick. 15 min.
7. Write the A2 decision rule into `docs/operations/V11_PAPER_VALIDATION.md` as a pre-registered runbook. ~30 min.

### Next week (2026-06-01 to 2026-06-05)

8. If A7 has cleared by Friday 2026-05-29 (best case): begin C1 spec drafting (Tier 1 universe expansion).
9. If A7 has reset 1 time: continue monitoring; no new work started.
10. If A7 has reset 2+ times with same root cause: open A2 Option 1 investigation.
11. If A7 has reset 3+ times with mixed root causes: open A2 Option 2 (roll back to V01).

### Week 3 (2026-06-08 onward)

12. C1 spec drafted and ready for orchestrator (~1 day to draft, ~few days to implement).
13. OR C3 (FX) spec drafted if V11 path closed.
14. OR C2 (RAMP-OMR portfolio) spec drafted if both V11 and OMR are healthy.

## Risks not addressed by the closure document

These are surfaced in this doc because they weren't explicit in prior
work:

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Local branch `v12-bear-to-cash` lost to hardware failure before push | Low | High | Push today per task 1. |
| OMR/MP strategy-toggle state inconsistent with intent | Medium | Medium | Audit via task 3 today. |
| A7 fails on day 1 due to V11 day-1 risks (empty ledger, V01-V11 plan equivalence on first rebalance) | Medium | Low | Documented in deploy log; first-day reset would be expected and informative, not failure. |
| V11 ships A7 but live deployment uncovers issues paper didn't catch | Medium | High | Standard live-vs-paper drift risk; not specific to this campaign. Address at live-promotion time, not here. |
| Closure document's "trial budget exhausted" framing carries over to non-RAMP work and discourages legitimate FX or OMR research | Low | Medium | Explicitly state in C3 / C2 specs that those families have fresh trial chains; the n_trials=36 limit is RAMP-specific. |
| The 2026-05-22 main-branch merge dropped V11-era changes that are silently relevant to A7 | Low | High | The merge resolution preserved ramp's changes per commit message. Spot-check `grid_search.py` and any V11-relevant files on main during this week. |

## Success criteria

This document succeeds if, by 2026-06-08:

1. The campaign artifacts are preserved on GitHub (B1/B2/B3 executed).
2. The strategy-toggle state is reconciled with intent (A3 resolved).
3. V11's A7 outcome is known (cleared, reset, or under investigation).
4. The next research direction has been chosen based on A7's actual
   outcome rather than pre-committed in the abstract.

It fails if the local branch is lost, if A7 fails without an actionable
diagnosis, or if research work begins on a path that A7's outcome would
have invalidated.

## Appendix: What the GitHub state inspection revealed

This document was written after inspecting the live GitHub repo via MCP
on 2026-05-25. Findings:

- `main` is at `d60686e` (2026-05-22). Most recent commit is the
  on_trial_complete callback restoration after the ramp-phase4 merge.
- `ramp-phase4-turnover-regime-research` is at `fc7de60` (2026-05-23).
  Most recent commit is the V11 production paper deploy session log.
- `v12-bear-to-cash` returns 404 from GitHub. The branch is local-only.
- The V11 deploy commit (`2cb1b7c`) confirms first rebalance fires at
  2026-05-25 (today) 15:55 ET, A7 helper at 16:05 ET.
- `config/trading/strategy_toggle.yaml` on the deploy branch shows the
  toggle anomaly described in A3.
- `docs/reports/ramp/` on the deploy branch contains V01-V11 reports
  through 20260523, but no V12+ reports (consistent with V12+ being
  local-only).
- No `RAMP_VARIANTS.md` glossary on the deploy branch (consistent with
  V12+ variants being local).

This inspection is reproducible: `gh api repos/shuyangw/Homeguard/branches`
and `gh api repos/shuyangw/Homeguard/contents/config/trading/strategy_toggle.yaml?ref=ramp-phase4-turnover-regime-research`.
