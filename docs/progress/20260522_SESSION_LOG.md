# Session Log — 2026-05-22

## Summary

Closed the full Phase 4 arc in a single day: re-enabled RAMP paper on EC2 (catching and fixing an A7 stale-snapshot vacuous-pass bug), hardened the Phase B harness (comparator VACUOUS exit code, per-period decomposition in reports, node_exporter textfile collector), re-baselined V01/V03 against the fresh SIP-adjusted data with per-period decomposition, then designed + implemented Phase C Wave 1 (V04/V05/V06/V11 turnover-control variants). Headline result: **V11 passes the methodology Section 4 cost-sensitivity gate that V01 fails**, with EXT-OOS 2025-26 Sharpe rescued from -0.216 to +0.527 and turnover cut from 91% to 39%. V11 is the Phase D paper-trade candidate.

## Changes Made

### Phase A operational fix
- **A7 stale-snapshot guard**: the A7 helper had been incrementing the counter daily against a stale 2026-05-18 RAMP decision snapshot because `ramp.enabled` was still `false`. Counter reached 3/5 on a false-positive trajectory. Added a snapshot-date guard so the helper only counts when the snapshot's UTC date matches today. Counter reset to 0; ramp.enabled flipped to `true`; homeguard-multi restarted.

### Phase B hardening
- **Comparator VACUOUS distinguishable from PASS**: `compare_paper_vs_plan.py` now returns exit code 3 (and `status='VACUOUS'`) when `logic_decisions` is None AND `strategy_inputs` is empty. A7 helper treats RC=3 as `[VACUOUS] counter unchanged` (no marker write, so a real later session today can still count).
- **Per-period decomposition in reports**: `build_period_decomposition_table()` added to `reports.py` with default periods IS 2017-2021 / OOS 2022 / OOS 2023 / OOS 2024 / EXT-OOS 2025-26 / Full. Every variant report now shows the breakout. This is what surfaced the V04 ranking bug below and the dominant role of EXT-OOS in V11's win.
- **node_exporter installer**: `infra/ec2/services/node-exporter.service` now includes `--collector.textfile.directory=/var/lib/node_exporter/textfile_collector`; `install_node_exporter.sh` pre-creates the dir with `ec2-user:ec2-user` ownership.

### Re-baseline against yfinance reports
- V01 + V03 re-runs at four cost tiers with per-period decomposition.
- Cross-comparison doc `20260522_phase4_re_baseline_vs_yfinance.md`: documents the yfinance-vs-SIP delta. V01 IS 2017-2021 Sharpe is 0.572 on SIP vs 0.784 on yfinance — yfinance's adjustments evidently flatter the gross numbers.

### Phase C Wave 1 — turnover-control variants
- Spec: `docs/superpowers/specs/2026-05-22-ramp-phase4-phaseC-wave1-design.md`
- Plan: `docs/superpowers/plans/2026-05-22-ramp-phase4-phaseC-wave1.md`
- New harness pieces:
  - `HarnessState`: +`position_open_dates` +`last_target_symbols`
  - `apply_trades`: threads `current_date`, maintains open_dates dict (set on 0->n, preserved on top-up, cleared on n->0)
  - `compute_trades`: widened `min_trade_value_usd` floor to `max(min_trade_value_usd, total_value * delta_rebalance_pct)`; full exits bypass
  - New `src/research/ramp_phase4/filters.py` with `rank_buffer` and `min_hold`
  - V04, V05, V06, V11 registered in REGISTRY
  - CLI auto-sets `delta_rebalance_pct=0.02` for V06 and V11
- 4 variants × 4 cost tiers × full window backtests run
- Cross-variant findings + Phase D readiness verdict written

Full Wave 1 detail in [`20260522_RAMP_PHASE4_WAVE1.md`](20260522_RAMP_PHASE4_WAVE1.md).

## Headline result table (5 bps per side, full window)

| Variant | CAGR | Sharpe | Max DD | Turnover | EXT-OOS Sharpe |
|---|---:|---:|---:|---:|---:|
| V01 baseline | 3.74% | 0.282 | -79.88% | 91% | -0.216 |
| V04 (rank buffer) | 4.89% | 0.313 | -78.87% | 82% | -0.099 |
| V05 (min hold 5d) | 11.08% | 0.503 | -67.22% | 45% | +0.556 |
| V06 (delta 2%) | 3.62% | 0.278 | -79.57% | 90% | -0.215 |
| **V11 (combined)** | **11.93%** | **0.528** | **-66.20%** | **39%** | **+0.527** |

**V11 cost-sensitivity sweep**: 0/2.5/5/7.5 bps → Sharpe 0.693/0.605/0.528/0.452. **Passes the 1.5x base-cost gate** (V01 collapses to 0.116 / -2.02% at 7.5 bps).

## Two real bugs caught this session

1. **A7 stale-snapshot vacuous-pass**: the per-day idempotency marker covered "already processed today" but not "snapshot date doesn't match today". When ramp.enabled was false and no new decision log was written, the helper kept incrementing against the prior snapshot. Fixed by adding a `SNAPSHOT_DATE == TODAY` guard before invoking the comparator. Production would have wrongly entered Task 14 production-resume in two more days.
2. **V04 ranking bug**: `_variant_v04` built `universe_ranking` from `plan.targets` only (the top_n symbols). A previously-held name that fell out of top_n had no rank entry, so the rank-buffer filter discarded every held name and V04 ran identical to V01 (turnover 90.64%, Sharpe 0.282). Caught by eyeballing the first V04 report. Fixed in `391ebea` by extending `_compute_plan_from_panel` with a `return_momentum=True` path that yields the full sorted momentum series; V04 and V11 now build ranking from all universe symbols.

The V04 bug is a testing gap worth documenting separately: unit tests on the `rank_buffer` filter all passed because the test data hand-built the `universe_ranking` correctly. There was no integration assertion of `V04_turnover < V01_turnover` on a real or synthetic backtest — the plan called for one but the implementer didn't add it. Adding such an assertion would catch this class of bug in seconds.

## Commits

Origin/ramp-phase4-turnover-regime-research moved from `72746f8` to `b9bde50` (29 commits this session).

- `53a105e` fix(ops): A7 helper guards against stale snapshots
- `799a75f` docs(ramp): mark 0.846 OOS Sharpe as gross-of-cost; add Phase 4 net-of-cost re-baseline
- `387799a` docs(progress): session findings consolidated
- `e40ea62` fix(comparator): exit code 3 for vacuous PASS
- `488641f` feat(research): per-period decomposition in variant reports
- `1548574` infra(node-exporter): enable textfile collector by default
- `636a7f7` report(ramp): re-baseline V01/V03 with per-period decomposition
- `ad3312d` docs(ramp): Phase 4 re-baseline vs existing yfinance reports
- `72746f8` docs(ramp): link Phase 4 re-baseline comparison from strategy doc
- Wave 1 build (15 commits `45ddf5a` → `578dfa8`): state fields, apply_trades+current_date, delta_rebalance_pct config, last_target_symbols, rank_buffer filter, min_hold filter, V04/V05/V06/V11 variants, CLI map
- `391ebea` fix(research): V04/V11 use full-universe momentum ranking for rank_buffer
- `569df37` report(ramp): Phase 4 Wave 1 V04/V05/V06/V11 (Alpaca SIP)
- `479b6f6` report(ramp): Phase 4 Wave 1 cross-variant findings + Phase D readiness verdict
- `b9bde50` docs(progress): Phase 4 Wave 1 session log + link findings in RAMP_STRATEGY

## Known Issues / Remaining Work

- **V11 fails the strict 2022 OOS degradation gate** (-0.343 Sharpe vs V01). Composite verdict favors V11 because the EXT-OOS rescue is twice as large, but a strict reviewer could disagree. Wave 2 V12 (BEAR-to-cash on V11 base) is the natural fix.
- **V06 at 2% is a no-op.** Worth a follow-up with `delta_rebalance_pct=0.05`.
- **A7 paper-validation comparator was built for V01.** V11 has filters the comparator's `_recompute_plan` doesn't model. Phase D paper-trade of V11 needs the comparator extended.
- **No statistical-significance gates** (PSR/DSR/PBO) applied to the Wave 1 Sharpe numbers. With 9 years of data and a multi-variant grid, full-window Sharpe 0.528 isn't necessarily significantly different from 0.282 once degrees-of-freedom corrections are applied. Worth running before committing to Phase D.
- **No `--timing one_day_lag` robustness check**. Engine supports it; never exercised. If V11's Sharpe collapses under one-day lag, the strategy has a lookahead.
- **A7 paper validation is running cleanly post-fix.** Counter at 0 with today's first legitimate session pending the 16:05 ET timer.
- **Local repo state**: `main` unchanged; ramp topic branch removed; lingering `.worktrees/phase4-followups/` files (Windows file-lock, gitignored). All commits live on `origin/ramp-phase4-turnover-regime-research` at `b9bde50`.

## Validation

- All 49+ `tests/research/ramp_phase4/` tests pass under the `fintech` env.
- V05/V11 reports independently confirm A7 helper's STALE-snapshot guard works on EC2: re-trigger with the stale 2026-05-21 snapshot produced `[STALE] snapshot date 2026-05-21 != today 2026-05-22; counter unchanged at 0`.
- V01 baseline numbers from Task 57 unchanged from Phase B's `20260519_phase4_v01.md` (sanity check: `delta_rebalance_pct=0.0` default preserves V01 behavior).
- V11 cost-sensitivity sweep at 4 tiers reports consistent turnover (39% across all tiers, varies only with the cost factor as expected).

## Phase D readiness

V11 is the candidate. To proceed:

1. Extend the A7 paper-validation comparator to model V11's filter state.
2. Run statistical-significance gates on V11 vs V01 (PSR/DSR/PBO).
3. One-day-lag robustness run for V11.
4. If all three pass: enable V11 in production paper on EC2, run 4-6 weeks of paper validation following the existing A7 discipline.
5. After paper validation: promote V11 as the production RAMP variant.

If any of (1-3) fail, the alternative is V05 alone — 95% of V11's edge with one fewer moving part and one fewer thing to model in the comparator.
