# Homeguard Session Findings — 2026-05-19

## Executive summary

Three workstreams shipped in one session. The headline finding is from RAMP Phase 4 Phase B:

**The existing RAMP backtest reports (re-eval 2026-05-04, root-cause 2026-05-05, Phase 3A/3B variant exploration) were computed against UNADJUSTED close prices and report 0%-cost gross Sharpe as if it were a tradeable number.** When the same strategy is re-evaluated on split-adjusted Alpaca SIP data with realistic transaction costs, the headline 0.846 walk-forward Sharpe collapses to 0.282 at 5 bps/side. The strategy has a real but modest gross edge (Sharpe 0.614 at 0 bps) that gets eaten by ~91% daily turnover.

Phase 4's hypothesis — that turnover control is the gating issue — is now empirically confirmed. Wave 1 turnover-control variants (V04-V11) are no longer "an optimization on top of V03"; they are a precondition for any net-positive RAMP variant.

The other two workstreams (Grafana backfill, CSCM scheduling fix) are operational improvements that landed cleanly.

---

## Workstream 1: Grafana Dashboard Gap Backfill

**Goal:** Make the `portfolio_overview` Grafana panels show continuous, interpretable data after trading-instance downtime. Aesthetic goal only — no analytical correctness claims on historical equity.

**Delivered:**

| Artifact | Status |
|---|---|
| `scripts/ops/backfill_regime_state.py` | done — re-runnable regime state recomputation from Alpaca SPY+VIX history |
| `scripts/ops/sync_grafana_annotations.py` | done — idempotent JSON → Grafana annotations sync |
| `config/monitoring/grafana/dashboards/portfolio_overview.json` | edited — tagged annotation source for `instance-off` windows |
| `config/monitoring/grafana/annotations/instance_off.json` | seeded with the visible 2026-04-16 → 2026-04-24 gap |
| `docs/runbooks/backfill_regime_state.md` | written |
| `.env.example` | added `GRAFANA_API_KEY` placeholder |

**Reference:** [`docs/progress/20260516_GRAFANA_GAP_BACKFILL.md`](20260516_GRAFANA_GAP_BACKFILL.md)

**Commit range on origin/main:** `da3d94b..856fd36` (15 commits, pushed)

**Notes:**

- The equity / drawdown / day-P&L panels are NOT backfilled. Three formula eras in VictoriaMetrics make destructive rebuild risky. Annotations are the chosen overlay.
- Regime state is backfillable because it's a pure function of public SPY + VIX data through the production regime detector.

---

## Workstream 2: CSCM Missed Rebalance Diagnosis + Fix

**Trigger:** User asked when CSCM last rebalanced. Investigation showed CSCM **missed** the 2026-05-17 Sunday rebalance window.

**Root cause:** Hourly polling phase-locked to bot startup time. When EC2 is started outside the scheduled Sat 23:00 UTC window (e.g., manually at Sat 02:30 UTC), the next Sunday-check time can fall AFTER the EventBridge stop at Sun 00:10 UTC, so the rebalance check never fires.

| Date | EC2 boot | Next Sun check after boot | EC2 stop | Result |
|---|---|---|---|---|
| 2026-05-10 | Sat 23:00:53 UTC (scheduled) | Sun 00:00:53 UTC | Sun 00:11:22 | rebalance fired @ 00:01:18 |
| **2026-05-17** | Sat 02:30:59 UTC (manual) | Sun **00:30:59 UTC** | Sun 00:12:02 | killed 20 min before next check |

**Fix delivered (`856fd36`):**

- `scripts/trading/run_cscm_live.py`: argparse `--check-interval` accepts `float` (was `int`).
- `infra/ec2/services/homeguard-cscm.service`: `ExecStart` adds `--check-interval 0.0833` (5 min).
- Deployed to EC2 the same session; restarted and verified.

**Reference:** [`docs/progress/20260519_CSCM_REBALANCE_DIAGNOSIS.md`](20260519_CSCM_REBALANCE_DIAGNOSIS.md)

**Side notes:**

- CSCM is in demo broker mode (`CSCM_USE_DEMO_BROKER=true`), so the missed rebalance had no real-money impact. Demo positions only.
- Real-world test of the fix: next CSCM rebalance window is Sat 2026-05-23 23:00 UTC → Sun 2026-05-24 00:10 UTC.

---

## Workstream 3: RAMP Phase 4

Two sub-phases. Both shipped to `origin/ramp-phase4-turnover-regime-research`.

### Phase A: Ops Redesign (Tasks 13 + 14 from the parent plan)

The Phase A *code* deliverables (F1 planner, F2 target-aware execution, F3 parity tests, F4 safe mode, F5 decision-log enrichment) were already complete on the branch from the 2026-05-15 session. This session redesigned the operational sequence for paper validation + production resume.

**Delivered (`origin/ramp` commits `c97c5d2`...`ddcd903`, 11 commits):**

| Artifact | Status |
|---|---|
| `scripts/ops/check_ramp_paper_session.sh` | rewritten — EC2-resident; counter at `/var/lib/homeguard/a7_clean_sessions`; emits VM gauge `hg_a7_clean_sessions`; per-day idempotency via marker file |
| `infra/ec2/services/homeguard-ramp-paper-check.service` + `.timer` | new — systemd timer fires Mon-Fri 16:05 ET (~5 min after RAMP rebalance closes) |
| `docs/progress/20260515_RAMP_PHASE4_DEPLOY_RUNBOOK.md` | updated — EC2-resident procedure, clean-session semantics table, rollback paths |
| `scripts/trading/compare_paper_vs_plan.py` | hardened — defensive `None` handling for `logic_decisions`; module-style invocation so `src.*` imports resolve under the systemd unit's working dir |

**Deploy status:**

- Installed on EC2 (`100.30.95.146`); timer scheduled at next 16:05 ET trigger.
- Smoke trigger PASSED on the 2026-05-18 RAMP decision snapshot.
- node_exporter `--collector.textfile.directory=/var/lib/node_exporter/textfile_collector` flag added inline; `infra/ec2/setup/install_node_exporter.sh` should be updated for future provisions.
- Counter reset to 0; real 5-session validation pending (next session: 2026-05-19 20:05 UTC if RAMP is re-enabled in `strategy_toggle.yaml`).

**Wall-clock-only remaining items:**

- Re-enable `ramp.enabled: true` in EC2's `config/trading/strategy_toggle.yaml`.
- Wait for 5 consecutive clean paper sessions (`hg_a7_clean_sessions == 5`).
- Run `bash scripts/ops/ramp_phase4_close_progress_doc.sh` to resume production.

### Phase B: Research Harness + V01/V03 (Critical Findings)

**Delivered (`origin/ramp` commits `b75a216`...`24eb8b0`, 23 commits):**

Module tree under `src/research/ramp_phase4/`:

| Module | Lines | Responsibility |
|---|---:|---|
| `config.py` | ~30 | `HarnessConfig` frozen dataclass |
| `data.py` | ~140 | Alpaca SIP daily loader + 1-min→daily aggregator with disk cache |
| `costs.py` | ~30 | flat-bps cost model |
| `metrics.py` | ~130 | Sharpe, CAGR, max DD, turnover, cost drag, regime attribution |
| `engine.py` | ~220 | Stateful target-weight backtest loop (MTM, trades, costs, regime) |
| `variants.py` | ~110 | `VariantSpec` registry + V01 + V03 plan_fns delegating to F1 planner |
| `reports.py` | ~110 | Markdown report builder per-variant + V01-vs-V03 parity |

Plus `scripts/backtest_scripts/ramp_phase4_backtest.py` CLI and `_make_parity_report.py` one-shot.

`src/strategies/advanced/market_regime_detector.py`: added `last_regime_scores` attribute (one-line addition, additive; the existing F5 decision log already needed this).

41 unit + integration tests across `tests/research/ramp_phase4/` (and `tests/strategies/test_market_regime_detector.py`). All passing under `fintech` env.

---

## RAMP Phase 4 Research Findings (the headline)

### Finding 1: Stale daily cache stored UNADJUSTED prices

The first round of V01 + V03 on `H:\Stock_Data\equities_daily_cache.parquet` (2017-01-03 → 2025-12-04) produced absurdly good results:

| Metric (5 bps) | V01 (stale, unadjusted) | V03 (stale, unadjusted) |
|---|---:|---:|
| CAGR | 129.22% | 95.04% |
| Sharpe | 0.554 | 0.620 |
| Max DD | -66.84% | -44.43% |
| Cost drag | 33% | 30% |

These numbers were **artifacts of unadjusted closes**. On stock-split days, the strategy saw a ~70% one-day "loss" and the post-split "discount" became a strong momentum buy signal:

- 2020-08-31 AAPL 4-for-1 split: $499.23 → $129.04 in the unadjusted data
- 2022-08-25 TSLA 3-for-1 split: similar gap

Dozens of such splits across the SP500 universe over 2017-2026 fed the strategy phantom alpha. The "129% CAGR" was largely buying-the-split-discount.

### Finding 2: Fresh split-adjusted SIP data produces the real picture

The data loader was reworked to aggregate `H:\Stock_Data\equities_1min_sip_split/` (the user's recent 24h SIP redownload — split-adjusted, current through 2026-05-16) to daily closes. Cache built at `H:\Stock_Data\cache\ramp_phase4\equities_daily_from_sip.parquet` (1.1M rows, 504 symbols, 4.8 min cold build, ~1s warm).

Re-run on split-adjusted data (2017-01-01 → 2026-05-16):

| Cost tier | V01 CAGR | V01 Sharpe | V01 Max DD | V01 cost drag |
|---|---:|---:|---:|---:|
| **0 bps (gross)** | 16.36% | **0.614** | -75.46% | 0% |
| 2.5 bps | 9.85% | 0.448 | -77.82% | 38% |
| 5.0 bps | 3.74% | 0.282 | -79.88% | **75%** |
| 7.5 bps | -2.02% | 0.116 | -81.76% | 114% |

| Cost tier | V03 CAGR | V03 Sharpe | V03 Max DD | V03 cost drag |
|---|---:|---:|---:|---:|
| **0 bps (gross)** | ~9% | ~0.46 | -55% | 0% |
| 5.0 bps | -0.84% | 0.077 | -66.76% | **111%** |

**Per-regime attribution (V01 @ 5 bps, full 2017-2026 window):**

| Regime | Days | Net return |
|---|---:|---:|
| STRONG_BULL | 593 | +145.85% |
| WEAK_BULL | 698 | +469.53% |
| SIDEWAYS | 398 | -26.56% |
| BEAR | 375 | -29.66% |
| UNPREDICTABLE | 40 | -80.52% |
| SAFE_MODE | 251 | 0% |

The strategy's edge is concentrated in BULL regimes. SIDEWAYS, BEAR, and UNPREDICTABLE are net drags. This is consistent with the Phase 3A finding that BEAR-to-cash improves gross.

### Finding 3: V01 vs V03 parity conclusion FLIPPED

**Earlier finding (on stale unadjusted data):** Option 1 — V03 wins net. Wrong.

**Corrected finding (on fresh adjusted data):** Option 2 — V03 wins gross drawdown but LOSES net Sharpe AND net CAGR to V01.

V03 halves gross during crash regimes (intended). But its turnover only drops ~20%, while its gross drops ~50%. Net effect: cost drag on V03 explodes past 100% of gross. V03 is a worse strategy net of cost than V01, even though V03 has lower max drawdown.

### Finding 4: Walk-forward "baseline" was misleadingly optimistic

The pre-Phase-4 RAMP reports were:

| Report | Date | Reported Sharpe | Cost model | Turnover model |
|---|---|---|---|---|
| Walk-forward validation (yfinance) | 2025-12-12 | 0.846 OOS | 0% | none (fresh portfolio daily) |
| Re-evaluation (yfinance) | 2026-05-04 | 0.823 OOS 2022-2024, 0.074 EXT-OOS | 0% | none |
| Root-cause (yfinance) | 2026-05-05 | various | 0% | none |
| Phase 3A variant exploration | 2026-05-05 | various | 0/5/7.5 bps | partial |
| Phase 3B BEAR optimizer | 2026-05-05 | various | 0/5/7.5 bps | partial |

The 2025-12-12 walk-forward "validation Sharpe 0.846" was computed at 0% transaction costs with no real turnover state. The proper-cost stateful equivalent on split-adjusted SIP is **0.282**. That is not a typo. The headline number used to argue RAMP's production-readiness was **gross**, not tradeable.

Phase 3A and 3B did model costs, but they ran against yfinance data which has different split-adjustment behavior than SIP. Worth re-running them on the new harness to verify their conclusions still hold.

### Finding 5: Phase C priority must re-order

The original Phase 4 plan's Section 11 recommended priority was:

1. F1-F6 (Phase A code + Phase B harness) — done
2. V01, V02, V03 (Wave 0 baselines) — V01 + V03 done; V02 (vanilla momentum) pending
3. V04, V05, V06, V11 (cost-control) — pending Phase C
4. V12, V16 (BEAR / WEAK_BULL drags) — pending Phase C
5. V21, V24 (regime thrash, breadth) — pending Phase C
6. V26, V28 (signal stability) — pending Phase C

With the corrected V01 numbers, the order should change:

1. **V11 combined turnover-lite is the new gating test.** If V11 can drop turnover from ~91% to ~25-40% on top of V01 while preserving gross Sharpe, RAMP becomes net-viable.
2. Only AFTER V11 succeeds does it make sense to layer V12 (BEAR cash) or V16 (WEAK_BULL exposure reduction) on top.
3. V02 (vanilla momentum, no regime) becomes interesting as a comparison: if vanilla beats V01 net, the regime overlay is destroying value.

---

## Open items / follow-up work

### Operational / wall-clock

- **A7 paper validation loop:** counter at 0. Re-enable `ramp.enabled: true` in EC2's `strategy_toggle.yaml`, then 5 consecutive clean Mon-Fri sessions land via the new timer. Task 14 production resume follows.
- **Push origin/main:** Local `main` was at `856fd36` (grafana). The user pushed origin/main during the session. Sync local checkout when convenient.
- **Push user's local ramp branch:** `7ea9bd7` (local) vs `24eb8b0` (origin/ramp). Rebase or merge after `git fetch`.

### Code / infra hardening

- `compare_paper_vs_plan.py` still calls `_recompute_plan(strategy_inputs)` even with empty inputs. Empty-input case produces a vacuous PASS — fine for SAFE_MODE days, worth tightening so a vacuous PASS is distinguishable from a real PASS.
- A7 helper does not check that the snapshot's decision timestamp matches today's UTC date. If RAMP doesn't fire today but yesterday's snapshot lingers in `_latest/`, the helper would re-process it. Marker file prevents that within a calendar day but not across days. Add a `decision_ts.date() == today` guard.
- `infra/ec2/setup/install_node_exporter.sh` needs to include `--collector.textfile.directory=/var/lib/node_exporter/textfile_collector` for future EC2 provisions.
- `src/research/ramp_phase4/data.py` aggregates from `equities_1min_sip_split/` at module-init time on cache miss. Worth adding a CLI to force-rebuild the cache (currently inferred from end_date vs cache max date).
- VIX is fetched via yfinance. This is a deliberate exception to the "yfinance is last-resort" rule: Alpaca SIP doesn't carry indices (VIX is a CBOE index, not an equity), and `src/utils/vix_provider.py` already uses yfinance as the project's canonical VIX source. yfinance for VIX is explicitly OK; yfinance for equity bars remains forbidden.
- 10 SP500 symbols missing from SIP tree (BRK.B, MMC, FI, K, CTRA, HOLX, BF.B, DAY, WBA, IPG); engine NaN-force-exits them. Worth investigating whether these are missing from the SIP feed itself or just from this slice.

### Research / Phase C planning

- **Validate Phase 3A and 3B findings on the new harness** before treating their recommendations as load-bearing. The unadjusted-data bug likely affected them too if they used yfinance with the older walk-forward script.
- **Re-baseline existing reports' headline numbers.** The 0.846 walk-forward Sharpe should be marked "gross, 0% cost, no turnover accounting" in the strategy docs; the proper net Sharpe is 0.282.
- **Write Phase C spec for Wave 1 turnover-control variants** (V04 rank buffer, V05 min-hold, V06 delta threshold, V11 combined). The harness is ready; just need the variant `plan_fn` implementations.
- **V01/V03 reports only show single-window metrics** (full 2017-2026), not per-period IS / OOS-2022 / OOS-2023 / OOS-2024 / EXT-OOS decomposition. Adding period-split tables to `reports.py` is a small follow-up that would help validate the harness against the existing yfinance reports.

### Documentation

- This session's per-area logs are scattered: `20260516_GRAFANA_GAP_BACKFILL.md`, `20260519_CSCM_REBALANCE_DIAGNOSIS.md`, `20260519_RAMP_PHASE4_phaseB_data_loader_fix.md`, `20260519_RAMP_PHASE4_phaseB_fresh_sip.md`, plus this consolidated doc. Together they should be enough for the next session to pick up.
- `docs/strategies/production/RAMP_STRATEGY.md` should be updated to reflect the corrected gross-vs-net numbers. The "0.846 OOS Sharpe" claim needs context.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` directories contain the spec + plan files but those are gitignored. Spec: `2026-05-19-ramp-phase4-phaseB-design.md`. Plans: `2026-05-19-ramp-phase4-phaseA-ops.md`, `2026-05-19-ramp-phase4-phaseB-harness.md`.

---

## Commits and branches

### `origin/main` movement during this session

- Was: `da3d94b` (Alpaca SIP universe snapshot)
- Now: `856fd36` (Grafana backfill + CSCM 5-min interval fix)
- 15 commits, pushed.

### `origin/ramp-phase4-turnover-regime-research` movement

- Was: `5d9a6ea` (Phase A code complete, deploy prep)
- Now: `24eb8b0` (Phase B harness + V01/V03 + parity + data-fresh recovery)
- 32 commits across Phase A ops + Phase B + data-fresh, pushed.

### Outstanding local-vs-origin

- Local `main`: at `856fd36`. Same as origin (no divergence).
- Local `ramp-phase4-turnover-regime-research`: at `7ea9bd7` (one local commit on top of `5d9a6ea`). Origin advanced significantly past this. Rebase or merge required.

### Key research-finding commits to bookmark

- `743c169` `fix(research): aggregate 1-min SIP to daily for fresh end-of-window coverage` — the data loader rework.
- `391ff35` `report(ramp): Phase 4 V01/V03 + parity on FRESH SIP daily (2017-01-01 to 2026-05-16)` — the corrected numbers.
- `24eb8b0` `docs(progress): Phase B data-fresh recovery session log` — the full per-task narrative.

### Reports

- `docs/reports/ramp/20260519_phase4_v01.md` — V01 baseline, fresh SIP, 4 cost tiers.
- `docs/reports/ramp/20260519_phase4_v03.md` — V03 baseline, fresh SIP, 4 cost tiers.
- `docs/reports/ramp/20260519_phase4_v01_vs_v03_parity.md` — V01 vs V03 parity finding, Option 2 selected.

(All pushed to `origin/ramp-phase4-turnover-regime-research` at `e8cec3b` and `24eb8b0`. `docs/reports/` is gitignored locally so they live on the remote.)

---

## Bottom line

The session's largest deliverable was not the code. It was the discovery that the existing RAMP backtest baseline (`0.846 OOS Sharpe`, used to argue production-readiness) was **gross of cost on unadjusted data**. With proper split-adjusted SIP data and stateful cost accounting, the same strategy delivers 0.282 net Sharpe. The Phase 4 plan's hypothesis that turnover control is the gating issue is correct. Wave 1 turnover-control variants are now the gating priority for any future Phase C work; not "an optimization on top of V03 base."

The user's investment in re-downloading SIP data was the enabler of this finding. Without that fresh data, this session would have produced a wrong conclusion.
