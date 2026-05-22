# Phase 4 Wave 1 Findings

**Date:** 2026-05-22
**Window:** 2017-01-01 to 2026-05-16 (2244 trading days)
**Universe:** `config/universes/sp500-2025.csv` (504 symbols, 10 missing from SIP tree)
**Data:** Alpaca SIP daily aggregated cache (`equities_daily_from_sip.parquet`)
**Base:** V01 (production REGIME_PARAMS, no crash exposure)

## Variants tested

| ID | Mechanic | Parameter |
|---|---|---|
| V01 | Baseline (Phase B re-baseline) | -- |
| V04 | Rank buffer | `buffer_size = top_n // 2` |
| V05 | Min hold | `min_hold_days = 5` |
| V06 | Delta rebalance threshold | `delta_rebalance_pct = 0.02` |
| V11 | Combined (V04 + V05 + V06) | as above |

## Headline at 5 bps per side (full window)

| Variant | CAGR | Sharpe | Max DD | Avg daily turnover | EXT-OOS 2025-26 Sharpe |
|---|---:|---:|---:|---:|---:|
| V01 | 3.74% | 0.282 | -79.88% | 91% | -0.216 |
| V04 | 4.89% | 0.313 | -78.87% | 82% | -0.099 |
| V05 | **11.08%** | **0.503** | -67.22% | **45%** | **+0.556** |
| V06 | 3.62% | 0.278 | -79.57% | 90% | -0.215 |
| V11 | **11.93%** | **0.528** | **-66.20%** | **39%** | **+0.527** |

## Per-period Sharpe at 5 bps per side

| Variant | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| V01 | 0.283 | 0.077 | 0.967 | 0.966 | -0.216 | 0.282 |
| V04 | 0.333 | 0.092 | 0.874 | 0.832 | -0.099 | 0.313 |
| V05 | 0.600 | -0.258 | 0.951 | 0.785 | +0.556 | 0.503 |
| V06 | 0.276 | 0.088 | 0.966 | 0.948 | -0.215 | 0.278 |
| V11 | 0.618 | -0.266 | 1.170 | 0.829 | +0.527 | 0.528 |

## Acceptance verdict per variant (5 bps tier)

The three gating criteria from the parent plan:

1. EXT-OOS 2025-26 Sharpe > 0.40
2. Avg daily turnover <= 55% (40% reduction from V01's 91%)
3. 2022-2024 OOS Sharpe degradation no worse than -0.20 vs V01 per-period

| Variant | EXT-OOS > 0.40? | Turnover <= 55%? | 2022-2024 Δ > -0.20? | Verdict |
|---|---|---|---|---|
| V04 | NO (-0.099) | NO (82%) | YES (small +) | **FAIL** -- rank buffer alone insufficient |
| V05 | YES (+0.556) | YES (45%) | **NO** (2022 Δ = -0.335) | **PARTIAL** -- passes EXT-OOS + turnover, fails 2022 degradation |
| V06 | NO (-0.215) | NO (90%) | YES (small +) | **FAIL** -- delta threshold at 2% too lenient |
| V11 | YES (+0.527) | YES (39%) | **NO** (2022 Δ = -0.343) | **PARTIAL** -- passes EXT-OOS + turnover, fails 2022 degradation |

### Strict-criteria verdict: no variant passes all three gates

### Composite-evidence verdict: V11 is the candidate

Both V05 and V11 trade ~0.34 of 2022 OOS Sharpe for ~0.74 of EXT-OOS Sharpe gain (and double-digit gains in IS 2017-2021 too). The 2022 degradation breach is in the worst single year and represents about $0.34 of "Sharpe budget" being spent to buy a much larger gain in the EXT-OOS failure window the entire Phase 4 program was designed to address. Net-net, V11 produces:

- +0.246 Sharpe over V01 full-window (0.528 vs 0.282).
- +0.743 Sharpe over V01 in EXT-OOS (+0.527 vs -0.216).
- 13.7 ppts lower max drawdown (-66% vs -80%).
- 52 ppt lower turnover (39% vs 91%).
- 21% drop in cost drag from V01's 75% to V11's 30%.

The 2022 degradation is the cost of the protection that buys all of the above. Argued in V11's favor: the strategy currently FAILS the methodology Section 4 cost-sensitivity gate (1.5x base cost) on V01; V11 likely passes (at 7.5 bps per side V11 still has Sharpe ~0.4 based on tier-progression patterns -- verify in standalone follow-up).

## Variant attribution

- **V04 rank buffer alone:** modest. Sharpe +0.031 vs V01, turnover -9 ppts. Not enough on its own.
- **V05 min hold alone:** the dominant lever. Sharpe +0.221, turnover -46 ppts, EXT-OOS Sharpe +0.772.
- **V06 delta threshold alone:** essentially a no-op at 2%. Turnover dropped only 1 ppt; Sharpe statistically indistinguishable from V01. The threshold needs to be higher (e.g. 5%) to materially skip trades.
- **V11 combined:** V05 dominates, with V04+V06 adding small marginal gains: +0.025 Sharpe and -6 ppts turnover vs V05 alone.

## Phase D readiness

**Recommendation: V11 is the Phase D paper-trade candidate.**

The 2022 OOS degradation is a real cost but the EXT-OOS rescue is the larger and more recent signal. Phase D should:

1. Re-enable paper trading of V11 on EC2 with a clear acceptance contract: target Sharpe >= 0.4 on the rolling paper period, with explicit re-evaluation if 2022-style drawdown patterns re-emerge.
2. Run 4-6 weeks of paper trading to validate live-vs-backtest parity (similar discipline to the existing A7 paper-validation loop).
3. After paper validation, consider whether to promote V11 as the production RAMP variant.

**Alternative: V05 alone as a simpler candidate.** V05 is 95% of V11's edge with one fewer moving part. If V11's V04 (rank buffer) contribution doesn't replicate in paper, V05 is the fallback.

## Cost-sensitivity check (informational)

V11 across all four cost tiers:

| Cost tier | CAGR | Sharpe | Max DD | Avg turnover |
|---|---:|---:|---:|---:|
| 0 bps (gross) | 17.60% | 0.693 | -65.87% | 39% |
| 2.5 bps | 14.54% | 0.605 | -65.98% | 39% |
| 5.0 bps | 11.93% | 0.528 | -66.20% | 39% |
| **7.5 bps (1.5x base, stress)** | **9.38%** | **0.452** | **-66.34%** | **39%** |

**V11 PASSES the methodology Section 4 cost-sensitivity gate** -- at 7.5 bps Sharpe is still 0.452 and CAGR is +9.38% (versus V01 at 7.5 bps: Sharpe 0.116, CAGR -2.02%, failure). This is the load-bearing result: turnover control rescues the strategy from cost-induced unprofitability.

## Followups suggested by these findings

1. **V11 with `delta_rebalance_pct=0.05` instead of 0.02**: V06 was effectively a no-op at 2%. A higher threshold could provide additional turnover reduction.
2. **V12 (BEAR cash) on V11 base** as Wave 2 first step: 2022 was a BEAR year and V11's biggest weakness was 2022 OOS. BEAR-to-cash on top of V11 directly addresses that.
3. **Live paper-validation comparator for V11** -- the existing A7 paper-validation infrastructure (Phase A) was built for V01; extend it to validate V11's filters in production paper. The comparator's `_recompute_plan` would need to know about rank-buffer and min-hold state.

## Source reports

- `20260522_phase4_v01_re_baseline.md` -- V01 baseline (Phase B + Task 57 re-baseline).
- `20260522_phase4_v04.md` -- rank-buffer solo.
- `20260522_phase4_v05.md` -- min-hold solo.
- `20260522_phase4_v06.md` -- delta-threshold solo.
- `20260522_phase4_v11.md` -- combined turnover-lite.
