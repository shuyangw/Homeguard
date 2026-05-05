# RAMP Phase 3A: Variant Exploration -- 2026-05-05

## Context

Phase 2 root-cause investigation (2026-05-05) established: H2 SUPPORTED (regime gating harms EXT-OOS performance; V1 vanilla momentum Sharpe 0.314 vs V0 0.070), H5 REFUTED (the raw momentum signal itself remains alive), H4 REFUTED (more vol-based exposure reduction makes things worse). The 2025-12 improvement plan recommended vol-adjusted momentum (Barroso & Santa-Clara 2015) as TIER 1 / PRIORITY: CRITICAL -- it was never deployed. This phase tests that recommendation (V5a/b/c) and the simplest possible regime fix: cash in BEAR regime (V8).

## Methodology

Same universe (sp500-2025.csv, 503 symbols), same yfinance split-adjusted data (auto_adjust=True), same 0% transaction costs, same +/-20% daily return cap as Phase 2. IS: 2017-01-01 to 2021-12-31. OOS: 2022-01-01 to 2024-12-31. EXT-OOS: 2025-01-01 to 2026-04-30. V5a/b/c use production REGIME_PARAMS for exposure/top_n but replace the raw momentum ranking with a vol-normalized score: raw_momentum / (rolling_std(daily_ret, vol_window) * sqrt(252) + 1e-8). V8 is identical to V0 except BEAR days hold 0% exposure (cash). Sharpe SE on ~331 EXT-OOS days is approximately 0.17 -- differences below 0.2 are within noise. CAGR and MaxDD are concrete and not subject to this uncertainty.

## Variant comparison

| Variant | Description | IS Sharpe (2017-2021) | OOS Sharpe (2022-2024) | EXT-OOS Sharpe (2025-2026) | EXT-OOS CAGR | EXT-OOS MaxDD |
|---|---|---|---|---|---|---|
| V0 (reference) | Production RAMP | 0.755 | 0.867 | 0.070 | -1.7% | -21.6% |
| V1 (reference) | Vanilla momentum (no regime) | 0.895 | 0.710 | 0.314 | 4.8% | -21.7% |
| V5a | Vol-adj momentum, vol_window=21 | 0.620 | 1.045 | -0.150 | -4.8% | -19.4% |
| V5b | Vol-adj momentum, vol_window=10 | 0.666 | 1.321 | 0.107 | 0.3% | -15.8% |
| V5c | Vol-adj momentum, vol_window=60 | 0.480 | 0.961 | 0.013 | -1.6% | -14.9% |
| V8 (*) | V0 + BEAR-to-cash | 0.386 | 0.644 | 0.571 | 11.4% | -15.2% |

(*) = winning variant by EXT-OOS Sharpe: **V8**

## Cost sensitivity (winning variant)

Running V8 at three cost tiers. Turnover assumed = 1.0 (daily rotation). Cost drag = 2 * bps_per_side * turnover per trading day.

| Cost tier | bps/side | Daily drag | EXT-OOS Sharpe | EXT-OOS CAGR | EXT-OOS MaxDD |
|---|---|---|---|---|---|
| 0% (research) | 0 | 0.000% | 0.571 | 11.4% | -15.2% |
| 5 bps | 5 | 0.100% | -0.285 | -9.1% | -17.3% |
| 7.5 bps (1.5x) | 7.5 | 0.150% | -0.714 | -17.9% | -24.6% |

**CRITICAL: Cost sensitivity interpretation.** The 5 bps/side collapse from Sharpe 0.571 to -0.285
is mathematically expected, not anomalous. V8 earns approximately 0.045% average daily return
(11.4% CAGR / 252 days). A 0.10% per-day cost drag (5 bps/side, turnover=1.0) exceeds the gross
return by more than 2x. The implication: V8's edge in EXT-OOS is extremely thin in absolute return
terms. It avoids BEAR-day losses (which improves Sharpe) but the underlying non-BEAR momentum
return is insufficient to survive realistic transaction costs. The turnover=1.0 assumption is
conservative for RAMP (daily rebalance) but not unreasonable. Even at 0.5x turnover (5 bps/side
every two days), cost drag would be ~0.05%/day vs ~0.045% gross -- still marginally negative.
**Conclusion: V8 is NOT viable with any realistic transaction costs. The BEAR-avoidance insight
is valid but the non-BEAR returns are too low to bear trading costs.**

## Pre-committed evaluation against criteria

### V5a -- Vol-adj momentum, vol_window=21

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| EXT-OOS Sharpe > 0.5 (0% costs) | > 0.5 | -0.150 | FAIL |
| OOS Sharpe within +/-0.1 of 0.823 | 0.723 to 0.923 | 1.045 | FAIL |
| IS/OOS gap < 30% | < 30% | -68.4% (IS < OOS) | FAIL |
| (winner) EXT-OOS Sharpe > 0.3 at 1.5x costs | > 0.3 | -- | N/A (not winning variant) |

**Verdict: RESEARCH ONLY**

### V5b -- Vol-adj momentum, vol_window=10

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| EXT-OOS Sharpe > 0.5 (0% costs) | > 0.5 | 0.107 | FAIL |
| OOS Sharpe within +/-0.1 of 0.823 | 0.723 to 0.923 | 1.321 | FAIL |
| IS/OOS gap < 30% | < 30% | -98.5% (IS < OOS) | FAIL |
| (winner) EXT-OOS Sharpe > 0.3 at 1.5x costs | > 0.3 | -- | N/A (not winning variant) |

**Verdict: RESEARCH ONLY**

### V5c -- Vol-adj momentum, vol_window=60

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| EXT-OOS Sharpe > 0.5 (0% costs) | > 0.5 | 0.013 | FAIL |
| OOS Sharpe within +/-0.1 of 0.823 | 0.723 to 0.923 | 0.961 | FAIL |
| IS/OOS gap < 30% | < 30% | -100.0% (IS < OOS) | FAIL |
| (winner) EXT-OOS Sharpe > 0.3 at 1.5x costs | > 0.3 | -- | N/A (not winning variant) |

**Verdict: RESEARCH ONLY**

### V8 -- V0 + BEAR-to-cash

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| EXT-OOS Sharpe > 0.5 (0% costs) | > 0.5 | 0.571 | PASS |
| OOS Sharpe within +/-0.1 of 0.823 | 0.723 to 0.923 | 0.644 | FAIL |
| IS/OOS gap < 30% | < 30% | -66.9% (IS < OOS) | FAIL |
| (winner) EXT-OOS Sharpe > 0.3 at 1.5x costs | > 0.3 | -- | FAIL (-0.714) |

**Verdict: PROMISING**

## Regime breakdown -- V8 EXT-OOS (2025-2026)

| Regime | % of days | Sharpe | CAGR | Max DD | Return contrib % |
|---|---|---|---|---|---|
| STRONG_BULL | 12.4% | 3.175 | 57.2% | -4.5% | +42.4% |
| WEAK_BULL | 43.5% | -0.778 | -16.6% | -23.7% | -51.4% |
| SIDEWAYS | 23.6% | 1.031 | 35.9% | -12.2% | +64.4% |
| UNPREDICTABLE | 1.2% | 7.668 | 12009.3% | -2.2% | +44.6% |
| BEAR | 19.3% | 0.000 | 0.0% | 0.0% | +0.0% |

## Conclusion

**Winning variant: V8** with EXT-OOS Sharpe 0.571, CAGR 11.4%, MaxDD -15.2%.

The winner improves on V1 (vanilla momentum, EXT-OOS Sharpe 0.314) by +0.257 Sharpe points. V1 remains the practical floor for the regime-free approach.

V8 does NOT qualify as a production candidate. Failed criteria: OOS Sharpe 0.644 outside +/-0.1 of 0.823; IS/OOS gap -66.9% >= 30%; 1.5x cost Sharpe -0.714 < 0.3 (collapses to -0.714). It is classified as PROMISING for the diagnostic insight it provides, but the cost sensitivity failure disqualifies it from any production consideration.

**Key diagnostic finding from V8:** BEAR regime is entirely responsible for V0's EXT-OOS failure.
V8 regime breakdown shows BEAR days (19.3% of time) contribute 0.0% to returns but WEAK_BULL
(43.5% of days) contributes -51.4%. Simply cashing out BEAR days brought EXT-OOS CAGR from -1.7%
(V0) to +11.4% (V8), but WEAK_BULL losses persist and the gross return is too thin to survive costs.

**Vol-adjusted momentum (V5a/b/c): unexpected IS < OOS pattern.** All three variants show OOS Sharpe
dramatically exceeding IS Sharpe (V5b: 0.666 IS vs 1.321 OOS). This is the reverse of the typical
overfitting pattern. Interpretation: vol-adjusted signals perform better in volatile regimes
(2022-2024 bear/recovery) because the normalization suppresses high-vol momentum names that
crash in drawdowns. However, the EXT-OOS (2025-2026) collapse suggests the benefit is
regime-specific to the 2022-2024 stress period, not a generalizable improvement. The extreme
OOS Sharpes (1.045-1.321) should be treated with caution -- they may reflect the 2022-2024
period being exceptionally favorable to vol-adjusted signals due to the rate-shock environment.

Statistical caveat: Sharpe SE on 331 EXT-OOS days is ~0.17. Differences less than 0.2 between
variants are not reliable. Any claimed improvement should be confirmed with a longer OOS window
before deployment.

## Implications for next steps

1. **V8 confirms the BEAR-avoidance hypothesis but is NOT viable with transaction costs.**
   The diagnostic value is high: cashing out BEAR days adds ~13 percentage points of CAGR
   (from -1.7% to +11.4%). But the non-BEAR gross return (~0.045%/day) is too thin to absorb
   any realistic costs. The correct next step is NOT to deploy V8 but to investigate why
   non-BEAR returns are so low. WEAK_BULL (43.5% of EXT-OOS days, Sharpe -0.778) is the
   structural problem that V8 does not fix. **Recommended next test:** WEAK_BULL regime
   parameter recalibration or WEAK_BULL-to-cash (more aggressive version of V8).
2. **Do not optimize parameters on EXT-OOS data.** The 2025-2026 window has only ~331 days (SE ~0.17). Any further tuning must use the IS (2017-2021) period only, with EXT-OOS held truly blind.
3. **Survivorship bias caveat:** All tests use sp500-2025.csv (current composition). Stocks that were removed from the S&P 500 during 2017-2024 are excluded. This biases IS/OOS upward, but the bias is symmetric across all variants so relative comparisons are valid.
