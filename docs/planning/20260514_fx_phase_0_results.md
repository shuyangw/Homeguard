# FX Phase 0 Results — 2026-05-14

Probes executed against Massive S3 flat-files (`https://files.massive.com`, bucket `flatfiles`) using `MASSIVE_S3_ACCESS_KEY`/`MASSIVE_S3_SECRET_KEY` (Currencies Starter tier).

## Probe 1: S3 bucket inventory

Script: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\probe\fx_phase_0_bucket_inventory.py`

| Prefix | Accessible | Children | Notes |
|---|---|---|---|
| `global_forex/minute_aggs_v1/` | ✅ True | 18 (2009-2026) | Baseline (already in use) |
| **`global_forex/quotes_v1/`** | **✅ True** | **18 (2009-2026)** | **DECISION GATE 1 CLEAR — Phase B can proceed at current tier** |
| `global_forex/trades_v1/` | ✅ True | 0 | Empty bucket (confirms no FX trade tape exists) |
| `global_forex/day_aggs_v1/` | ✅ True | 18 (2009-2026) | Daily aggregates available; not currently needed |
| `us_indices/` | ✅ True | 3 (day_aggs_v1, minute_aggs_v1, values_v1) | DXY-relevant; Phase D candidate |
| `us_treasuries/` | ✅ True | 0 | **Empty** — Phase E uses FRED via `pandas-datareader` instead |
| `us_stocks_sip/` | ✅ True | 4 (day/minute/quotes/trades) | Already in use elsewhere |
| `us_options_opra/` | ✅ True | 4 (day/minute/quotes/trades) | Already in use elsewhere |
| `us_futures_{cme,cbot,comex,nymex}/` | ✅ True (all 4) | 4 each (minute/quotes/session/trades) | Phase C uses Databento; not Massive |

**Verdict**: all 12 probed prefixes accessible. `global_forex/quotes_v1/` access confirms Phase B can execute at current $49/mo Currencies Starter tier with no upgrade required.

## Probe 2: Quote data schema (from docs, not size-sampled)

Schema per `https://massive.com/docs/flat-files/forex/quotes`:

| Column | Type | Units | Description |
|---|---|---|---|
| `ticker` | string | — | Trading symbol (likely `C:EUR-USD` per minute-aggs convention) |
| `participant_timestamp` | integer | nanoseconds | Exchange-side timestamp |
| `bid_price` | number | currency units | Best bid |
| `ask_price` | number | currency units | Best ask |
| `bid_exchange` | integer | — | Source ECN for bid quote |
| `ask_exchange` | integer | — | Source ECN for ask quote |

**Plan correction**: source plan assumed 8 columns including `sip_timestamp` and `bid_size`/`ask_size`. Actual schema is 6 columns. FX quotes do NOT carry sizes (consistent with OTC market structure). One `exchange` field is actually two (`bid_exchange`, `ask_exchange`) — useful for routing analysis.

**Storage implication**: simpler schema = smaller rows. Plan's 300-500 GB estimate for full universe should be revised DOWN, probably to 200-350 GB range. To confirm, Phase B's Tier 1 (5 pairs) will provide empirical sizing before committing to remaining tiers.

Quote data size estimation deferred: a sample download wasn't run since schema confirmation from docs answers the architecture question. Tier 1 of Phase B will perform empirical sizing as its first step.

## Probe 3: NZDCHF outlier investigation

Script: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\probe\nzdchf_outlier_investigation.py`. Tested Dec 2025 NZDCHF vs NZDUSD × USDCHF.

**Baseline**: n=30,872 bars, mean=2.89 bps, std=11.33 bps, outliers >50bps = 0.99%.

### Hypothesis 1: Stale-bar lag — **REJECTED**

| Lag (min) | std (bps) | outliers (%) |
|---|---|---|
| -3 | 11.59 | 1.08 |
| -2 | 11.52 | 1.06 |
| -1 | 11.42 | 1.01 |
| **0** | **11.33** | **0.99** |
| +1 | 11.44 | 1.03 |
| +2 | 11.53 | 1.06 |
| +3 | 11.57 | 1.13 |

Best lag is 0 (i.e., no lag). Stale-bar timing is NOT the cause.

### Hypothesis 2: Bad-tick noise — **CONFIRMED**

60-minute rolling MAD filter at 6× threshold:
- Drops 7.17% of bars
- Reduces std from 11.33 → 3.82 bps
- **Reduces outliers from 0.99% → 0.00%**

### Hypothesis 3: Asia-session thinness — **PARTIAL (NY close, not Asia)**

Outlier rate by UTC hour (Dec 2025):

| Hour (UTC) | Outlier rate |
|---|---|
| 00-03 (Asia open) | 0.00% |
| 04 | 0.08% |
| 07-14 (Europe) | clean |
| 19 (NY close starting) | 1.17% |
| **20 (NY close)** | **1.83%** |
| 21-22 (NY close end) | 0.10-0.16% |
| 23 (Asia open transition) | 0.00% |

Outliers concentrate at **NY close (19:00-20:00 UTC)**, not Asia. Likely thin-liquidity end-of-day prints during the brief gap when only late NY desks are quoting.

### Verdict

**Bad-tick noise at NY-close microstructure**. Recommended action: implement 60-minute rolling MAD filter (6× threshold) as a validation-layer outlier flagger. The MAD filter drops 7% of bars but eliminates outliers. For strategy backtests, the filter should be applied at the data-load layer; for raw research, document the elevated outlier rate at NY-close UTC 19-20.

## Probe 4: Massive pricing / docs

- Currencies Basic: quotes NOT included
- **Currencies Starter (current tier, $49/mo): quotes included ✓**
- Currencies Business: quotes included
- Historical depth on Starter+: back to 2009-09-25 (same as minute aggs)

**Subscription verdict**: **stay at Starter**. No upgrade required for any phase in this plan.

## DECISION GATE 1 — Status

**CLEAR**. Phase B can execute at current Starter tier without subscription upgrade.

## DECISION GATE 2 (pre-Phase-B-execution)

Still applies (YAGNI recheck before committing 200-350 GB on speculative quote data). Surfaces when Phase B implementation plan gets written (post Phases A/D/E/C completion).

## Summary for downstream phases

| Phase | Action enabled by Phase 0 |
|---|---|
| A | Proceed; standard density-probe pattern applies |
| D | Proceed; `us_indices/` available if DXY needed later |
| E | Proceed; `us_treasuries/` is empty so FRED via pandas-datareader stays the path |
| C | Proceed; Databento continues to handle CME FX futures |
| B | **Proceed when triggered**; subscription is sufficient; expect ~200-350 GB (revised down from 300-500); MAD filter is part of quality pipeline |
| F | Still deferred |
