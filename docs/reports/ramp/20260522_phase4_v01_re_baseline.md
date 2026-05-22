# Phase 4 V01 - Fresh portfolio every day; production REGIME_PARAMS; ignores crash exposure

## Header

- Variant: V01
- Description: Fresh portfolio every day; production REGIME_PARAMS; ignores crash exposure
- Code commit: 1548574
- Data snapshot: Alpaca SIP DuckDB Parquet
- Timing mode: near_close
- Cost tiers run: [0.0, 2.5, 5.0, 7.5]
- Universe: config\universes\sp500-2025.csv
- Known limitations: survivorship bias, daily close approximation, no point-in-time index membership

## Metrics by cost tier

### 0.0 bps per side

| Metric | Value |
|---|---:|
| CAGR | 16.36% |
| Sharpe | 0.614 |
| Max DD | -75.46% |
| Avg daily turnover | 90.83% |
| Cost drag | 0.00% |

### 0.0 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 15.00% | 4.64% | 41.99% | 39.32% | 1.14% | 16.36% |
| Sharpe | 0.572 | 0.364 | 1.476 | 1.505 | 0.169 | 0.614 |
| Max DD | -75.46% | -28.47% | -21.14% | -14.00% | -28.93% | -75.46% |
| Avg turnover | 83.03% | 103.13% | 96.79% | 100.68% | 98.89% | 90.83% |
| Cost drag | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### 2.5 bps per side

| Metric | Value |
|---|---:|
| CAGR | 9.85% |
| Sharpe | 0.448 |
| Max DD | -77.82% |
| Avg daily turnover | 90.75% |
| Cost drag | 37.72% |

### 2.5 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 9.10% | -1.92% | 33.54% | 30.72% | -4.92% | 9.85% |
| Sharpe | 0.427 | 0.220 | 1.222 | 1.234 | -0.022 | 0.448 |
| Max DD | -77.82% | -30.57% | -21.91% | -14.75% | -29.69% | -77.82% |
| Avg turnover | 82.98% | 103.02% | 96.63% | 100.55% | 98.81% | 90.75% |
| Cost drag | 34.44% | 103.44% | 19.33% | 21.45% | 11673.97% | 37.72% |

### 5.0 bps per side

| Metric | Value |
|---|---:|
| CAGR | 3.74% |
| Sharpe | 0.282 |
| Max DD | -79.88% |
| Avg daily turnover | 90.64% |
| Cost drag | 75.28% |

### 5.0 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 3.53% | -8.02% | 25.58% | 22.75% | -10.68% | 3.74% |
| Sharpe | 0.283 | 0.077 | 0.967 | 0.966 | -0.216 | 0.282 |
| Max DD | -79.88% | -32.54% | -23.08% | -15.48% | -30.48% | -79.88% |
| Avg turnover | 82.94% | 102.94% | 96.47% | 100.30% | 98.58% | 90.64% |
| Cost drag | 72.41% | 234.10% | 38.40% | 41.79% | 0.00% | 75.28% |

### 7.5 bps per side

| Metric | Value |
|---|---:|
| CAGR | -2.02% |
| Sharpe | 0.116 |
| Max DD | -81.76% |
| Avg daily turnover | 90.45% |
| Cost drag | 114.17% |

### 7.5 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | -1.71% | -13.68% | 18.04% | 15.22% | -16.08% | -2.02% |
| Sharpe | 0.139 | -0.066 | 0.711 | 0.695 | -0.411 | 0.116 |
| Max DD | -81.76% | -34.47% | -24.18% | -16.24% | -31.49% | -81.76% |
| Avg turnover | 82.88% | 102.69% | 96.05% | 99.97% | 98.20% | 90.45% |
| Cost drag | 114.47% | 400.69% | 57.24% | 61.21% | 0.00% | 114.17% |

## Regime attribution (5.0 bps tier)

| Regime | Days | Net return |
|---|---:|---:|
| BEAR | 375 | -29.66% |
| SAFE_MODE | 251 | 0.00% |
| SIDEWAYS | 398 | -26.56% |
| STRONG_BULL | 593 | 145.85% |
| UNPREDICTABLE | 40 | -80.52% |
| WEAK_BULL | 698 | 469.53% |
