# Phase 4 V03 - Target-weight-correct production; honors planner exposure_pct

## Header

- Variant: V03
- Description: Target-weight-correct production; honors planner exposure_pct
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
| CAGR | 8.79% |
| Sharpe | 0.483 |
| Max DD | -60.46% |
| Avg daily turnover | 72.94% |
| Cost drag | 0.00% |

### 0.0 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 5.95% | -1.25% | 24.40% | 39.21% | -1.84% | 8.79% |
| Sharpe | 0.365 | 0.141 | 1.633 | 1.559 | 0.022 | 0.483 |
| Max DD | -60.46% | -19.01% | -11.41% | -14.41% | -21.73% | -60.46% |
| Avg turnover | 69.76% | 56.22% | 54.86% | 99.58% | 90.45% | 72.94% |
| Cost drag | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### 2.5 bps per side

| Metric | Value |
|---|---:|
| CAGR | 3.88% |
| Sharpe | 0.281 |
| Max DD | -63.70% |
| Avg daily turnover | 72.84% |
| Cost drag | 53.60% |

### 2.5 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 1.39% | -4.70% | 20.02% | 30.76% | -7.30% | 3.88% |
| Sharpe | 0.177 | -0.010 | 1.364 | 1.281 | -0.200 | 0.281 |
| Max DD | -63.70% | -20.36% | -12.15% | -15.09% | -22.36% | -63.70% |
| Avg turnover | 69.73% | 56.08% | 54.67% | 99.40% | 90.27% | 72.84% |
| Cost drag | 74.27% | 1909.66% | 16.97% | 21.17% | 0.00% | 53.60% |

### 5.0 bps per side

| Metric | Value |
|---|---:|
| CAGR | -0.84% |
| Sharpe | 0.077 |
| Max DD | -66.76% |
| Avg daily turnover | 72.67% |
| Cost drag | 110.88% |

### 5.0 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | -3.01% | -7.95% | 15.72% | 22.71% | -12.43% | -0.84% |
| Sharpe | -0.012 | -0.158 | 1.090 | 0.999 | -0.424 | 0.077 |
| Max DD | -66.76% | -21.77% | -12.91% | -15.78% | -23.08% | -66.76% |
| Avg turnover | 69.66% | 55.86% | 54.31% | 99.11% | 89.98% | 72.67% |
| Cost drag | 162.50% | 0.00% | 34.10% | 41.43% | 0.00% | 110.88% |

### 7.5 bps per side

| Metric | Value |
|---|---:|
| CAGR | -5.18% |
| Sharpe | -0.119 |
| Max DD | -69.62% |
| Avg daily turnover | 72.50% |
| Cost drag | 171.62% |

### 7.5 bps per side -- per-period

| Metric | IS 2017-2021 | OOS 2022 | OOS 2023 | OOS 2024 | EXT-OOS 2025-26 | Full |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | -7.14% | -10.83% | 11.72% | 15.38% | -16.91% | -5.18% |
| Sharpe | -0.199 | -0.296 | 0.825 | 0.724 | -0.632 | -0.119 |
| Max DD | -69.62% | -22.90% | -13.68% | -16.46% | -27.16% | -69.62% |
| Avg turnover | 69.58% | 55.63% | 54.12% | 98.78% | 89.65% | 72.50% |
| Cost drag | 265.40% | 0.00% | 51.15% | 60.57% | 0.00% | 171.62% |

## Regime attribution (5.0 bps tier)

| Regime | Days | Net return |
|---|---:|---:|
| BEAR | 375 | -35.56% |
| SAFE_MODE | 251 | 0.00% |
| SIDEWAYS | 398 | -30.39% |
| STRONG_BULL | 593 | 74.06% |
| UNPREDICTABLE | 40 | -59.67% |
| WEAK_BULL | 698 | 193.34% |
