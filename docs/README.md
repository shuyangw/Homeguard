# Homeguard Documentation Hub

**Welcome to the Homeguard Backtesting Framework documentation.**

This documentation is organized by topic for easy navigation. Select the category below that matches your needs.

---

## Architecture (Understanding the System)

**Start here to understand how Homeguard is built and how components interact.**

- [**Architecture Overview**](architecture/ARCHITECTURE_OVERVIEW.md) - [+] High-level system design and 5-layer architecture
- [**Module Reference**](architecture/MODULE_REFERENCE.md) - [+] Detailed module-by-module reference (60+ modules)
- [**Data Flow**](architecture/DATA_FLOW.md) - [+] Data flow diagrams and execution paths

**When to read**: Before modifying core engine code or adding new features

---

## User Guides (How to Use Homeguard)

**Practical guides for using the backtesting framework.**

### Getting Started
- [**Backtesting Guide**](guides/BACKTESTING_GUIDE.md) - Main user guide for running backtests
- [**Backtesting README**](guides/BACKTESTING_README.md) - Quick start guide

### Advanced Topics
- [**Advanced Strategies Guide**](guides/ADVANCED_STRATEGIES_GUIDE.md) - Creating complex trading strategies
- [**Risk Management Guide**](guides/RISK_MANAGEMENT_GUIDE.md) - Position sizing, stop losses, portfolio constraints
- [**Position Sizing Methods**](guides/POSITION_SIZING_METHODS.md) - 5 position sizing algorithms explained
- [**Data Ingestion Pipeline**](guides/DATA_INGESTION_PIPELINE.md) - Fetching and storing market data
- [**Caching Guide**](guides/CACHING_GUIDE.md) - Performance optimization
- [**Make Commands**](guides/MAKE_COMMANDS.md) - Makefile usage reference

### Regime Analysis & Advanced Validation
- [**Regime Analysis User Guide**](guides/REGIME_ANALYSIS_USER_GUIDE.md) - User-facing guide
- [**Regime-Based Testing Architecture**](architecture/REGIME_BASED_TESTING.md) - Technical design and algorithms

**What it does**: Prevent overfitting and assess strategy robustness across different market conditions (bull/bear, high/low volatility, drawdown phases)

**When to read**: Before deploying strategies to production, when validating robustness

---

## API Reference (Technical Documentation)

**Detailed API documentation for developers.**

- [**API Reference**](api/API_REFERENCE.md) - Main API documentation
- [**API Reference (Sweep)**](api/API_REFERENCE_SWEEP.md) - Multi-symbol sweep functionality
- [**Sweep Statistics Guide**](api/SWEEP_STATISTICS_GUIDE.md) - Understanding sweep metrics

**When to read**: When integrating Homeguard into your code or automating backtests

---

## [+] Testing & Validation (Quality Assurance)

**Comprehensive testing documentation - our accuracy test suite validates engine correctness.**

### Quick Start
- [**Test Suite Quick Start**](testing/TEST_SUITE_QUICK_START.md) - [*] **START HERE** - Run 50 tests in 5 seconds

### Test Plans & Results
- [**Backtest Accuracy Test Plan**](testing/BACKTEST_ACCURACY_TEST_PLAN.md) - Master test plan (reference)
- [**Backtest Accuracy Tests Complete**](testing/BACKTEST_ACCURACY_TESTS_COMPLETE.md) - [+] Final results (50/50 tests passing)

### Deep Dives
- [**Lookahead Bias Analysis**](testing/LOOKAHEAD_BIAS_ANALYSIS.md) - Preventing future data leakage
- [**Synthetic Data Validation**](testing/SYNTHETIC_DATA_VALIDATION_COMPLETE.md) - Mathematical proof of engine correctness

**Status**: [+] **All 50 tests passing** (100% pass rate)

**When to read**: Before committing engine changes, when validating accuracy

---

## QuantStats Integration

**Professional performance reporting with 50+ metrics.**

- [**QuantStats README**](quantstats/README.md) - Overview of QuantStats integration
- [**QuantStats Metrics Explained**](quantstats/QUANTSTATS_METRICS_EXPLAINED.md) - Understanding the 50+ metrics

**When to read**: When generating tearsheets or analyzing performance

---

## Strategies Documentation

**Comprehensive documentation for all trading strategies.**

### Production Strategies (Deployed on EC2)
- [**RAMP Strategy**](strategies/production/RAMP_STRATEGY.md) - Regime-aware momentum protection (S&P 500)
- [**OMR Strategy**](strategies/production/OMR_STRATEGY.md) - Overnight mean reversion (leveraged ETFs)

### Research Strategies (Advanced)
- [**Hurst Mean Reversion**](strategies/research/HURST_MR_STRATEGY.md) - Hurst exponent-based mean reversion for crypto
- [**ML Crypto Mean Reversion**](strategies/research/ML_CRYPTO_MR_STRATEGY.md) - Machine learning enhanced crypto mean reversion

### Research Strategies (Legacy)

| Strategy | Documentation | Description |
|----------|---------------|-------------|
| Moving Average Crossover | [MOVING_AVERAGE_CROSSOVER.md](strategies/research/MOVING_AVERAGE_CROSSOVER.md) | Fast/slow MA crossover signals |
| Triple Moving Average | [TRIPLE_MOVING_AVERAGE.md](strategies/research/TRIPLE_MOVING_AVERAGE.md) | Three MA trend alignment |
| Mean Reversion (BB) | [MEAN_REVERSION_BOLLINGER.md](strategies/research/MEAN_REVERSION_BOLLINGER.md) | Bollinger Band mean reversion |
| RSI Mean Reversion | [RSI_MEAN_REVERSION.md](strategies/research/RSI_MEAN_REVERSION.md) | RSI oversold/overbought signals |
| Mean Reversion L/S | [MEAN_REVERSION_LONG_SHORT.md](strategies/research/MEAN_REVERSION_LONG_SHORT.md) | Long/short flip-flop strategy |
| MACD Momentum | [MACD_MOMENTUM.md](strategies/research/MACD_MOMENTUM.md) | MACD line crossover signals |
| Breakout | [BREAKOUT.md](strategies/research/BREAKOUT.md) | N-period high breakout |
| Volatility Targeted | [VOLATILITY_TARGETED_MOMENTUM.md](strategies/research/VOLATILITY_TARGETED_MOMENTUM.md) | Vol-scaled momentum |
| Cross-Sectional Momentum | [CROSS_SECTIONAL_MOMENTUM.md](strategies/research/CROSS_SECTIONAL_MOMENTUM.md) | Relative strength ranking |
| Pairs Trading | [PAIRS_TRADING.md](strategies/research/PAIRS_TRADING.md) | Cointegration-based pairs |
| Volume Breakout | [VOLUME_BREAKOUT.md](strategies/research/VOLUME_BREAKOUT.md) | Volume spike confirmation |
| ATR Squeeze Breakout | [ATR_SQUEEZE_BREAKOUT.md](strategies/research/ATR_SQUEEZE_BREAKOUT.md) | Volatility contraction breakout |
| 52-Week High Breakout | [52_WEEK_HIGH_BREAKOUT.md](strategies/research/52_WEEK_HIGH_BREAKOUT.md) | Near 52-week high selection |

**When to read**: When understanding strategy logic, parameters, or implementing new strategies

---

## [*] Deployment & Infrastructure

**Production deployment on AWS EC2 with automated scheduling and monitoring.**

### Quick Start Guides
- [**Quick Start Deployment**](guides/QUICK_START_DEPLOYMENT.md) - Fast 5-minute cloud deployment
- [**Complete Deployment Guide**](guides/DEPLOYMENT_GUIDE.md) - Comprehensive setup for Windows/Mac/Linux
- [**Infrastructure Overview**](INFRASTRUCTURE_OVERVIEW.md) - Complete AWS architecture, cost breakdown, daily operations

### Infrastructure Details
- [**Terraform README**](../infra/terraform/README.md) - Infrastructure as code configuration and management
- [**SSH Scripts Documentation**](../infra/ec2/SSH_SCRIPTS_README.md) - Quick-access management scripts (10 scripts)
- [**Health Check Cheatsheet**](HEALTH_CHECK_CHEATSHEET.md) - Comprehensive monitoring guide
- [**Troubleshooting Guide**](guides/TROUBLESHOOTING.md) - Common issues and solutions

### Live Trading Guides
- [**Live Paper Trading Guide**](guides/LIVE_PAPER_TRADING.md) - Complete paper trading setup and usage
- [**Quick Start Trading**](guides/QUICK_START_TRADING.md) - Fast start guide for live trading
- [**OMR Strategy Architecture**](architecture/OMR_STRATEGY_ARCHITECTURE.md) - Overnight mean reversion deployment
- [**Alpaca Paper Trading Guide**](guides/ALPACA_PAPER_TRADING_GUIDE.md) - Alpaca-specific setup and monitoring

**Key Features**:
- [+] EC2 instance with Python 3.11 (Amazon Linux 2023, t4g.medium ARM64)
- [+] Lambda-powered auto-start (9:00 AM ET) and auto-stop (4:30 PM ET) Monday-Friday
- [+] Systemd service with auto-restart on failure
- [+] SSH management scripts for status checks, logs, and restarts
- [+] Automated 6-point health monitoring
- [+] Infrastructure as code (Terraform)
- [+] ~$12/month cost with market-hours scheduling

**Current Deployment**:
- **Instance IP**: See `.env` file (`EC2_IP` variable)
- **Instance ID**: See `.env` file (`EC2_INSTANCE_ID` variable)
- **Region**: us-east-1 (N. Virginia)
- **Service**: homeguard-trading.service (systemd)

**When to read**: When deploying to cloud, monitoring production bot, or managing infrastructure

---

## Progress Tracking (Historical Record)

**Historical progress documentation - tracks features, fixes, and improvements over time.**

Location: [`progress/`](progress/)

**Current files**: 2025 progress reports, roadmaps, and validation summaries.

**Archived**: Older files (2024) moved to [`archive/progress-2024/`](archive/progress-2024/)

**When to read**: When researching past issues or understanding feature evolution

---

## Planning Documents

**Implementation plans and roadmaps.**

Location: [`planning/`](planning/)

- Strategy implementation plans (Kelly Criterion, LightGBM, etc.)
- Backtest standardization plans
- Integration designs

**When to read**: When understanding planned features or implementation approaches

---

## Archived Documentation

**Historical documentation preserved for reference.**

Location: [`archive/`](archive/)

- [`progress-2024/`](archive/progress-2024/) - Development progress from Oct-Nov 2024
- [`legacy-reports/`](archive/legacy-reports/) - Older backtest reports (Nov 2025)

**When to read**: When researching historical context or past implementations

---

## Quick Navigation

### I want to...

**Run my first backtest** -> [Backtesting Guide](guides/BACKTESTING_GUIDE.md)

**Deploy to AWS cloud** -> [Quick Start Deployment](guides/QUICK_START_DEPLOYMENT.md) [*] **NEW**

**Monitor production bot** -> [Health Check Cheatsheet](HEALTH_CHECK_CHEATSHEET.md) [*] **NEW**

**Create a custom strategy** -> [Advanced Strategies Guide](guides/ADVANCED_STRATEGIES_GUIDE.md)

**Understand existing strategies** -> [Strategies Documentation](#strategies-documentation)

**Validate strategy robustness** -> [Regime Analysis User Guide](guides/REGIME_ANALYSIS_USER_GUIDE.md)

**Prevent overfitting** -> [Regime-Based Testing Architecture](architecture/REGIME_BASED_TESTING.md) [*] **NEW**

**Understand the architecture** -> [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)

**Run the test suite** -> [Test Suite Quick Start](testing/TEST_SUITE_QUICK_START.md)

**Configure risk management** -> [Risk Management Guide](guides/RISK_MANAGEMENT_GUIDE.md)

**Fetch market data** -> [Data Ingestion Pipeline](guides/DATA_INGESTION_PIPELINE.md)

**Generate tearsheets** -> [QuantStats README](quantstats/README.md)

**Use the API programmatically** -> [API Reference](api/API_REFERENCE.md)

**Understand test results** -> [Backtest Accuracy Tests Complete](testing/BACKTEST_ACCURACY_TESTS_COMPLETE.md)

---

## Documentation Standards

### Keeping Docs Updated

**CRITICAL**: When you modify the system architecture (add/remove/move modules), **always update**:
- [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)
- [Module Reference](architecture/MODULE_REFERENCE.md) *(when available)*

See [CLAUDE.md](../CLAUDE.md) for complete coding guidelines.

### Contributing to Docs

When adding new documentation:
1. Place in appropriate folder (`guides/`, `api/`, `testing/`, `architecture/`)
2. Update this README.md with a link
3. Use markdown format
4. Include "Last Updated" date
5. Add to relevant section above

---

## Getting Help

**Questions about usage?** -> Read [Backtesting Guide](guides/BACKTESTING_GUIDE.md)

**Questions about code?** -> Read [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)

**Found a bug?** -> Run [Test Suite](testing/TEST_SUITE_QUICK_START.md) to validate

**Want to contribute?** -> Read [CONTRIBUTING.md](../CONTRIBUTING.md) and coding standards in [CLAUDE.md](../CLAUDE.md)

**Troubleshoot an issue** -> [Troubleshooting Guide](guides/TROUBLESHOOTING.md)

**View changelog** -> [CHANGELOG.md](../CHANGELOG.md)

---

## Documentation Status

| Category | Files | Status | Completeness |
|----------|-------|--------|--------------|
| Architecture | 3 | [+] Complete | 100% |
| User Guides | 8 | [+] Complete | 100% |
| Strategies | 17 | [+] Complete | 100% (2 prod + 15 research) |
| API Reference | 4 | [+] Complete | 100% |
| Testing | 5 | [+] Complete | 100% |
| QuantStats | 3 | [+] Complete | 100% |
| Deployment & Infrastructure | 9 | [+] Complete | 100% |
| Planning | 8 | [+] Complete | 100% |
| Progress | 10+ | [+] Complete | 100% (current year) |
| Archive | 50+ | [+] Complete | 100% (historical) |

**Overall Status**: [+] **Excellent** (35+ comprehensive docs, fully organized)

---

## Recent Updates

**2025-12-21** (Documentation Refactoring):
- [+] Removed all emojis from 262+ markdown files (~6,600 replacements)
- [+] Renamed module READMEs to descriptive names (DATA_ENGINE.md, GUI.md, etc.)
- [+] Created 17 strategy documentation files (2 production, 15 research)
- [+] Added web backend documentation (BACKEND.md)
- [+] Created emoji cleanup script (scripts/cleanup_emojis.py)
- [+] Updated CLAUDE.md with no-emoji documentation rule
- [+] Redesigned repo README.md as concise entry point
- [+] Added Strategies Documentation section to docs hub

**2025-12-06** (Documentation Cleanup):
- [+] Sanitized hardcoded IPs/instance IDs from docs (use `.env` placeholders)
- [+] Consolidated `docs/plans/` and `docs/todos/` into `docs/planning/`
- [+] Archived legacy reports and 2024 progress files
- [+] Added CONTRIBUTING.md, CHANGELOG.md, TROUBLESHOOTING.md
- [+] Updated navigation hub with new structure

**2025-11-15** (AWS Deployment & Infrastructure):
- [+] **Complete AWS EC2 production deployment**
  - EC2 instance with t4g.medium ARM64 (Python 3.11, 4 GB RAM)
  - Lambda-powered automated scheduling (9 AM start, 4:30 PM stop ET Mon-Fri)
  - Systemd service with auto-restart capabilities
  - ~$12/month total cost with market-hours scheduling
- [+] Created 10 SSH management scripts for easy monitoring (infra/ec2/)
- [+] Automated 6-point health check system
- [+] Complete infrastructure documentation (8 comprehensive docs)
- [+] Terraform infrastructure as code configuration
- [+] Added "[*] Deployment & Infrastructure" section to docs hub

**2025-11-06** (Regime Analysis):
- [+] **All 4 Levels of Regime-Based Testing Complete**
  - Level 1: Transparent integration (BacktestEngine parameter)
  - Level 2: GUI integration (checkbox toggle)
  - Level 3: Advanced CLI tools (walk-forward validation)
  - Level 4: Enhanced GUI display & file export
- [+] Created 10+ comprehensive documentation files
- [+] Updated main README.md with regime analysis section
- [+] Updated docs/README.md navigation hub
- [+] Updated REGIME_BASED_TESTING.md architecture doc to v2.0

**2025-11-05**:
- [+] Deleted 5 outdated/superseded docs
- [+] Reorganized into topic folders (architecture/, guides/, api/, testing/)
- [+] Created Architecture Overview
- [+] Created this navigation hub (README.md)
- [+] Updated CLAUDE.md with architecture update requirement

**2025-11-05** (Testing):
- [+] Completed Priority 3 tests (Survivorship Bias, Corporate Actions)
- [+] Added 2 synthetic validation tests (Alternating Wins/Losses, Extreme Volatility)
- [+] Fixed pandas FutureWarning
- [+] **All 50 tests passing** in 3.10 seconds

**2025-11-04** (Testing):
- [+] Completed Priority 2 tests (Cash Constraints, Data Integrity)
- [+] Fixed duplicate timestamp bug
- [+] 38/38 tests passing

**2025-11-03**:
- [+] Parallel chart generation
- [+] Portfolio mode GUI improvements
- [+] Benchmark comparison feature

---

**Last Updated**: 2025-12-21
**Maintained By**: Keep this index updated when adding/removing documentation
**Review Frequency**: Monthly or after major changes
