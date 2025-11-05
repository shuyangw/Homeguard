# Homeguard Documentation Hub

**Welcome to the Homeguard Backtesting Framework documentation.**

This documentation is organized by topic for easy navigation. Select the category below that matches your needs.

---

## 📐 Architecture (Understanding the System)

**Start here to understand how Homeguard is built and how components interact.**

- [**Architecture Overview**](architecture/ARCHITECTURE_OVERVIEW.md) - ✅ High-level system design and 5-layer architecture
- [**Module Reference**](architecture/MODULE_REFERENCE.md) - ✅ Detailed module-by-module reference (60+ modules)
- [**Data Flow**](architecture/DATA_FLOW.md) - ✅ Data flow diagrams and execution paths

**When to read**: Before modifying core engine code or adding new features

---

## 📚 User Guides (How to Use Homeguard)

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

**When to read**: When implementing strategies, optimizing performance, or managing risk

---

## 🔌 API Reference (Technical Documentation)

**Detailed API documentation for developers.**

- [**API Reference**](api/API_REFERENCE.md) - Main API documentation
- [**API Reference (Sweep)**](api/API_REFERENCE_SWEEP.md) - Multi-symbol sweep functionality
- [**Sweep Migration Guide**](api/SWEEP_MIGRATION_GUIDE.md) - Migrating to sweep functionality
- [**Sweep Statistics Guide**](api/SWEEP_STATISTICS_GUIDE.md) - Understanding sweep metrics

**When to read**: When integrating Homeguard into your code or automating backtests

---

## ✅ Testing & Validation (Quality Assurance)

**Comprehensive testing documentation - our accuracy test suite validates engine correctness.**

### Quick Start
- [**Test Suite Quick Start**](testing/TEST_SUITE_QUICK_START.md) - ⭐ **START HERE** - Run 50 tests in 5 seconds

### Test Plans & Results
- [**Backtest Accuracy Test Plan**](testing/BACKTEST_ACCURACY_TEST_PLAN.md) - Master test plan (reference)
- [**Backtest Accuracy Tests Complete**](testing/BACKTEST_ACCURACY_TESTS_COMPLETE.md) - ✅ Final results (50/50 tests passing)

### Deep Dives
- [**Lookahead Bias Analysis**](testing/LOOKAHEAD_BIAS_ANALYSIS.md) - Preventing future data leakage
- [**Synthetic Data Validation**](testing/SYNTHETIC_DATA_VALIDATION_COMPLETE.md) - Mathematical proof of engine correctness

**Status**: ✅ **All 50 tests passing** (100% pass rate)

**When to read**: Before committing engine changes, when validating accuracy

---

## 📊 QuantStats Integration

**Professional performance reporting with 50+ metrics.**

- [**QuantStats README**](quantstats/README.md) - Overview of QuantStats integration
- [**QuantStats Migration Guide**](quantstats/MIGRATION_GUIDE_FOR_USERS.md) - Migrating to QuantStats reporting
- [**QuantStats Metrics Explained**](quantstats/QUANTSTATS_METRICS_EXPLAINED.md) - Understanding the 50+ metrics

**When to read**: When generating tearsheets or analyzing performance

---

## 📝 Progress Tracking (Historical Record)

**Historical progress documentation - tracks features, fixes, and improvements over time.**

Location: [`progress/`](progress/)

**Files** (chronological):
- Template: `progress/TEMPLATE.md`
- 2024-10-31: Strategy implementation reports
- 2024-11-01: Cleanup, bugfixes (file links, JSON serialization)
- 2024-11-02: Risk management integration, symbol auto-download
- 2025-11-02: Multi-symbol portfolio status
- 2025-11-03: Benchmark comparison, parallel charts, portfolio GUI improvements
- Metrics fixes: Performance optimizations (486s → 0.22s)
- Portfolio mode fixes

**When to read**: When researching past issues or understanding feature evolution

---

## 🔧 Internal Documentation

**Planning and cleanup docs (for maintainers).**

- [**Doc Cleanup Plan**](DOC_CLEANUP_PLAN.md) - Documentation reorganization plan
- *(Other internal docs)*

---

## Quick Navigation

### I want to...

**Run my first backtest** → [Backtesting Guide](guides/BACKTESTING_GUIDE.md)

**Create a custom strategy** → [Advanced Strategies Guide](guides/ADVANCED_STRATEGIES_GUIDE.md)

**Understand the architecture** → [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)

**Run the test suite** → [Test Suite Quick Start](testing/TEST_SUITE_QUICK_START.md)

**Configure risk management** → [Risk Management Guide](guides/RISK_MANAGEMENT_GUIDE.md)

**Fetch market data** → [Data Ingestion Pipeline](guides/DATA_INGESTION_PIPELINE.md)

**Generate tearsheets** → [QuantStats README](quantstats/README.md)

**Use the API programmatically** → [API Reference](api/API_REFERENCE.md)

**Understand test results** → [Backtest Accuracy Tests Complete](testing/BACKTEST_ACCURACY_TESTS_COMPLETE.md)

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

**Questions about usage?** → Read [Backtesting Guide](guides/BACKTESTING_GUIDE.md)

**Questions about code?** → Read [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)

**Found a bug?** → Run [Test Suite](testing/TEST_SUITE_QUICK_START.md) to validate

**Want to contribute?** → Read coding standards in [CLAUDE.md](../CLAUDE.md)

---

## Documentation Status

| Category | Files | Status | Completeness |
|----------|-------|--------|--------------|
| Architecture | 3/3 | ✅ Complete | 100% (All 3 docs) |
| User Guides | 8 | ✅ Complete | 100% |
| API Reference | 4 | ✅ Complete | 100% |
| Testing | 5 | ✅ Complete | 100% |
| QuantStats | 3 | ✅ Complete | 100% |
| Progress | 14+ | ✅ Complete | 100% (historical) |

**Overall Status**: 🟢 **Excellent** (23+ comprehensive docs, fully organized)

---

## Recent Updates

**2025-11-05**:
- ✅ Deleted 5 outdated/superseded docs
- ✅ Reorganized into topic folders (architecture/, guides/, api/, testing/)
- ✅ Created Architecture Overview
- ✅ Created this navigation hub (README.md)
- ✅ Updated CLAUDE.md with architecture update requirement

**2025-11-05** (Testing):
- ✅ Completed Priority 3 tests (Survivorship Bias, Corporate Actions)
- ✅ Added 2 synthetic validation tests (Alternating Wins/Losses, Extreme Volatility)
- ✅ Fixed pandas FutureWarning
- ✅ **All 50 tests passing** in 3.10 seconds

**2025-11-04** (Testing):
- ✅ Completed Priority 2 tests (Cash Constraints, Data Integrity)
- ✅ Fixed duplicate timestamp bug
- ✅ 38/38 tests passing

**2025-11-03**:
- ✅ Parallel chart generation
- ✅ Portfolio mode GUI improvements
- ✅ Benchmark comparison feature

---

**Last Updated**: 2025-11-05
**Maintained By**: Keep this index updated when adding/removing documentation
**Review Frequency**: Monthly or after major changes
