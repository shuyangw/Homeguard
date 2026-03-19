# Homeguard

Algorithmic trading framework for backtesting, live trading, and strategy research.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # Add Alpaca API credentials
cp settings.ini.example settings.ini  # Set data paths

# 3. Download data
python scripts/download_symbols.py --csv config/universes/sp500-2025.csv --skip-existing

# 4. Run a backtest
python -m src.backtest_runner --config config/backtesting/ma_single.yaml
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

---

## Core Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Backtesting Engine** | VectorBT-based backtesting with GUI and CLI | [BACKTESTING_ENGINE.md](src/backtesting/BACKTESTING_ENGINE.md) |
| **Live Trading** | Alpaca paper/live trading with strategy adapters | [LIVE_TRADING_SYSTEM.md](src/trading/LIVE_TRADING_SYSTEM.md) |
| **Strategies** | 20+ strategies (production and research) | [STRATEGY_FRAMEWORK.md](src/strategies/STRATEGY_FRAMEWORK.md) |
| **Data Engine** | Multi-threaded data ingestion and storage | [DATA_ENGINE.md](src/data_engine/DATA_ENGINE.md) |
| **Streaming** | Real-time WebSocket market data | [STREAMING.md](src/streaming/STREAMING.md) |
| **Web UI** | React frontend + FastAPI backend | [WEB.md](src/web/WEB.md) |
| **Desktop GUI** | PyQt-based desktop interface | [GUI.md](src/gui/GUI.md) |

---

## Production Strategies

Currently deployed on EC2:

| Strategy | Schedule | Description |
|----------|----------|-------------|
| **OMR** | 3:50 PM entry, 9:31 AM exit | Overnight mean reversion on leveraged ETFs |
| **RAMP** | 3:55 PM rebalance | Regime-aware momentum protection on S&P 500 |

See [docs/strategies/production/](docs/strategies/production/) for strategy documentation.

---

## Documentation

### Getting Started
- [SETUP.md](SETUP.md) - Installation and configuration
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CLAUDE.md](CLAUDE.md) - Coding standards and guidelines

### Architecture
- [Architecture Overview](docs/architecture/ARCHITECTURE_OVERVIEW.md) - System design
- [Module Reference](docs/architecture/MODULE_REFERENCE.md) - All modules documented
- [Data Flow](docs/architecture/DATA_FLOW.md) - Data pipeline diagrams

### User Guides
- [Backtesting Guide](docs/guides/BACKTESTING_GUIDE.md) - Running backtests
- [Live Trading Guide](docs/guides/LIVE_PAPER_TRADING.md) - Paper/live trading setup
- [Regime Analysis Guide](docs/guides/REGIME_ANALYSIS_USER_GUIDE.md) - Market regime detection
- [Risk Management](docs/guides/RISK_MANAGEMENT_GUIDE.md) - Position sizing methods

### Deployment
- [Infrastructure Overview](docs/INFRASTRUCTURE_OVERVIEW.md) - AWS architecture
- [Health Check Cheatsheet](docs/HEALTH_CHECK_CHEATSHEET.md) - Monitoring guide
- [Terraform README](infra/terraform/README.md) - Infrastructure as code

### Strategies
- [Production Strategies](docs/strategies/production/) - RAMP, OMR documentation
- [Research Strategies](docs/strategies/research/) - Experimental strategies

### Full Documentation Index
See [docs/README.md](docs/README.md) for complete documentation listing.

---

## Project Structure

```
Homeguard/
  src/
    backtesting/      # Backtest engine, optimization, regimes
    trading/          # Live trading, brokers, execution
    strategies/       # Strategy implementations
    data/             # Data providers and caching
    data_engine/      # Data ingestion system
    streaming/        # Real-time WebSocket streaming
    web/              # Web API + React frontend
    gui/              # Desktop GUI
    utils/            # Logger, timezone, utilities
  config/             # YAML backtest configurations
  config/universes/     # Symbol CSV lists
  scripts/            # Utility scripts
  infra/              # Infrastructure (terraform + EC2 ops)
  tests/              # Unit tests
  docs/               # Documentation
```

---

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Quick test subset
python -m pytest tests/engine/ -v

# Type checking (optional)
mypy src/
```

See [tests/README.md](tests/README.md) for testing documentation.

---

## License

*(Add license here)*
