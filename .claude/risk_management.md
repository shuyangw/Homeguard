# Risk Management and Position Sizing

**CRITICAL**: All backtesting code MUST use proper position sizing and risk management. Using 99% of capital per trade is unrealistic and will produce misleading backtest results.

## Default Risk Management

**Risk management is ENABLED BY DEFAULT** with sensible defaults:

```python
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.utils.risk_config import RiskConfig

# Default: Uses RiskConfig.moderate() automatically
engine = BacktestEngine(initial_capital=100000)
# Position size: 10% per trade
# Stop loss: 2% per trade
# Max positions: 10
```

## RiskConfig Presets

**Three preset profiles available:**

```python
# Conservative (5% per trade, 1% stop loss)
config = RiskConfig.conservative()

# Moderate (10% per trade, 2% stop loss) - DEFAULT
config = RiskConfig.moderate()

# Aggressive (20% per trade, 3% stop loss)
config = RiskConfig.aggressive()

# Pass to engine
engine = BacktestEngine(initial_capital=100000, risk_config=config)
```

## Position Sizing Methods

**Five position sizing methods available** (see `docs/POSITION_SIZING_METHODS.md` for details):

1. **Fixed Percentage** (Default) - 10% per trade
2. **Fixed Dollar** - $10,000 per trade
3. **Volatility-Based (ATR)** - Risk 1% per trade based on ATR
4. **Kelly Criterion** - Mathematically optimal size
5. **Risk Parity** - Equal risk contribution across positions

For detailed examples and formulas, see [`docs/guides/RISK_MANAGEMENT_GUIDE.md`](../docs/guides/RISK_MANAGEMENT_GUIDE.md) and [`docs/guides/POSITION_SIZING_METHODS.md`](../docs/guides/POSITION_SIZING_METHODS.md).

## Stop Loss Types

**Four stop loss types available:**

1. **Fixed Percentage** - Exit when down 2%
2. **ATR-Based Trailing** - Dynamic stop based on volatility
3. **Time-Based** - Exit after N bars regardless of P&L
4. **Profit Target** - Take profit at 5%, stop loss at 2%

## Why 99% Per Trade Fails

Single-trade loss = near-total portfolio loss. No diversification. Results won't transfer to live trading.
Use 10% per trade: 10 positions possible, 10% loss on one = 1% portfolio impact.

## Strategy Requirements

Strategies MUST: accept `RiskConfig`, use configured sizing, apply stop losses, respect portfolio constraints.

## Common Mistakes

1. 99% per trade  2. Ignoring volatility  3. Full Kelly (too aggressive)  4. No stop losses

## Further Reading

- [`docs/guides/RISK_MANAGEMENT_GUIDE.md`](../docs/guides/RISK_MANAGEMENT_GUIDE.md) - Complete risk management guide
- [`docs/guides/POSITION_SIZING_METHODS.md`](../docs/guides/POSITION_SIZING_METHODS.md) - Position sizing formulas and examples
- [`backtest_guidelines/guidelines.md`](../backtest_guidelines/guidelines.md) - Common backtesting pitfalls
