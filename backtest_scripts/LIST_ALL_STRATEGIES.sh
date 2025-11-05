#!/bin/bash
# List All Available Strategies
# Shows all built-in strategies and their parameters

cat << 'EOF'
================================================================================
                       AVAILABLE BACKTEST STRATEGIES
================================================================================

MOVING AVERAGE STRATEGIES:
  1. MovingAverageCrossover
     - Fast/slow MA crossover strategy
     - Parameters: fast_window, slow_window, ma_type (sma/ema)
     - Default: 20/50 SMA

  2. TripleMovingAverage
     - Triple MA trend alignment
     - Parameters: fast_window, medium_window, slow_window, ma_type
     - Default: 10/20/50 EMA

MEAN REVERSION STRATEGIES:
  3. MeanReversion
     - Bollinger Bands bounce strategy
     - Parameters: window, num_std, exit_at_middle
     - Default: 20-period, 2.0 std dev

  4. RSIMeanReversion
     - RSI oversold/overbought strategy
     - Parameters: rsi_window, oversold, overbought
     - Default: 14-period RSI, 30/70 levels

  5. OvernightMeanReversion
     - Overnight mean reversion based on VWAP distance
     - Parameters: distance_threshold, use_vwap, use_prior_return
     - Default: 2% distance threshold, VWAP-based
     - Enters at close, exits at next open

MOMENTUM STRATEGIES:
  6. MomentumStrategy
     - MACD crossover strategy
     - Parameters: fast, slow, signal
     - Default: 12/26/9

  7. BreakoutStrategy (Enhanced)
     - Price breakout strategy with advanced filters
     - Parameters: breakout_window, exit_window, volatility_filter,
                   volume_confirmation, use_atr_stop
     - Default: 20/10 windows, filters disabled
     - Optional: Volatility filter, volume confirmation, ATR trailing stop

  8. VolatilityTargetedMomentum
     - Time-series momentum with volatility scaling
     - Parameters: lookback_period, vol_window, target_vol, max_leverage
     - Default: 200-day lookback, 15% target vol, 2x max leverage
     - Position size = target_vol / current_vol

  9. CrossSectionalMomentum
     - Ranks stocks by momentum, selects top performers
     - Parameters: lookback_periods, weights, top_percentile, rebalance_period
     - Default: [3m,6m,12m] weighted, top 20%, monthly rebalance
     - REQUIRES: Multi-symbol sweep mode (--sweep --universe)

PAIRS/STATISTICAL ARBITRAGE:
  10. PairsTrading
      - Cointegration-based pairs trading
      - Parameters: entry_zscore, exit_zscore, cointegration_pvalue,
                    zscore_window
      - Default: 2.0 entry Z, 0.5 exit Z, 20-day window
      - REQUIRES: Exactly 2 symbols

================================================================================

BASIC USAGE:
  python src/backtest_runner.py --strategy [StrategyName] --symbols [SYMBOL] \
    --start [DATE] --end [DATE]

SWEEP USAGE (test strategy on multiple stocks):
  python src/backtest_runner.py --strategy [StrategyName] --universe [UNIVERSE] \
    --sweep --start [DATE] --end [DATE]

EXAMPLES:
  # Single stock backtest
  python src/backtest_runner.py --strategy MovingAverageCrossover \
    --symbols AAPL --start 2023-01-01 --end 2024-01-01

  # Sweep across FAANG stocks
  python src/backtest_runner.py --strategy BreakoutStrategy \
    --universe FAANG --sweep --start 2023-01-01 --end 2024-01-01

  # Cross-sectional strategy on DOW30
  python src/backtest_runner.py --strategy CrossSectionalMomentum \
    --universe DOW30 --sweep --start 2023-01-01 --end 2024-01-01

  # Pairs trading
  python src/backtest_runner.py --strategy PairsTrading \
    --symbols KO,PEP --start 2023-01-01 --end 2024-01-01

SEE ALSO:
  backtest_scripts/sweeps/list_universes.sh  - Show available universes
  backtest_scripts/sweeps/README.md          - Sweep functionality guide

================================================================================
EOF
