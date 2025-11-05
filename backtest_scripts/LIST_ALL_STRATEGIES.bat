@echo off
REM List All Available Strategies
REM Shows all built-in strategies and their parameters

echo ================================================================================
echo                        AVAILABLE BACKTEST STRATEGIES
echo ================================================================================
echo.
echo MOVING AVERAGE STRATEGIES:
echo   1. MovingAverageCrossover
echo      - Fast/slow MA crossover strategy
echo      - Parameters: fast_window, slow_window, ma_type (sma/ema)
echo      - Default: 20/50 SMA
echo.
echo   2. TripleMovingAverage
echo      - Triple MA trend alignment
echo      - Parameters: fast_window, medium_window, slow_window, ma_type
echo      - Default: 10/20/50 EMA
echo.
echo MEAN REVERSION STRATEGIES:
echo   3. MeanReversion
echo      - Bollinger Bands bounce strategy
echo      - Parameters: window, num_std, exit_at_middle
echo      - Default: 20-period, 2.0 std dev
echo.
echo   4. RSIMeanReversion
echo      - RSI oversold/overbought strategy
echo      - Parameters: rsi_window, oversold, overbought
echo      - Default: 14-period RSI, 30/70 levels
echo.
echo   5. OvernightMeanReversion
echo      - Overnight mean reversion based on VWAP distance
echo      - Parameters: distance_threshold, use_vwap, use_prior_return
echo      - Default: 2%% distance threshold, VWAP-based
echo      - Enters at close, exits at next open
echo.
echo MOMENTUM STRATEGIES:
echo   6. MomentumStrategy
echo      - MACD crossover strategy
echo      - Parameters: fast, slow, signal
echo      - Default: 12/26/9
echo.
echo   7. BreakoutStrategy (Enhanced)
echo      - Price breakout strategy with advanced filters
echo      - Parameters: breakout_window, exit_window, volatility_filter,
echo                    volume_confirmation, use_atr_stop
echo      - Default: 20/10 windows, filters disabled
echo      - Optional: Volatility filter, volume confirmation, ATR trailing stop
echo.
echo   8. VolatilityTargetedMomentum
echo      - Time-series momentum with volatility scaling
echo      - Parameters: lookback_period, vol_window, target_vol, max_leverage
echo      - Default: 200-day lookback, 15%% target vol, 2x max leverage
echo      - Position size = target_vol / current_vol
echo.
echo   9. CrossSectionalMomentum
echo      - Ranks stocks by momentum, selects top performers
echo      - Parameters: lookback_periods, weights, top_percentile, rebalance_period
echo      - Default: [3m,6m,12m] weighted, top 20%%, monthly rebalance
echo      - REQUIRES: Multi-symbol sweep mode (--sweep --universe)
echo.
echo PAIRS/STATISTICAL ARBITRAGE:
echo   10. PairsTrading
echo       - Cointegration-based pairs trading
echo       - Parameters: entry_zscore, exit_zscore, cointegration_pvalue,
echo                     zscore_window
echo       - Default: 2.0 entry Z, 0.5 exit Z, 20-day window
echo       - REQUIRES: Exactly 2 symbols
echo.
echo ================================================================================
echo.
echo BASIC USAGE:
echo   python src\backtest_runner.py --strategy [StrategyName] --symbols [SYMBOL] ^
echo     --start [DATE] --end [DATE]
echo.
echo SWEEP USAGE (test strategy on multiple stocks):
echo   python src\backtest_runner.py --strategy [StrategyName] --universe [UNIVERSE] ^
echo     --sweep --start [DATE] --end [DATE]
echo.
echo EXAMPLES:
echo   REM Single stock backtest
echo   python src\backtest_runner.py --strategy MovingAverageCrossover ^
echo     --symbols AAPL --start 2023-01-01 --end 2024-01-01
echo.
echo   REM Sweep across FAANG stocks
echo   python src\backtest_runner.py --strategy BreakoutStrategy ^
echo     --universe FAANG --sweep --start 2023-01-01 --end 2024-01-01
echo.
echo   REM Cross-sectional strategy on DOW30
echo   python src\backtest_runner.py --strategy CrossSectionalMomentum ^
echo     --universe DOW30 --sweep --start 2023-01-01 --end 2024-01-01
echo.
echo   REM Pairs trading
echo   python src\backtest_runner.py --strategy PairsTrading ^
echo     --symbols KO,PEP --start 2023-01-01 --end 2024-01-01
echo.
echo SEE ALSO:
echo   backtest_scripts\sweeps\list_universes.bat  - Show available universes
echo   backtest_scripts\sweeps\README.md           - Sweep functionality guide
echo.
echo ================================================================================
