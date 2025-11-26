@echo off
REM Optimize Breakout Window Sizes
REM Tests different lookback periods for breakout


echo ========================================
echo Breakout Window Optimization
echo Strategy: Breakout Strategy
echo Parameter Grid:
echo   Breakout Window: 10, 15, 20, 25, 30
echo   Exit Window: 5, 7, 10
echo Optimization Metric: Sharpe Ratio
echo Symbol: AMZN
echo Period: 2022-01-01 to 2023-01-01
echo ========================================
echo.
echo Testing 15 parameter combinations...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --optimize ^
  --strategy BreakoutStrategy ^
  --symbols AMZN ^
  --start 2022-01-01 ^
  --end 2023-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --param-grid "breakout_window=10,15,20,25,30;exit_window=5,7,10" ^
  --metric sharpe_ratio

echo.
echo Optimization complete!

