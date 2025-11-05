@echo off
REM Optimize to Minimize Drawdown
REM Finds most conservative parameter set


echo ========================================
echo Minimize Drawdown Optimization
echo Strategy: Mean Reversion (Bollinger Bands)
echo Parameter Grid:
echo   Window: 15, 20, 25, 30
echo   Std Dev: 1.5, 2.0, 2.5
echo Optimization Metric: Max Drawdown (minimize)
echo Symbol: AAPL
echo Period: 2022-01-01 to 2023-01-01
echo ========================================
echo.
echo Finding parameters with lowest drawdown...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --optimize ^
  --strategy MeanReversion ^
  --symbols AAPL ^
  --start 2022-01-01 ^
  --end 2023-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --param-grid "window=15,20,25,30;num_std=1.5,2.0,2.5" ^
  --metric max_drawdown

echo.
echo Optimization complete!

