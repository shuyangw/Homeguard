@echo off
REM Optimize RSI Oversold/Overbought Levels
REM Finds optimal RSI thresholds


echo ========================================
echo RSI Parameter Optimization
echo Strategy: RSI Mean Reversion
echo Parameter Grid:
echo   Oversold: 20, 25, 30, 35
echo   Overbought: 65, 70, 75, 80
echo Optimization Metric: Sharpe Ratio
echo Symbol: MSFT
echo Period: 2022-01-01 to 2023-01-01
echo ========================================
echo.
echo This will test 16 parameter combinations...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --optimize ^
  --strategy RSIMeanReversion ^
  --symbols MSFT ^
  --start 2022-01-01 ^
  --end 2023-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --param-grid "oversold=20,25,30,35;overbought=65,70,75,80" ^
  --metric sharpe_ratio

echo.
echo Optimization complete!

