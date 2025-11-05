@echo off
REM Optimize MA Crossover Parameters
REM Tests multiple fast/slow window combinations


echo ========================================
echo MA Crossover Parameter Optimization
echo Strategy: Moving Average Crossover
echo Parameter Grid:
echo   Fast: 10, 15, 20, 25, 30
echo   Slow: 40, 50, 60
echo Optimization Metric: Sharpe Ratio
echo Symbol: AAPL
echo Period: 2022-01-01 to 2023-01-01
echo ========================================
echo.
echo This will test 15 parameter combinations...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --optimize ^
  --strategy MovingAverageCrossover ^
  --symbols AAPL ^
  --start 2022-01-01 ^
  --end 2023-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --param-grid "fast_window=10,15,20,25,30;slow_window=40,50,60" ^
  --metric sharpe_ratio

echo.
echo Optimization complete!
echo Best parameters have been identified.

