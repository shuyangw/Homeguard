@echo off
REM Optimize for Maximum Returns (not risk-adjusted)
REM Finds parameters that maximize total return


echo ========================================
echo Total Return Optimization
echo Strategy: Moving Average Crossover
echo Parameter Grid:
echo   Fast: 5, 10, 15, 20
echo   Slow: 30, 40, 50
echo Optimization Metric: Total Return
echo Symbol: GOOGL
echo Period: 2022-01-01 to 2023-01-01
echo ========================================
echo.
echo Optimizing for highest total return...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --optimize ^
  --strategy MovingAverageCrossover ^
  --symbols GOOGL ^
  --start 2022-01-01 ^
  --end 2023-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --param-grid "fast_window=5,10,15,20;slow_window=30,40,50" ^
  --metric total_return

echo.
echo Optimization complete!

