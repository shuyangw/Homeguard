@echo off
REM Save Results with Report and Plots
REM Generates CSV report and displays charts


echo ========================================
echo Backtest with Reporting
echo Strategy: Moving Average Crossover
echo Symbol: AAPL
echo Period: 2023-01-01 to 2024-01-01
echo Output: CSV report + equity curve plots
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --symbols AAPL ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --save-report ^
  --show-plots

echo.
echo Backtest complete! Check for generated CSV file.

