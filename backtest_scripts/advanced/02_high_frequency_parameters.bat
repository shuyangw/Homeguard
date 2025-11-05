@echo off
REM High Frequency Trading Parameters
REM Very short MA windows for frequent trading


echo ========================================
echo High Frequency Parameters Backtest
echo Strategy: Moving Average Crossover
echo Parameters: Fast=3, Slow=10 (very short)
echo Symbol: AAPL
echo Period: 2023-01-01 to 2024-01-01
echo Expected: High trade frequency
echo ========================================
echo.
echo Warning: This will generate many trades!
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --symbols AAPL ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --params "fast_window=3,slow_window=10"

echo.
echo Backtest complete! Check trade count.

