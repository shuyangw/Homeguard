@echo off
REM Multi-Year Robustness Test (3 Years)
REM Tests strategy through different market conditions


echo ========================================
echo Multi-Year Robustness Test
echo Strategy: Moving Average Crossover
echo Symbol: AAPL
echo Period: 2021-01-01 to 2024-01-01 (3 years)
echo Testing through:
echo   - Bull market (2021)
echo   - Bear market (2022)
echo   - Recovery (2023-2024)
echo ========================================
echo.
echo This covers multiple market regimes...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --symbols AAPL ^
  --start 2021-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0

echo.
echo Backtest complete!

