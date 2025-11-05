@echo off
REM Stress Test on Volatile Stock
REM Tests strategy on high volatility symbol


echo ========================================
echo Volatility Stress Test
echo Strategy: Mean Reversion (Bollinger Bands)
echo Symbol: TSLA (high volatility)
echo Period: 2023-01-01 to 2024-01-01
echo Purpose: Test strategy in volatile conditions
echo ========================================
echo.
echo Testing on high volatility stock...
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MeanReversion ^
  --symbols TSLA ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0

echo.
echo Stress test complete!

