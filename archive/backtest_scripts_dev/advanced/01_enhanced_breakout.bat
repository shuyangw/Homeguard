@echo off
REM Enhanced Breakout Strategy with Volatility Filter
REM This script tests the breakout strategy with volatility and volume filters

if not defined BACKTEST_START set BACKTEST_START=2023-01-01
if not defined BACKTEST_END set BACKTEST_END=2024-01-01
if not defined BACKTEST_CAPITAL set BACKTEST_CAPITAL=100000
if not defined BACKTEST_FEES set BACKTEST_FEES=0
if not defined BACKTEST_VISUALIZE set BACKTEST_VISUALIZE=--visualize
if not defined BACKTEST_QUANTSTATS set BACKTEST_QUANTSTATS=--quantstats
if not defined BACKTEST_VERBOSITY set BACKTEST_VERBOSITY=1

echo.
echo ========================================
echo Enhanced Breakout Strategy
echo Symbol: AMZN
echo Period: %BACKTEST_START% to %BACKTEST_END%
echo Filters: Volatility + Volume + ATR Stop
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy BreakoutStrategy ^
  --symbols AMZN ^
  --start %BACKTEST_START% ^
  --end %BACKTEST_END% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "01_enhanced_breakout" ^
  --params "breakout_window=20,exit_window=10,volatility_filter=True,volume_confirmation=True,use_atr_stop=True" ^
  --verbosity %BACKTEST_VERBOSITY%

echo.
echo Backtest complete!
