@echo off
REM Quick Sweep - Fast testing with FAANG stocks
REM Uses FAANG (5 stocks) for quick validation and testing

if "%~1"=="" (
    echo Error: Strategy name required
    echo.
    echo Usage: quick_sweep.bat STRATEGY [PARAMS]
    echo.
    echo Examples:
    echo   quick_sweep.bat MovingAverageCrossover
    echo   quick_sweep.bat BreakoutStrategy
    echo   quick_sweep.bat MeanReversion "window=30,num_std=2.5"
    echo.
    exit /b 1
)

set STRATEGY=%~1
set PARAMS=%~2

echo.
REM Use environment variables if set by parent script, otherwise use defaults
if not defined BACKTEST_PARALLEL set BACKTEST_PARALLEL=--parallel
if not defined BACKTEST_MAX_WORKERS set BACKTEST_MAX_WORKERS=4

echo ========================================
echo Quick Sweep (FAANG)
echo ========================================
echo Strategy: %STRATEGY%
if not "%PARAMS%"=="" (
    echo Parameters: %PARAMS%
)
echo Universe: FAANG (5 stocks)
echo Period: Last year (2023-2024)
echo ========================================
echo.

if "%PARAMS%"=="" (
    python "%~dp0..\..\src\backtest_runner.py" ^
      --strategy %STRATEGY% ^
      --universe FAANG ^
      --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
      --start 2023-01-01 ^
      --end 2024-01-01 ^
      --parallel ^
      --run-name "quick_sweep_%STRATEGY%"
) else (
    python "%~dp0..\..\src\backtest_runner.py" ^
      --strategy %STRATEGY% ^
      --universe FAANG ^
      --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
      --start 2023-01-01 ^
      --end 2024-01-01 ^
      --params "%PARAMS%" ^
      --parallel ^
      --run-name "quick_sweep_%STRATEGY%"
)

echo.
echo Quick sweep complete! Check logs for results.
