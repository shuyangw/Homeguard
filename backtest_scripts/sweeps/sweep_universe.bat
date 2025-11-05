@echo off
REM Generic Universe Sweep Script
REM Usage: sweep_universe.bat UNIVERSE STRATEGY [START_DATE] [END_DATE]
REM Example: sweep_universe.bat FAANG MovingAverageCrossover 2023-01-01 2024-01-01

if "%~1"=="" (
    echo Error: Universe name required
    echo.
    echo Usage: sweep_universe.bat UNIVERSE STRATEGY [START_DATE] [END_DATE]
    echo.
    echo Examples:
    echo   sweep_universe.bat FAANG MovingAverageCrossover
    echo   sweep_universe.bat DOW30 BreakoutStrategy 2023-01-01 2024-01-01
    echo   sweep_universe.bat TECH_GIANTS MeanReversion
    echo.
    echo Run 'list_universes.bat' to see available universes
    exit /b 1
)

if "%~2"=="" (
    echo Error: Strategy name required
    echo.
    echo Available strategies:
    echo   - MovingAverageCrossover
    echo   - TripleMovingAverage
    echo   - MeanReversion
    echo   - RSIMeanReversion
    echo   - MomentumStrategy
    echo   - BreakoutStrategy
    echo   - VolatilityTargetedMomentum
    echo   - OvernightMeanReversion
    exit /b 1
)

set UNIVERSE=%~1
set STRATEGY=%~2

REM Use provided dates or defaults
if not "%~3"=="" (
    set START_DATE=%~3
) else (
    if not defined BACKTEST_START set START_DATE=2023-01-01
)

if not "%~4"=="" (
    set END_DATE=%~4
) else (
    if not defined BACKTEST_END set END_DATE=2024-01-01
)

REM Use environment variables or defaults
if not defined BACKTEST_CAPITAL set BACKTEST_CAPITAL=100000
if not defined BACKTEST_FEES set BACKTEST_FEES=0
if not defined BACKTEST_VERBOSITY set BACKTEST_VERBOSITY=1

echo.
REM Use environment variables if set by parent script, otherwise use defaults
if not defined BACKTEST_PARALLEL set BACKTEST_PARALLEL=--parallel
if not defined BACKTEST_MAX_WORKERS set BACKTEST_MAX_WORKERS=4

REM ============================================================================
REM PARSE COMMAND-LINE ARGUMENTS (optional overrides)
REM ============================================================================

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="--parallel" (
    set BACKTEST_PARALLEL=--parallel
    shift
    goto parse_args
)
if /i "%~1"=="--no-parallel" (
    set BACKTEST_PARALLEL=
    shift
    goto parse_args
)
if /i "%~1"=="--max-workers" (
    set BACKTEST_MAX_WORKERS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--fees" (
    set BACKTEST_FEES=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--capital" (
    set BACKTEST_CAPITAL=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--start" (
    set BACKTEST_START=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--end" (
    set BACKTEST_END=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--verbosity" (
    set BACKTEST_VERBOSITY=%~2
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
shift
goto parse_args

:done_parsing


echo ========================================
echo Universe Sweep
echo ========================================
echo Universe: %UNIVERSE%
echo Strategy: %STRATEGY%
echo Period: %START_DATE% to %END_DATE%
echo Capital: $%BACKTEST_CAPITAL%
echo Fees: %BACKTEST_FEES%
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sweep_%UNIVERSE%_%STRATEGY%" ^
  --sort-by "Sharpe Ratio" ^
  --parallel ^
  --verbosity %BACKTEST_VERBOSITY%

echo.
echo Sweep complete! Check logs directory for CSV and HTML reports.
