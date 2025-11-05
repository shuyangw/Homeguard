@echo off
REM Long Period Backtest - 2 Years (SWEEP MODE)
REM Tests strategy robustness over longer timeframe across DOW30


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
echo Long Period Backtest - 2 Years (SWEEP)
echo Strategy: Moving Average Crossover
echo Universe: DOW30 (30 stocks)
echo Period: 2022-01-01 to 2024-01-01
if "%BACKTEST_PARALLEL%"=="--parallel" (
    echo Mode: Parallel ^(%BACKTEST_MAX_WORKERS% workers^)
) else (
    echo Mode: Sequential
)
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --universe DOW30 ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start 2022-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0 ^
  --run-name "06_long_period_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --top-n 10

echo.
echo Sweep backtest complete! Check logs for CSV and HTML reports.
