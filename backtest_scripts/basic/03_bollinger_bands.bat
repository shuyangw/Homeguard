@echo off
REM Bollinger Bands Mean Reversion (SWEEP MODE)
REM Tests Bollinger Bands bounce strategy across FAANG stocks


REM Use environment variables if set by parent script, otherwise use defaults
if not defined BACKTEST_CAPITAL set BACKTEST_CAPITAL=100000
if not defined BACKTEST_FEES set BACKTEST_FEES=0
if not defined BACKTEST_START set BACKTEST_START=2023-01-01
if not defined BACKTEST_END set BACKTEST_END=2024-01-01
if not defined BACKTEST_VERBOSITY set BACKTEST_VERBOSITY=1
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
echo Bollinger Bands Backtest (SWEEP)
echo Strategy: Mean Reversion (Bollinger Bands)
echo Universe: FAANG (5 stocks)
echo Period: %BACKTEST_START% to %BACKTEST_END%
if "%BACKTEST_PARALLEL%"=="--parallel" (
    echo Mode: Parallel ^(%BACKTEST_MAX_WORKERS% workers^)
) else (
    echo Mode: Sequential
)
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MeanReversion ^
  --universe FAANG ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %BACKTEST_START% ^
  --end %BACKTEST_END% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "03_bollinger_bands_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity %BACKTEST_VERBOSITY%

echo.
echo Sweep backtest complete! Check logs for CSV and HTML reports.
