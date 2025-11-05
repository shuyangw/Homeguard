@echo off
REM Volatility Targeted Momentum Strategy (SWEEP MODE)
REM Position sizing scales with volatility across TECH_GIANTS

if not defined BACKTEST_START set BACKTEST_START=2023-01-01
if not defined BACKTEST_END set BACKTEST_END=2024-01-01
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
echo Volatility Targeted Momentum (SWEEP)
echo Universe: TECH_GIANTS (10 stocks)
echo Period: %BACKTEST_START% to %BACKTEST_END%
echo Target Volatility: 15%% annualized
if "%BACKTEST_PARALLEL%"=="--parallel" (
    echo Mode: Parallel ^(%BACKTEST_MAX_WORKERS% workers^)
) else (
    echo Mode: Sequential
)
echo ========================================
echo.

python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy VolatilityTargetedMomentum ^
  --universe TECH_GIANTS ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %BACKTEST_START% ^
  --end %BACKTEST_END% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "02_volatility_targeted_momentum_sweep" ^
  --params "ma_window=200,vol_window=20,target_vol=0.15,max_leverage=2.0" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity %BACKTEST_VERBOSITY%

echo.
echo Sweep backtest complete! Check logs for CSV and HTML reports.
