@echo off
REM Run All Basic Backtests
REM Executes all 5 basic backtest scenarios sequentially
REM
REM Usage: RUN_ALL_BASIC.bat [OPTIONS]
REM
REM Arguments (all optional):
REM   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%%)
REM   --capital VALUE     Initial capital amount (e.g., 100000)
REM   --start DATE        Start date in YYYY-MM-DD format
REM   --end DATE          End date in YYYY-MM-DD format
REM   --quantstats        Enable QuantStats reporting (tearsheets)
REM   --verbosity LEVEL   Logging verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose
REM   --parallel          Enable parallel sweep execution (default)
REM   --no-parallel       Disable parallel execution (run sequentially)
REM   --max-workers N     Number of parallel workers (default: 4)
REM
REM Examples:
REM   RUN_ALL_BASIC.bat --fees 0.002 --capital 50000
REM   RUN_ALL_BASIC.bat --quantstats --verbosity 2
REM   RUN_ALL_BASIC.bat --no-parallel
REM   RUN_ALL_BASIC.bat --max-workers 8

REM ============================================================================
REM DEFAULT PARAMETERS - Can be overridden by command-line arguments
REM ============================================================================

REM Initial capital for all backtests (default: 100000)
set BACKTEST_CAPITAL=100000

REM Transaction fees as decimal (default: 0 = no fees)
set BACKTEST_FEES=0

REM Start date for all backtests (YYYY-MM-DD format)
set BACKTEST_START=2023-01-01

REM End date for all backtests (YYYY-MM-DD format)
set BACKTEST_END=2024-01-01

REM QuantStats reporting enabled by default
set BACKTEST_QUANTSTATS=--quantstats

REM Visualization enabled by default
set BACKTEST_VISUALIZE=--visualize

REM Logging verbosity (0-3, default: 1)
set BACKTEST_VERBOSITY=1

REM Parallel execution (default: enabled with --parallel flag)
set BACKTEST_PARALLEL=--parallel

REM Maximum parallel workers (default: 4)
set BACKTEST_MAX_WORKERS=4

REM ============================================================================
REM PARSE COMMAND-LINE ARGUMENTS
REM ============================================================================

:parse_args
if "%~1"=="" goto done_parsing
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
if /i "%~1"=="--quantstats" (
    set BACKTEST_QUANTSTATS=--quantstats
    shift
    goto parse_args
)
if /i "%~1"=="--verbosity" (
    set BACKTEST_VERBOSITY=%~2
    shift
    shift
    goto parse_args
)
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
echo Unknown argument: %~1
shift
goto parse_args

:done_parsing

REM Save the script directory for absolute paths
set SCRIPT_DIR=%~dp0

echo ================================================================================
echo                        RUN ALL BASIC BACKTESTS
echo ================================================================================
echo.
echo Configuration:
echo   Initial Capital: $%BACKTEST_CAPITAL%
echo   Transaction Fees: %BACKTEST_FEES%
echo   Period: %BACKTEST_START% to %BACKTEST_END%
if "%BACKTEST_PARALLEL%"=="--parallel" (
    echo   Parallel Mode: ENABLED ^(workers: %BACKTEST_MAX_WORKERS%^)
) else (
    echo   Parallel Mode: DISABLED ^(sequential execution^)
)
if "%BACKTEST_QUANTSTATS%"=="--quantstats" (
    echo   QuantStats Reports: ENABLED
) else (
    echo   QuantStats Reports: DISABLED
)
echo.
echo This will run 5 basic backtests sequentially:
echo   1. Simple MA Crossover (AAPL)
echo   2. RSI Mean Reversion (AAPL)
echo   3. Bollinger Bands (MSFT)
echo   4. MACD Momentum (GOOGL)
echo   5. Breakout Strategy (AMZN)
echo.
echo Estimated time: 5-10 minutes
echo.

echo [1/5] Running MA Crossover...
call "%SCRIPT_DIR%basic\01_simple_ma_crossover.bat"
echo.

echo [2/5] Running RSI Mean Reversion...
call "%SCRIPT_DIR%basic\02_rsi_mean_reversion.bat"
echo.

echo [3/5] Running Bollinger Bands...
call "%SCRIPT_DIR%basic\03_bollinger_bands.bat"
echo.

echo [4/5] Running MACD Momentum...
call "%SCRIPT_DIR%basic\04_macd_momentum.bat"
echo.

echo [5/5] Running Breakout Strategy...
call "%SCRIPT_DIR%basic\05_breakout_strategy.bat"
echo.

echo ================================================================================
echo                        ALL BASIC BACKTESTS COMPLETE!
echo ================================================================================
echo.
echo All 5 basic backtests have been completed.
if "%BACKTEST_QUANTSTATS%"=="--quantstats" (
    echo.
    echo QuantStats tearsheets have been generated for each strategy.
    echo Check the configured log_output_dir in settings.ini for reports.
    echo Open tearsheet.html files in your browser to view detailed performance.
)
echo.
echo Review the results above to compare strategy performance.
echo.
