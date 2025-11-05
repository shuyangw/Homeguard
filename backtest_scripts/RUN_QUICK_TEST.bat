@echo off
REM Quick Test - Single Backtest with QuantStats Report
REM Runs one fast backtest to verify system is working
REM
REM Usage: RUN_QUICK_TEST.bat [--symbol AAPL] [--fees 0.002] [--capital 50000]
REM
REM Arguments (all optional):
REM   --symbol VALUE      Stock symbol to test (e.g., AAPL, MSFT)
REM   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%%)
REM   --capital VALUE     Initial capital amount (e.g., 100000)
REM   --start DATE        Start date in YYYY-MM-DD format
REM   --end DATE          End date in YYYY-MM-DD format
REM   --no-report         Skip QuantStats report generation (console output only)
REM
REM Examples:
REM   RUN_QUICK_TEST.bat
REM   RUN_QUICK_TEST.bat --symbol MSFT --fees 0.002
REM   RUN_QUICK_TEST.bat --symbol AAPL --no-report

REM ============================================================================
REM DEFAULT PARAMETERS - Can be overridden by command-line arguments
REM ============================================================================

REM Symbol to backtest (default: AAPL)
set QUICK_SYMBOL=AAPL

REM Initial capital (default: 100000)
set QUICK_CAPITAL=100000

REM Transaction fees as decimal (default: 0 = no fees)
set QUICK_FEES=0

REM Start date (YYYY-MM-DD format)
set QUICK_START=2024-01-01

REM End date (YYYY-MM-DD format)
set QUICK_END=2024-06-30

REM QuantStats report enabled by default
set QUICK_QUANTSTATS=--quantstats

REM Visualization enabled by default
set QUICK_VISUALIZE=--visualize

REM ============================================================================
REM PARSE COMMAND-LINE ARGUMENTS
REM ============================================================================

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="--symbol" (
    set QUICK_SYMBOL=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--fees" (
    set QUICK_FEES=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--capital" (
    set QUICK_CAPITAL=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--start" (
    set QUICK_START=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--end" (
    set QUICK_END=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--no-report" (
    set QUICK_QUANTSTATS=
    shift
    goto parse_args
)
echo Unknown argument: %~1
shift
goto parse_args

:done_parsing

REM Change to repo root (parent directory) and save current dir

echo ================================================================================
echo                   QUICK TEST WITH QUANTSTATS REPORTING
echo ================================================================================
echo.
echo This will run a backtest with comprehensive QuantStats tearsheet:
echo   Strategy: Moving Average Crossover
echo   Symbol: %QUICK_SYMBOL%
echo   Period: %QUICK_START% to %QUICK_END%
echo   Initial Capital: $%QUICK_CAPITAL%
echo   Transaction Fees: %QUICK_FEES%
echo.
echo Report includes:
echo   - Executive summary with performance rating
echo   - 50+ performance metrics (Sharpe, Sortino, Calmar, etc.)
echo   - Benchmark comparison (vs SPY)
echo   - Risk analysis (VaR, CVaR, drawdown)
echo   - Monthly/yearly returns heatmaps
echo   - Interactive HTML tearsheet
echo.
echo Estimated time: 1-2 minutes
echo.

python "%~dp0..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --symbols %QUICK_SYMBOL% ^
  --start %QUICK_START% ^
  --end %QUICK_END% ^
  --capital %QUICK_CAPITAL% ^
  --fees %QUICK_FEES% ^
  %QUICK_VISUALIZE% ^
  %QUICK_QUANTSTATS%

echo.
echo ================================================================================
echo                           QUICK TEST COMPLETE!
echo ================================================================================
echo.
if "%QUICK_QUANTSTATS%"=="--quantstats" (
    echo NEXT STEPS:
    echo   1. Check the console output above for the report location
    echo   2. Open the tearsheet.html file in your web browser
    echo   3. Review the executive summary and performance metrics
    echo.
    echo The report includes:
    echo   - tearsheet.html ^(main report^)
    echo   - quantstats_metrics.txt ^(text summary^)
    echo   - daily_returns.csv ^(returns data^)
    echo   - equity_curve.csv ^(portfolio value^)
) else (
    echo Backtest completed! No report generated (--no-report flag used).
)
echo.

