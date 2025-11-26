@echo off
REM Sector Rotation Sweep (SWEEP MODE)
REM Tests RSI strategy across multiple sectors independently


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
echo Sector Rotation Sweep
echo Strategy: RSI Mean Reversion
echo Testing across 4 sector universes:
echo   - TECH_GIANTS
echo   - FINANCE
echo   - HEALTHCARE
echo   - ENERGY
echo Period: 2023-01-01 to 2024-01-01
if "%BACKTEST_PARALLEL%"=="--parallel" (
    echo Mode: Parallel ^(%BACKTEST_MAX_WORKERS% workers^)
) else (
    echo Mode: Sequential
)
echo ========================================
echo.

echo [1/4] Testing TECH sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy RSIMeanReversion ^
  --universe TECH_GIANTS ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 250000 ^
  --fees 0 ^
  --run-name "04_sector_TECH_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity 0

echo.
echo [2/4] Testing FINANCE sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy RSIMeanReversion ^
  --universe FINANCE ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 250000 ^
  --fees 0 ^
  --run-name "04_sector_FINANCE_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity 0

echo.
echo [3/4] Testing HEALTHCARE sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy RSIMeanReversion ^
  --universe HEALTHCARE ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 250000 ^
  --fees 0 ^
  --run-name "04_sector_HEALTHCARE_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity 0

echo.
echo [4/4] Testing ENERGY sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy RSIMeanReversion ^
  --universe ENERGY ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 250000 ^
  --fees 0 ^
  --run-name "04_sector_ENERGY_sweep" ^
  --sort-by "Sharpe Ratio" ^
  --verbosity 0

echo.
echo ========================================
echo All sector sweeps complete!
echo Check logs directory for separate sector reports.
echo ========================================
