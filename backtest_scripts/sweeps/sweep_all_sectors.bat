@echo off
REM Sweep All Sector Universes with One Strategy
REM Tests a strategy across all sector universes: SEMICONDUCTORS, ENERGY, FINANCE, HEALTHCARE, CONSUMER

if "%~1"=="" (
    echo Error: Strategy name required
    echo.
    echo Usage: sweep_all_sectors.bat STRATEGY [START_DATE] [END_DATE]
    echo.
    echo Example: sweep_all_sectors.bat MovingAverageCrossover 2023-01-01 2024-01-01
    exit /b 1
)

set STRATEGY=%~1

if not "%~2"=="" (
    set START_DATE=%~2
) else (
    set START_DATE=2023-01-01
)

if not "%~3"=="" (
    set END_DATE=%~3
) else (
    set END_DATE=2024-01-01
)

if not defined BACKTEST_CAPITAL set BACKTEST_CAPITAL=100000
if not defined BACKTEST_FEES set BACKTEST_FEES=0

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
echo Sector-Wide Sweep
echo ========================================
echo Strategy: %STRATEGY%
echo Period: %START_DATE% to %END_DATE%
echo.
echo Testing across 6 sector universes:
echo   1. SEMICONDUCTORS (10 stocks)
echo   2. ENERGY (10 stocks)
echo   3. FINANCE (10 stocks)
echo   4. HEALTHCARE (10 stocks)
echo   5. CONSUMER (10 stocks)
echo   6. TECH_GIANTS (10 stocks)
echo ========================================
echo.

REM Sector 1: Semiconductors
echo.
echo [1/6] Sweeping SEMICONDUCTORS sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe SEMICONDUCTORS ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_SEMICONDUCTORS_%STRATEGY%" ^
  --verbosity 0

REM Sector 2: Energy
echo.
echo [2/6] Sweeping ENERGY sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe ENERGY ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_ENERGY_%STRATEGY%" ^
  --verbosity 0

REM Sector 3: Finance
echo.
echo [3/6] Sweeping FINANCE sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe FINANCE ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_FINANCE_%STRATEGY%" ^
  --verbosity 0

REM Sector 4: Healthcare
echo.
echo [4/6] Sweeping HEALTHCARE sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe HEALTHCARE ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_HEALTHCARE_%STRATEGY%" ^
  --verbosity 0

REM Sector 5: Consumer
echo.
echo [5/6] Sweeping CONSUMER sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe CONSUMER ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_CONSUMER_%STRATEGY%" ^
  --verbosity 0

REM Sector 6: Tech Giants
echo.
echo [6/6] Sweeping TECH_GIANTS sector...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy %STRATEGY% ^
  --universe TECH_GIANTS ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "sector_TECH_GIANTS_%STRATEGY%" ^
  --verbosity 0

echo.
echo ========================================
echo All sector sweeps complete!
echo ========================================
echo.
echo Results saved to logs with prefix: sector_*_%STRATEGY%
echo.
echo Compare HTML reports to see which sector works best with %STRATEGY%
echo.
