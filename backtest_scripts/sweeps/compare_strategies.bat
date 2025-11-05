@echo off
REM Compare Multiple Strategies on Same Universe
REM Runs several strategies on one universe and saves results for comparison

if "%~1"=="" (
    echo Error: Universe name required
    echo.
    echo Usage: compare_strategies.bat UNIVERSE [START_DATE] [END_DATE]
    echo.
    echo Example: compare_strategies.bat FAANG 2023-01-01 2024-01-01
    exit /b 1
)

set UNIVERSE=%~1

REM Use provided dates or defaults
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
echo Strategy Comparison on %UNIVERSE%
echo ========================================
echo Period: %START_DATE% to %END_DATE%
echo.
echo Testing the following strategies:
echo   1. MovingAverageCrossover (MA Cross)
echo   2. BreakoutStrategy (Breakout)
echo   3. MeanReversion (Bollinger Bands)
echo   4. MomentumStrategy (MACD)
echo   5. RSIMeanReversion (RSI)
echo ========================================
echo.

REM Strategy 1: Moving Average Crossover
echo.
echo [1/5] Testing MovingAverageCrossover...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MovingAverageCrossover ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "compare_%UNIVERSE%_MA_Crossover" ^
  --parallel ^
  --verbosity 0

REM Strategy 2: Breakout
echo.
echo [2/5] Testing BreakoutStrategy...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy BreakoutStrategy ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "compare_%UNIVERSE%_Breakout" ^
  --parallel ^
  --verbosity 0

REM Strategy 3: Mean Reversion
echo.
echo [3/5] Testing MeanReversion...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MeanReversion ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "compare_%UNIVERSE%_MeanReversion" ^
  --parallel ^
  --verbosity 0

REM Strategy 4: Momentum (MACD)
echo.
echo [4/5] Testing MomentumStrategy...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy MomentumStrategy ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "compare_%UNIVERSE%_MACD" ^
  --parallel ^
  --verbosity 0

REM Strategy 5: RSI Mean Reversion
echo.
echo [5/5] Testing RSIMeanReversion...
python "%~dp0..\..\src\backtest_runner.py" ^
  --strategy RSIMeanReversion ^
  --universe %UNIVERSE% ^
  --sweep ^
  %BACKTEST_PARALLEL% ^
  --max-workers %BACKTEST_MAX_WORKERS% ^
  --start %START_DATE% ^
  --end %END_DATE% ^
  --capital %BACKTEST_CAPITAL% ^
  --fees %BACKTEST_FEES% ^
  --run-name "compare_%UNIVERSE%_RSI" ^
  --parallel ^
  --verbosity 0

echo.
echo ========================================
echo All strategy comparisons complete!
echo ========================================
echo.
echo Results saved to logs directory with prefix: compare_%UNIVERSE%_
echo.
echo Compare the HTML reports to see which strategy performs best on %UNIVERSE%
echo.
