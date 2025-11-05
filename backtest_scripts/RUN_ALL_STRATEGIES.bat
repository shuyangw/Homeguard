@echo off
REM Run All Backtests (Basic + Intermediate + Advanced + Optimization)
REM Executes all available backtest scenarios sequentially
REM
REM Usage: RUN_ALL_STRATEGIES.bat [OPTIONS]
REM
REM Arguments (all optional):
REM   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%%)
REM   --capital VALUE     Initial capital amount (e.g., 100000)
REM   --start DATE        Start date in YYYY-MM-DD format
REM   --end DATE          End date in YYYY-MM-DD format
REM   --quantstats        Enable QuantStats reporting with tearsheet
REM   --visualize         Enable TradingView visualization (deprecated)
REM   --verbosity LEVEL   Logging verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose
REM   --parallel          Enable parallel sweep execution (default)
REM   --no-parallel       Disable parallel execution (run sequentially)
REM   --max-workers N     Number of parallel workers (default: 4)
REM   --basic-only        Run only basic strategies (5 tests)
REM   --inter-only        Run only intermediate strategies (6 tests)
REM   --adv-only          Run only advanced strategies (6 tests)
REM   --opt-only          Run only optimization strategies (5 tests)
REM
REM Examples:
REM   RUN_ALL_STRATEGIES.bat --fees 0.002 --capital 50000 --quantstats
REM   RUN_ALL_STRATEGIES.bat --basic-only --quantstats --verbosity 2
REM   RUN_ALL_STRATEGIES.bat --start 2023-01-01 --end 2024-12-31 --quantstats
REM   RUN_ALL_STRATEGIES.bat --no-parallel --max-workers 8

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

REM QuantStats reporting enabled (empty = disabled, --quantstats = enabled)
set BACKTEST_QUANTSTATS=--quantstats

REM Visualization enabled by default
set BACKTEST_VISUALIZE=--visualize

REM Logging verbosity (0-3, default: 1)
set BACKTEST_VERBOSITY=1

REM Parallel execution (default: enabled with --parallel flag)
set BACKTEST_PARALLEL=--parallel

REM Maximum parallel workers (default: 4)
set BACKTEST_MAX_WORKERS=4

REM Test categories to run (default: all)
set RUN_BASIC=1
set RUN_INTERMEDIATE=1
set RUN_ADVANCED=1
set RUN_OPTIMIZATION=1

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
if /i "%~1"=="--visualize" (
    set BACKTEST_VISUALIZE=--visualize
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
if /i "%~1"=="--basic-only" (
    set RUN_BASIC=1
    set RUN_INTERMEDIATE=0
    set RUN_ADVANCED=0
    set RUN_OPTIMIZATION=0
    shift
    goto parse_args
)
if /i "%~1"=="--inter-only" (
    set RUN_BASIC=0
    set RUN_INTERMEDIATE=1
    set RUN_ADVANCED=0
    set RUN_OPTIMIZATION=0
    shift
    goto parse_args
)
if /i "%~1"=="--adv-only" (
    set RUN_BASIC=0
    set RUN_INTERMEDIATE=0
    set RUN_ADVANCED=1
    set RUN_OPTIMIZATION=0
    shift
    goto parse_args
)
if /i "%~1"=="--opt-only" (
    set RUN_BASIC=0
    set RUN_INTERMEDIATE=0
    set RUN_ADVANCED=0
    set RUN_OPTIMIZATION=1
    shift
    goto parse_args
)
echo Unknown argument: %~1
shift
goto parse_args

:done_parsing

REM Save the script directory for absolute paths
set SCRIPT_DIR=%~dp0

REM Count total tests to run
set /a TOTAL_TESTS=0
if %RUN_BASIC%==1 set /a TOTAL_TESTS+=5
if %RUN_INTERMEDIATE%==1 set /a TOTAL_TESTS+=6
if %RUN_ADVANCED%==1 set /a TOTAL_TESTS+=6
if %RUN_OPTIMIZATION%==1 set /a TOTAL_TESTS+=5

echo ================================================================================
echo                        RUN ALL BACKTEST STRATEGIES
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
echo   QuantStats: %BACKTEST_QUANTSTATS%
echo   Verbosity: %BACKTEST_VERBOSITY%
echo.
echo Test Categories:
if %RUN_BASIC%==1 echo   [X] Basic Strategies (5 tests)
if %RUN_INTERMEDIATE%==1 echo   [X] Intermediate Strategies (6 tests)
if %RUN_ADVANCED%==1 echo   [X] Advanced Strategies (6 tests)
if %RUN_OPTIMIZATION%==1 echo   [X] Optimization Strategies (5 tests)
echo.
echo Total Tests: %TOTAL_TESTS%
echo Estimated Time: 15-30 minutes (depending on categories selected)
echo.
echo Starting automated run...
echo.

set TEST_COUNTER=0

REM ============================================================================
REM BASIC STRATEGIES
REM ============================================================================

if %RUN_BASIC%==1 (
    echo.
    echo ================================================================================
    echo                        BASIC STRATEGIES (5 tests^)
    echo ================================================================================
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running MA Crossover (AAPL^)...
    call "%SCRIPT_DIR%basic\01_simple_ma_crossover.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running RSI Mean Reversion (AAPL^)...
    call "%SCRIPT_DIR%basic\02_rsi_mean_reversion.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Bollinger Bands (MSFT^)...
    call "%SCRIPT_DIR%basic\03_bollinger_bands.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running MACD Momentum (GOOGL^)...
    call "%SCRIPT_DIR%basic\04_macd_momentum.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Breakout Strategy (AMZN^)...
    call "%SCRIPT_DIR%basic\05_breakout_strategy.bat"
    echo.
)

REM ============================================================================
REM INTERMEDIATE STRATEGIES
REM ============================================================================

if %RUN_INTERMEDIATE%==1 (
    echo.
    echo ================================================================================
    echo                     INTERMEDIATE STRATEGIES (6 tests^)
    echo ================================================================================
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Custom MA Parameters...
    call "%SCRIPT_DIR%intermediate\01_custom_ma_parameters.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Multi-Symbol Portfolio...
    call "%SCRIPT_DIR%intermediate\02_multi_symbol_portfolio.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Tight RSI Levels...
    call "%SCRIPT_DIR%intermediate\03_tight_rsi_levels.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Triple MA Trend...
    call "%SCRIPT_DIR%intermediate\04_triple_ma_trend.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Higher Capital Lower Fees...
    call "%SCRIPT_DIR%intermediate\05_higher_capital_lower_fees.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Long Period Test...
    call "%SCRIPT_DIR%intermediate\06_long_period_test.bat"
    echo.
)

REM ============================================================================
REM ADVANCED STRATEGIES
REM ============================================================================

if %RUN_ADVANCED%==1 (
    echo.
    echo ================================================================================
    echo                       ADVANCED STRATEGIES (6 tests^)
    echo ================================================================================
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Large Portfolio (10 symbols^)...
    call "%SCRIPT_DIR%advanced\01_large_portfolio_10_symbols.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running High Frequency Parameters...
    call "%SCRIPT_DIR%advanced\02_high_frequency_parameters.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Multi-Year Robustness Test...
    call "%SCRIPT_DIR%advanced\03_multi_year_robustness_test.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Sector Rotation Portfolio...
    call "%SCRIPT_DIR%advanced\04_sector_rotation_portfolio.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Stress Test (Volatile Stock^)...
    call "%SCRIPT_DIR%advanced\05_stress_test_volatile_stock.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Running Save Results and Plot...
    call "%SCRIPT_DIR%advanced\06_save_results_and_plot.bat"
    echo.
)

REM ============================================================================
REM OPTIMIZATION STRATEGIES
REM ============================================================================

if %RUN_OPTIMIZATION%==1 (
    echo.
    echo ================================================================================
    echo                     OPTIMIZATION STRATEGIES (5 tests^)
    echo ================================================================================
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Optimizing MA Crossover...
    call "%SCRIPT_DIR%optimization\01_optimize_ma_crossover.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Optimizing RSI Levels...
    call "%SCRIPT_DIR%optimization\02_optimize_rsi_levels.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Optimizing for Returns...
    call "%SCRIPT_DIR%optimization\03_optimize_for_returns.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Optimizing Minimize Drawdown...
    call "%SCRIPT_DIR%optimization\04_optimize_minimize_drawdown.bat"
    echo.

    set /a TEST_COUNTER+=1
    echo [%TEST_COUNTER%/%TOTAL_TESTS%] Optimizing Breakout Windows...
    call "%SCRIPT_DIR%optimization\05_optimize_breakout_windows.bat"
    echo.
)

REM ============================================================================
REM COMPLETION SUMMARY
REM ============================================================================

echo.
echo ================================================================================
echo                        ALL BACKTESTS COMPLETE!
echo ================================================================================
echo.
echo Completed %TOTAL_TESTS% backtest(s) successfully.
echo.
if "%BACKTEST_QUANTSTATS%"=="--quantstats" (
    echo QuantStats tearsheets have been generated for each strategy.
    echo Check the configured log_output_dir in settings.ini for reports.
    echo.
    echo Open tearsheet.html files in your browser to view detailed performance reports.
    echo.
)
echo Review the results above to compare strategy performance.
echo.
echo Key metrics to compare:
echo   - Sharpe Ratio (risk-adjusted returns^)
echo   - Max Drawdown (worst-case loss^)
echo   - CAGR (compound annual growth rate^)
echo   - Win Rate (percentage of winning days^)
echo   - Profit Factor (avg win / avg loss^)
echo.
