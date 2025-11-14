@echo off
REM Live Paper Trading Launcher (Windows)
REM
REM Usage:
REM   run_paper_trading.bat                          (Run MA Crossover continuous)
REM   run_paper_trading.bat --once                   (Run once and exit)
REM   run_paper_trading.bat --strategy omr           (Run OMR strategy)
REM   run_paper_trading.bat --strategy triple-ma     (Run Triple MA strategy)
REM   run_paper_trading.bat --universe leveraged     (Trade leveraged ETFs)
REM   run_paper_trading.bat --no-intraday-prefetch  (Disable 3:45PM data pre-fetch)
REM   run_paper_trading.bat --background             (Run in background/minimized window)
REM   run_paper_trading.bat --help                   (Show all options)

echo ================================================================================
echo                      HOMEGUARD LIVE PAPER TRADING
echo ================================================================================
echo.

REM Check if Python environment exists
if not exist "C:\Users\qwqw1\anaconda3\envs\fintech\python.exe" (
    echo [ERROR] Python environment not found!
    echo Expected: C:\Users\qwqw1\anaconda3\envs\fintech\python.exe
    echo.
    echo Please ensure the fintech conda environment is set up.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo.
    echo Please create a .env file with your Alpaca credentials:
    echo   ALPACA_PAPER_KEY_ID=your_key_id
    echo   ALPACA_PAPER_SECRET_KEY=your_secret_key
    echo   ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets/v2
    echo.
    pause
    exit /b 1
)

REM Parse arguments for --background flag
set BACKGROUND_MODE=0
set OTHER_ARGS=

:parse_args
if "%~1"=="" goto after_parse
if "%~1"=="--background" (
    set BACKGROUND_MODE=1
    shift
    goto parse_args
)
set OTHER_ARGS=%OTHER_ARGS% %1
shift
goto parse_args

:after_parse

REM Check if background mode is enabled
if %BACKGROUND_MODE%==1 (
    REM Create logs directory if it doesn't exist
    if not exist "logs" mkdir logs

    REM Generate log filename with timestamp
    for /f "tokens=2 delims==;" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set LOGFILE=logs\paper_trading_%datetime:~0,8%_%datetime:~8,6%.log

    echo [INFO] Starting paper trading in background mode...
    echo [INFO] Log file: %LOGFILE%
    echo [INFO] Running in minimized window
    echo.
    echo To Stop:
    echo   1. Find "Homeguard Paper Trading" window and close it
    echo   2. Or use Task Manager to end python.exe process
    echo   3. Or run: taskkill /F /FI "WINDOWTITLE eq Homeguard*"
    echo.
    echo To Monitor:
    echo   type %LOGFILE%
    echo.

    REM Run in background using start /MIN
    start "Homeguard Paper Trading" /MIN cmd /c "C:\Users\qwqw1\anaconda3\envs\fintech\python.exe" "scripts\trading\run_live_paper_trading.py" %OTHER_ARGS% ^>^> "%LOGFILE%" 2^>^&1

    echo ================================================================================
    echo [SUCCESS] Paper trading started in background!
    echo ================================================================================
    pause
) else (
    REM Run in foreground mode
    "C:\Users\qwqw1\anaconda3\envs\fintech\python.exe" "scripts\trading\run_live_paper_trading.py" %OTHER_ARGS%

    REM Check exit code
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ================================================================================
        echo [ERROR] Script exited with error code %ERRORLEVEL%
        echo ================================================================================
        pause
        exit /b %ERRORLEVEL%
    )

    echo.
    echo ================================================================================
    echo Script completed successfully
    echo ================================================================================
    pause
)
