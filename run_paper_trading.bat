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

REM Run the script with all passed arguments
"C:\Users\qwqw1\anaconda3\envs\fintech\python.exe" "scripts\trading\run_live_paper_trading.py" %*

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
