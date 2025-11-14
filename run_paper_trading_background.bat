@echo off
REM Live Paper Trading Background Launcher (Windows)
REM
REM Usage:
REM   run_paper_trading_background.bat                   (Run OMR in background)
REM   run_paper_trading_background.bat --strategy omr    (Run OMR strategy)
REM   run_paper_trading_background.bat --help            (Show options)
REM
REM This script runs paper trading in the background and logs to a file.
REM To stop: Use Task Manager or run: taskkill /F /FI "WINDOWTITLE eq Homeguard*"

echo ================================================================================
echo            HOMEGUARD LIVE PAPER TRADING - BACKGROUND MODE
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

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Generate log filename with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOGFILE=logs\paper_trading_%datetime:~0,8%_%datetime:~8,6%.log

echo [INFO] Starting paper trading in background...
echo [INFO] Log file: %LOGFILE%
echo [INFO] To view logs: type logs\paper_trading_*.log
echo [INFO] To stop: Use Task Manager or Ctrl+C in the background window
echo.

REM Run in background using start /B (background) with output redirection
REM Note: start /B runs in same console but doesn't block
REM For true background, we'll launch a minimized window
start "Homeguard Paper Trading" /MIN cmd /c "C:\Users\qwqw1\anaconda3\envs\fintech\python.exe" "scripts\trading\run_live_paper_trading.py" %* >> "%LOGFILE%" 2>&1

echo ================================================================================
echo [SUCCESS] Paper trading started in background!
echo ================================================================================
echo.
echo Process Details:
echo   - Running in minimized window titled "Homeguard Paper Trading"
echo   - Logs: %LOGFILE%
echo   - View logs: tail -f %LOGFILE% (Git Bash) or type %LOGFILE%
echo.
echo To Stop:
echo   1. Find "Homeguard Paper Trading" window and close it
echo   2. Or use Task Manager: Find "python.exe" running run_live_paper_trading.py
echo   3. Or run: taskkill /F /FI "WINDOWTITLE eq Homeguard*"
echo.
echo To Monitor:
echo   tail -f %LOGFILE%
echo.
echo ================================================================================
pause
