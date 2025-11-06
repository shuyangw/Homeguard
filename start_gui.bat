@echo off
REM ============================================================================
REM Homeguard GUI Launcher
REM ============================================================================
REM This script launches the Homeguard backtesting GUI application.
REM
REM Requirements:
REM   - Anaconda/Miniconda installed
REM   - 'fintech' conda environment configured
REM
REM Usage:
REM   Double-click this file or run from command line: start_gui.bat
REM ============================================================================

echo.
echo ============================================================================
echo  Homeguard Backtesting Framework - GUI Launcher
echo ============================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found in PATH
    echo.
    echo Please ensure Anaconda or Miniconda is installed and added to PATH.
    echo.
    echo Common conda locations:
    echo   - C:\Users\%USERNAME%\anaconda3\Scripts\conda.exe
    echo   - C:\Users\%USERNAME%\miniconda3\Scripts\conda.exe
    echo   - C:\ProgramData\Anaconda3\Scripts\conda.exe
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating fintech conda environment...
call conda activate fintech
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to activate 'fintech' environment
    echo.
    echo Please create the environment first:
    echo   conda create -n fintech python=3.13
    echo   conda activate fintech
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo [2/3] Launching Homeguard GUI...
echo Current directory: %CD%
echo.

REM Add src to Python path and launch GUI
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python -m gui

REM Capture exit code
set GUI_EXIT_CODE=%ERRORLEVEL%

echo.
echo [3/3] GUI closed with exit code: %GUI_EXIT_CODE%

REM If there was an error, keep the window open
if %GUI_EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] GUI exited with an error
    echo.
    pause
)

exit /b %GUI_EXIT_CODE%
