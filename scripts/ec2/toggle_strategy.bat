@echo off
REM Toggle Strategy - Windows version
REM Usage: toggle_strategy.bat status
REM        toggle_strategy.bat omr enable
REM        toggle_strategy.bat mp disable

setlocal

set REPO_DIR=%~dp0..\..
cd /d "%REPO_DIR%"

if "%1"=="" (
    python -c "import sys; sys.path.insert(0,'.'); from src.trading.state import StrategyStateManager; m = StrategyStateManager(); m.print_status()"
    goto :eof
)

if "%1"=="status" (
    python -c "import sys; sys.path.insert(0,'.'); from src.trading.state import StrategyStateManager; m = StrategyStateManager(); m.print_status()"
    goto :eof
)

if "%2"=="" (
    echo Error: Missing action
    echo Usage: toggle_strategy.bat ^<strategy^> ^<enable^|disable^>
    goto :eof
)

if "%2"=="enable" (
    python -c "import sys; sys.path.insert(0,'.'); from src.trading.state import StrategyStateManager; m = StrategyStateManager(); m.set_enabled('%1', True, 'toggle_strategy.bat'); print('%1 enabled')"
    goto :eof
)

if "%2"=="disable" (
    python -c "import sys; sys.path.insert(0,'.'); from src.trading.state import StrategyStateManager; m = StrategyStateManager(); m.set_enabled('%1', False, 'toggle_strategy.bat'); print('%1 disabled')"
    goto :eof
)

echo Unknown action: %2
echo Valid actions: enable, disable
