@echo off
REM View Homeguard Trading Bot Live Logs (Plain Text - No Colors)
REM Windows Batch Script - For CMD compatibility

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Viewing Live Trading Bot Logs (Plain)
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Strip ANSI color codes using sed on remote server
ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-trading -f --output=cat | sed 's/\x1b\[[0-9;]*m//g'"
