@echo off
REM View Homeguard Trading Bot Live Logs
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Viewing Live Trading Bot Logs
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Use --output=cat to strip systemd formatting and colors for Windows CMD compatibility
ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-trading -f --output=cat"
