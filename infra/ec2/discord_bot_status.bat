@echo off
REM Check Homeguard Discord Bot Status
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Checking Homeguard Discord Bot Status
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl status homeguard-discord --no-pager"

echo.
echo ========================================
echo Recent Activity (last 10 lines):
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-discord -n 10 --no-pager"

pause
