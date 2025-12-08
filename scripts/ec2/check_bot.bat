@echo off
REM Check Homeguard Trading Bot Status
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Checking Homeguard Trading Bot Status
echo ========================================
echo.

echo --- OMR Strategy ---
ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl status homeguard-omr --no-pager -l | head -15"

echo.
echo --- MP Strategy ---
ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl status homeguard-mp --no-pager -l | head -15"

echo.
echo ========================================
echo Recent Activity (OMR - last 5 lines):
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-omr -n 5 --no-pager"

echo.
echo ========================================
echo Recent Activity (MP - last 5 lines):
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-mp -n 5 --no-pager"

pause
