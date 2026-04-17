@echo off
REM Check IB Gateway status on EC2

call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo IB Gateway Status
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl status homeguard-xvfb homeguard-gateway --no-pager -l | head -30"
