@echo off
REM Restart Homeguard Discord Bot
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Restarting Homeguard Discord Bot
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl restart homeguard-discord"

echo.
echo Waiting 5 seconds for service to start...
timeout /t 5 /nobreak > nul

echo.
echo ========================================
echo Current Status:
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl status homeguard-discord --no-pager"

pause
