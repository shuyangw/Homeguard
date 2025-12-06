@echo off
REM Restart Homeguard Trading Bot
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Restarting Homeguard Trading Bot
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl restart homeguard-trading && echo 'Bot restarted successfully' && sleep 3 && sudo systemctl status homeguard-trading --no-pager"

pause
