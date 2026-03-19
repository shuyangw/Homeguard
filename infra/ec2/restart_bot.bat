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

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl restart homeguard-omr homeguard-mp && echo 'Both strategies restarted successfully' && sleep 3 && echo '--- OMR ---' && sudo systemctl status homeguard-omr --no-pager | head -10 && echo '' && echo '--- MP ---' && sudo systemctl status homeguard-mp --no-pager | head -10"

pause
