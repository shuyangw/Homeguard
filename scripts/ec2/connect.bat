@echo off
REM Quick SSH to Homeguard Trading Bot EC2 Instance
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Connecting to Homeguard Trading Bot
echo Instance: %EC2_IP%
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP%
