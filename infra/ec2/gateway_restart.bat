@echo off
REM Restart IB Gateway on EC2 (will require 2FA approval on IB Key app)

call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Restarting IB Gateway
echo ========================================
echo.
echo [!] You will need to approve 2FA on IB Key mobile app.
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo systemctl restart homeguard-gateway && echo '[+] Gateway restarting... approve 2FA on IB Key app'"
