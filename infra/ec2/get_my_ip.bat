@echo off
REM Quick helper to get your current public IP for AWS security groups

setlocal

echo ========================================
echo Your Current Public IP Address
echo ========================================
echo.

REM Get IP using PowerShell
for /f "tokens=*" %%a in ('powershell -Command "(Invoke-WebRequest -Uri 'https://checkip.amazonaws.com' -UseBasicParsing).Content.Trim()"') do set IP=%%a

if "%IP%"=="" (
    echo [ERROR] Could not retrieve IP address
    echo Please check your internet connection
    exit /b 1
)

echo IP Address: %IP%
echo For AWS:    %IP%/32
echo.
echo Use this IP in AWS Security Group rules:
echo   Type: SSH
echo   Port: 22
echo   Source: %IP%/32
echo.
echo Security Group: homeguard-trading-bot-sg
echo Region: us-east-1
echo.
echo Direct link:
echo https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#SecurityGroups:search=homeguard-trading-bot-sg
echo.

endlocal
