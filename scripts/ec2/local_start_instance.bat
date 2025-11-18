@echo off
REM Start Homeguard Trading Bot EC2 Instance
REM
REM This script starts the EC2 instance from your local Windows machine using AWS CLI.
REM The instance will automatically start the trading bot via the systemd service.
REM
REM Prerequisites:
REM   - AWS CLI installed and configured
REM   - AWS credentials with EC2 permissions (ec2:StartInstances, ec2:DescribeInstances)
REM
REM Usage:
REM   scripts\ec2\local_start_instance.bat
REM

setlocal enabledelayedexpansion

REM Configuration
set INSTANCE_ID=i-02500fe2392631ff2
set REGION=us-east-1

echo ==========================================
echo Start Homeguard Trading Bot EC2 Instance
echo ==========================================
echo.
echo Instance ID: %INSTANCE_ID%
echo Region: %REGION%
echo.

REM Check if AWS CLI is installed
where aws >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] AWS CLI is not installed
    echo.
    echo Please install AWS CLI:
    echo   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
    exit /b 1
)

REM Check current instance state
echo Checking instance state...
for /f "tokens=*" %%a in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].State.Name" --output text 2^>nul') do set INSTANCE_STATE=%%a

if "%INSTANCE_STATE%"=="" (
    echo [ERROR] Failed to get instance state
    echo Please check:
    echo   1. AWS CLI is configured: aws configure
    echo   2. You have permissions to access instance %INSTANCE_ID%
    echo   3. Instance exists in region %REGION%
    exit /b 1
)

echo Current state: %INSTANCE_STATE%
echo.

REM Handle different states
if "%INSTANCE_STATE%"=="running" (
    echo [INFO] Instance is already running
    echo.

    REM Get public IP
    for /f "tokens=*" %%a in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2^>nul') do set PUBLIC_IP=%%a

    if not "!PUBLIC_IP!"=="None" (
        echo Public IP: !PUBLIC_IP!
        echo.
        echo To connect:
        echo   ssh -i ~/.ssh/homeguard-trading.pem ec2-user@!PUBLIC_IP!
        echo   or run: scripts\ec2\connect.bat
    )

    exit /b 0
)

if "%INSTANCE_STATE%"=="pending" (
    echo [INFO] Instance is already starting...
    echo Waiting for instance to be running...
) else if "%INSTANCE_STATE%"=="stopped" (
    echo [INFO] Starting instance...

    REM Start the instance
    aws ec2 start-instances --instance-ids %INSTANCE_ID% --region %REGION% >nul

    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to start instance
        exit /b 1
    )

    echo [SUCCESS] Start command sent
    echo Waiting for instance to be running...
) else (
    echo [WARNING] Instance is in state: %INSTANCE_STATE%
    echo Cannot start instance in this state
    exit /b 1
)

REM Wait for instance to be running
echo.
echo This may take 30-60 seconds...
echo.

aws ec2 wait instance-running --instance-ids %INSTANCE_ID% --region %REGION%

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Timeout waiting for instance to start
    exit /b 1
)

echo [SUCCESS] Instance is now running!
echo.

REM Get public IP
for /f "tokens=*" %%a in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].PublicIpAddress" --output text') do set PUBLIC_IP=%%a

echo Public IP: %PUBLIC_IP%
echo.

REM Give some time for SSH to be ready
echo Waiting 10 seconds for SSH to be ready...
timeout /t 10 /nobreak >nul

echo.
echo ==========================================
echo Instance Started Successfully!
echo ==========================================
echo.
echo To connect to the instance:
echo   ssh -i ~/.ssh/homeguard-trading.pem ec2-user@%PUBLIC_IP%
echo   or run: scripts\ec2\connect.bat
echo.
echo To check bot status:
echo   scripts\ec2\check_bot.bat
echo.
echo To view logs:
echo   scripts\ec2\view_logs.bat
echo.

endlocal
