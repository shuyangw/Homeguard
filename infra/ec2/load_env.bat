@echo off
REM ============================================================================
REM Load environment variables from .env file
REM
REM This helper script parses the root .env file and sets environment variables
REM for use in Windows batch scripts. Call this at the start of EC2 scripts.
REM
REM Usage: call "%~dp0load_env.bat"
REM ============================================================================

REM Find project root (two levels up from infra/ec2/)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."

REM Check if .env file exists
if not exist "%PROJECT_ROOT%\.env" (
    echo ERROR: .env file not found at %PROJECT_ROOT%\.env
    echo.
    echo Please create .env from the template:
    echo   copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env"
    echo.
    echo Then edit .env with your EC2 instance details:
    echo   EC2_IP=your_instance_ip
    echo   EC2_INSTANCE_ID=your_instance_id
    echo   EC2_REGION=us-east-1
    echo   EC2_SSH_KEY_PATH=path_to_your_pem_file
    echo   EC2_USER=ec2-user
    exit /b 1
)

REM Clear any existing EC2 variables to ensure fresh load
set "EC2_IP="
set "EC2_INSTANCE_ID="
set "EC2_REGION="
set "EC2_USER="
set "EC2_SSH_KEY_PATH="

REM Parse .env file and set variables
REM This handles KEY="value" and KEY=value formats
for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%\.env") do (
    REM Skip comments (lines starting with #)
    echo %%a | findstr /b "#" >nul || (
        REM Only process EC2_ variables
        echo %%a | findstr /b "EC2_" >nul && (
            REM Remove quotes from value
            set "tmpval=%%b"
            if defined tmpval (
                call :setvar %%a
            )
        )
    )
)

REM Validate required EC2 variables
if not defined EC2_IP (
    echo ERROR: EC2_IP not set in .env file
    exit /b 1
)
if "%EC2_IP%"=="<YOUR_EC2_IP>" (
    echo ERROR: EC2_IP is still set to placeholder value
    echo Please edit .env and set your actual EC2 IP address
    exit /b 1
)

if not defined EC2_INSTANCE_ID (
    echo ERROR: EC2_INSTANCE_ID not set in .env file
    exit /b 1
)
if "%EC2_INSTANCE_ID%"=="<YOUR_INSTANCE_ID>" (
    echo ERROR: EC2_INSTANCE_ID is still set to placeholder value
    echo Please edit .env and set your actual EC2 instance ID
    exit /b 1
)

REM Set defaults if not specified
if not defined EC2_REGION set "EC2_REGION=us-east-1"
if not defined EC2_USER set "EC2_USER=ec2-user"
if not defined EC2_SSH_KEY_PATH set "EC2_SSH_KEY_PATH=%USERPROFILE%\.ssh\homeguard-trading.pem"

REM Expand ~ to %USERPROFILE% for SSH key path
if "%EC2_SSH_KEY_PATH:~0,1%"=="~" (
    set "EC2_SSH_KEY_PATH=%USERPROFILE%%EC2_SSH_KEY_PATH:~1%"
)

REM Clean up temp variables
set "SCRIPT_DIR="
set "PROJECT_ROOT="
set "tmpval="

exit /b 0

:setvar
REM Helper to set variable, removing quotes from value
set "varname=%1"
set "varval=%tmpval:"=%"
set "%varname%=%varval%"
exit /b
