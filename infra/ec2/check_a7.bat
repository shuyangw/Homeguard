@echo off
REM A7 paper-validation counter check (Windows Task Scheduler wrapper).
REM
REM Schedules via:
REM   schtasks /create /tn HomeguardA7Check /tr "%CD%\infra\ec2\check_a7.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:10 /f
REM
REM See docs/operations/A7_MONITORING_SETUP.md for full setup instructions.

REM Locate Git Bash (try common install paths)
set "GIT_BASH_EXE="
if exist "C:\Program Files\Git\bin\bash.exe" set "GIT_BASH_EXE=C:\Program Files\Git\bin\bash.exe"
if exist "C:\Program Files (x86)\Git\bin\bash.exe" set "GIT_BASH_EXE=C:\Program Files (x86)\Git\bin\bash.exe"

if "%GIT_BASH_EXE%"=="" (
    echo [!] Git Bash not found in common locations.
    echo [!] Edit %~f0 to set GIT_BASH_EXE explicitly.
    exit /b 1
)

REM Resolve repo root assuming this .bat lives in <repo>/infra/ec2/
set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"
"%GIT_BASH_EXE%" -lc "./scripts/ops/check_a7_counter.sh"
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
