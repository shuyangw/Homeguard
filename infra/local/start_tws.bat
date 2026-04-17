@echo off
REM Launch TWS via IBC (local dev machine).
REM Overrides the defaults in C:\IBC\StartTWS.bat without modifying it.

set TWS_MAJOR_VRSN=1037
set CONFIG=%USERPROFILE%\Documents\IBC\config.ini
set IBC_PATH=C:\IBC
set TWS_PATH=C:\Jts
set LOG_PATH=C:\IBC\Logs
set TWOFA_TIMEOUT_ACTION=exit
set APP=TWS
set TITLE=IBC (%APP% %TWS_MAJOR_VRSN%)

start "%TITLE%" /Min "%IBC_PATH%\scripts\DisplayBannerAndLaunch.bat"
