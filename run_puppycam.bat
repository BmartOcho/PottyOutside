@echo off
setlocal

REM ========= CONFIGURE THIS PATH =========
set REPO=C:\Users\User\OneDrive\Desktop\PottyOutside\PottyOutside
REM ======================================

if not exist "%REPO%" (
  echo Repo path not found: %REPO%
  pause
  exit /b 1
)

set VENV=%REPO%\.venv
set PY=%VENV%\Scripts\python.exe

mkdir "%REPO%\logs" 2>nul
set HEADLESS=1

REM Rotate log by date/time
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set DT=%%c-%%a-%%b
for /f "tokens=1 delims=." %%t in ("%time%") do set TM=%%t
set LOG=%REPO%\logs\puppycam_%DT%_%TM%.log

cd /d "%REPO%"

REM Ensure we have the py launcher or python
where py >nul 2>&1
if %errorlevel%==0 ( set PYLAUNCH=py -3 ) else (
  where python >nul 2>&1
  if %errorlevel%==0 ( set PYLAUNCH=python ) else (
    echo Python not found. Install it, then re-run. >> "%LOG%"
    echo Python not found. Install it, then re-run.
    pause
    exit /b 1
  )
)

REM Create venv if missing
if not exist "%PY%" (
  echo Creating virtual environment... >> "%LOG%"
  %PYLAUNCH% -m venv "%VENV%" >> "%LOG%" 2>&1
  if not exist "%PY%" (
    echo Failed to create venv at %VENV% >> "%LOG%"
    echo Failed to create venv. See log: %LOG%
    pause
    exit /b 1
  )
)

REM Write requirements.txt if missing
if not exist "%REPO%\requirements.txt" (
  echo opencv-python> "%REPO%\requirements.txt"
  echo requests>> "%REPO%\requirements.txt"
  echo ultralytics>> "%REPO%\requirements.txt"
  echo keyboard>> "%REPO%\requirements.txt"
  echo win10toast; sys_platform == 'win32'>> "%REPO%\requirements.txt"
)

REM Upgrade pip and install deps
echo Installing/Updating packages... >> "%LOG%"
"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%PY%" -m pip install -r "%REPO%\requirements.txt" >> "%LOG%" 2>&1

echo Starting PuppyCam headless from venv... >> "%LOG%"
"%PY%" -u "%REPO%\camera_controls.py" >> "%LOG%" 2>&1

echo Finished. Log: %LOG%
