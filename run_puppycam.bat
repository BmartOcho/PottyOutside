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

REM Rotate log by date/time
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set DT=%%c-%%a-%%b
for /f "tokens=1 delims=." %%t in ("%time%") do set TM=%%t
set TM=%TM::=-%
set LOG=%REPO%\logs\puppycam_%DT%_%TM%.log

cd /d "%REPO%"

REM Ensure we have the py launcher or python
where py >nul 2>&1
if %errorlevel%==0 ( set PYLAUNCH=py -3 ) else (
  where python >nul 2>&1
  if %errorlevel%==0 ( set PYLAUNCH=python ) else (
    echo Python not found. Install it, then re-run.
    pause
    exit /b 1
  )
)

REM Create venv if missing
if not exist "%PY%" (
  echo Creating virtual environment...
  %PYLAUNCH% -m venv "%VENV%"
  if not exist "%PY%" (
    echo Failed to create venv at %VENV%
    pause
    exit /b 1
  )
)

REM Install/update deps
echo Installing/updating packages (may take a few minutes on first run)...
"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%PY%" -m pip install -r "%REPO%\requirements.txt" >> "%LOG%" 2>&1

REM Warn if .env is missing (Telegram creds + camera indices live there)
if not exist "%REPO%\.env" (
  echo WARNING: no .env file found in %REPO%.
  echo Copy .env.example to .env and fill in TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
  echo Continuing anyway - alerts will only go to the console.
  pause
)

echo Starting PuppyCam. Press Q in the video window to stop.
echo Log: %LOG%

REM Run with auto-restart: a clean quit (Q key -> exit code 0) ends the loop;
REM a crash (nonzero exit) restarts after 5 seconds so the watcher survives
REM a week unattended.
:loop
"%PY%" -u "%REPO%\claude_potty_outside.py" >> "%LOG%" 2>&1
if %errorlevel%==0 goto done
echo PuppyCam crashed (exit %errorlevel%) at %date% %time% - restarting in 5s... >> "%LOG%"
echo PuppyCam crashed - restarting in 5 seconds (Ctrl+C here to abort)...
timeout /t 5 /nobreak >nul
goto loop

:done
echo PuppyCam stopped normally. Log: %LOG%
pause
