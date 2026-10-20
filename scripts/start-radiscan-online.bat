@echo off
REM ---------------------------------------------------------------------------
REM RadiScan ONLINE: the API runs on this laptop (GPU), ngrok publishes it at a
REM fixed address, and the dashboard on Vercel talks to that address.
REM
REM   Dashboard : https://radiscan-ai.vercel.app
REM   API       : https://splicing-ahoy-reunite.ngrok-free.dev/api/v1
REM
REM Double-click this file. The site works while the two black windows stay
REM open. Close them to take RadiScan offline.
REM
REM Safe to run again at any time: it first stops a RadiScan server or ngrok
REM tunnel left over from an earlier run, so the newest code is always the one
REM that answers. (Without this, an old server keeps port 8000 and the new one
REM fails silently in its own window.)
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set "NGROK_DOMAIN=splicing-ahoy-reunite.ngrok-free.dev"
set "SITE=https://radiscan-ai.vercel.app"
set "PORT=8000"

where ngrok >nul 2>&1
if errorlevel 1 (
  echo ngrok is not installed. Install it once with:
  echo     winget install ngrok -s msstore
  echo then run:  ngrok config add-authtoken YOUR_TOKEN
  pause
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo Python environment .venv not found. Create it first: py -3.12 -m venv .venv
  pause
  exit /b 1
)

if not exist "models\checkpoints\best_model.pth" (
  echo models\checkpoints\best_model.pth is missing. Download the trained model first.
  pause
  exit /b 1
)

REM -- 1. Stop leftovers from an earlier run -----------------------------------
REM The free ngrok plan allows one tunnel at a time, so an old one has to go.
tasklist /FI "IMAGENAME eq ngrok.exe" 2>nul | find /I "ngrok.exe" >nul
if not errorlevel 1 (
  echo Stopping the ngrok tunnel from an earlier run...
  taskkill /IM ngrok.exe /F >nul 2>&1
)

REM Is something already listening on the API port?
set "BUSY="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do set "BUSY=1"
if defined BUSY (
  REM Only stop it if it is RadiScan; never kill an unrelated program.
  curl -s -m 5 http://127.0.0.1:%PORT%/ 2>nul | find "RadiScan" >nul
  if errorlevel 1 (
    echo.
    echo Port %PORT% is used by another program, not RadiScan.
    echo Close that program, then run this file again.
    pause
    exit /b 1
  )
  echo Stopping the RadiScan server from an earlier run...
  for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    taskkill /PID %%p /T /F >nul 2>&1
  )
  timeout /t 3 /nobreak >nul
)

REM -- 2. Start the API ---------------------------------------------------------
REM Allow the Vercel dashboard (and local dev) to call this API from a browser.
set "CORS_ORIGINS=%SITE%,http://localhost:3000"

echo Starting the RadiScan API on this laptop...
start "RadiScan API (port %PORT%)" cmd /k "call .venv\Scripts\activate.bat && uvicorn services.api.main:app --host 127.0.0.1 --port %PORT%"

echo Waiting for the model to load...
set /a TRIES=0
:wait_api
timeout /t 3 /nobreak >nul
set /a TRIES+=1
curl -s -m 3 http://127.0.0.1:%PORT%/api/v1/health/live >nul 2>&1
if not errorlevel 1 goto api_up
if !TRIES! lss 30 goto wait_api
echo.
echo The API did not start. Read the error in the "RadiScan API" window.
pause
exit /b 1

:api_up
REM -- 3. Publish it ------------------------------------------------------------
echo Publishing it at https://%NGROK_DOMAIN% ...
start "RadiScan tunnel (ngrok)" cmd /k "ngrok http %PORT% --url=https://%NGROK_DOMAIN%"

timeout /t 6 /nobreak >nul
start "" "%SITE%"

echo.
echo RadiScan is online at %SITE%
echo Keep the "RadiScan API" and "RadiScan tunnel" windows open.
echo.
endlocal
