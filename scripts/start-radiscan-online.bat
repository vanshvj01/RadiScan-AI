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
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

set "NGROK_DOMAIN=splicing-ahoy-reunite.ngrok-free.dev"
set "SITE=https://radiscan-ai.vercel.app"

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

REM Allow the Vercel dashboard (and local dev) to call this API from a browser.
set "CORS_ORIGINS=%SITE%,http://localhost:3000"

echo Starting the RadiScan API on this laptop...
start "RadiScan API (port 8000)" cmd /k "call .venv\Scripts\activate.bat && uvicorn services.api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for the model to load...
timeout /t 20 /nobreak >nul

echo Publishing it at https://%NGROK_DOMAIN% ...
start "RadiScan tunnel (ngrok)" cmd /k "ngrok http 8000 --url=https://%NGROK_DOMAIN%"

timeout /t 6 /nobreak >nul
start "" "%SITE%"

echo.
echo RadiScan is online at %SITE%
echo Keep the "RadiScan API" and "RadiScan tunnel" windows open.
echo.
endlocal
