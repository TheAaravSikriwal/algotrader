@echo off
REM ---------------------------------------------------------------------
REM  Day trader -- desktop launcher
REM
REM  Runs the app straight out of this folder, so whatever the code says
REM  today is what opens. There is no copy to keep in sync: edit the repo
REM  and the next launch picks it up.
REM
REM  It kills any Streamlit already running first. That matters more than
REM  it looks: Streamlit caches imported modules, so a server left over
REM  from before a code change keeps serving the OLD core/ files while
REM  showing the new pages, which looks like a bug that will not die.
REM ---------------------------------------------------------------------

cd /d "%~dp0"
title Day trader

echo.
echo   Day trader
echo   ----------
echo   Folder: %CD%
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo   ERROR: .venv not found in this folder.
    echo   Expected: %CD%\.venv\Scripts\streamlit.exe
    echo.
    pause
    exit /b 1
)

echo   Stopping any old copy...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

REM Give the port a moment to come free before rebinding it.
timeout /t 2 /nobreak >nul

for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do set COMMIT=%%i
if defined COMMIT (echo   Version: %COMMIT%) else (echo   Version: not a git checkout)

echo   Starting...
echo.
echo   The app opens at http://localhost:8501
echo   Close this window to stop it.
echo.

start "" http://localhost:8501

".venv\Scripts\streamlit.exe" run app.py ^
  --server.port 8501 ^
  --server.headless true ^
  --browser.gatherUsageStats false

echo.
echo   The app stopped. Anything above this line is the reason.
pause
