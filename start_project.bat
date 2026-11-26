@echo off
REM Double-click this file on Windows. It finds Python, checks the project,
REM runs the tests/quick experiments if needed, then starts the dashboard.
cd /d "%~dp0"

set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" (where py >nul 2>nul && set PY=py -3)
if "%PY%"=="" (
  echo Python was not found on PATH. Install Python 3.10+ and tick "Add Python to PATH".
  pause
  exit /b 1
)

echo ============================================================
echo  SG-DMNMF - starting project. A browser window will open.
echo ============================================================
%PY% run_project.py all

echo.
echo If the dashboard did not open, run: python run_dashboard.py
echo then open http://127.0.0.1:8000 in your browser.
pause
