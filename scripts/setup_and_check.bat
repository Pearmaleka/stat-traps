@echo off
REM One-time setup + offline checks (no API key needed). Run from "Anaconda Prompt".
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
where conda >nul 2>nul || (echo [!] Open "Anaconda Prompt" and run this file from there. & pause & exit /b 1)
call conda env list | findstr /b /c:"stattraps " >nul || call conda create -y -q -n stattraps python=3.11
call conda activate stattraps || (echo [!] conda activate failed & pause & exit /b 1)
echo [1/4] Installing packages...
python -m pip install -q -r requirements.txt || (echo [!] pip failed & pause & exit /b 1)
echo [2/4] Generating the 90 instances...
python -m stattraps.suite --out instances
echo [3/4] Unit tests...
python -m pytest -q tests
echo [4/4] Offline validation: scripted model through the real agent loop (about 1-2 minutes)...
python -m stattraps.validate --instances instances
echo.
echo ==== Setup finished. Next: copy configs\models.example.json to configs\models.json and .env.example to .env ====
pause
