@echo off
REM Usage (from "Anaconda Prompt"):  scripts\list_models.bat gemini
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
call conda activate stattraps || (echo [!] run scripts\setup_and_check.bat first & pause & exit /b 1)
python -m stattraps.models --list %~1
pause
