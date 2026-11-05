@echo off
REM Usage (from "Anaconda Prompt"):  scripts\run_models.bat deepseek-v4-pro,gemini [k] [budget_per_model_usd]
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
call conda activate stattraps || (echo [!] run scripts\setup_and_check.bat first & pause & exit /b 1)
if "%~1"=="" (echo Usage: scripts\run_models.bat model1,model2 [k] [budget] & exit /b 1)
if not exist configs\models.json (echo [!] Copy configs\models.example.json to configs\models.json and edit it & pause & exit /b 1)
if not exist .env (echo [!] Copy .env.example to .env and add your API keys & pause & exit /b 1)
set K=%~2
if "%K%"=="" set K=2
set BUDGET=%~3
if "%BUDGET%"=="" set BUDGET=3
python -m stattraps.run --use %~1 --k %K% --budget-usd %BUDGET% --prompts neutral,cue --offpeak-only 2>&1 | python -u -c "import sys;f=open('run_log.txt','a',encoding='utf-8');[(print(l,end='',flush=True),f.write(l),f.flush()) for l in sys.stdin]"
python -m stattraps.analyze --results results --instances instances --out report
echo.
echo ==== Done. Results: report\summary.md and report\figures ====
pause
