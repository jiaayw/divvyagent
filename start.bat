@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.11+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || (echo Python 3.11 or newer is required. & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if not exist .env copy .env.example .env >nul
python app.py
pause
