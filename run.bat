@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat

python -c "import mediapipe, cv2" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt || goto :error
)

python main.py %*
goto :eof

:error
echo.
echo Setup failed. Install Python 3.9-3.12 from python.org and tick "Add python.exe to PATH".
pause
