@echo off
title AI Agent Launcher

:: Check Python
echo [1/4] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python and add to PATH.
    pause
    exit /b 1
)

:: Prepare virtual environment
echo [2/4] Preparing virtual environment...
if not exist "venv\" (
    echo Creating venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create venv.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

:: Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo Dependency installation failed. Check network or requirements.txt.
    pause
    exit /b 1
)

:: Launch agent in a new window that stays open
echo [4/4] Launching AI Agent...
start /wait cmd /c "python main.py & echo. & echo AI Agent session ended. & pause"
exit