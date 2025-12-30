@echo off
REM 5x5 Tic-Tac-Toe Game Launcher for Windows
REM This batch file makes it easy to start the game with a double-click

echo.
echo ================================================================
echo   5x5 Tic-Tac-Toe with AI - Game Launcher
echo ================================================================
echo.
echo Starting game...
echo.

REM Try to run the game
python main_launcher.py

REM If there's an error, pause so user can see it
if errorlevel 1 (
    echo.
    echo ================================================================
    echo   Error: Could not start the game
    echo ================================================================
    echo.
    echo Possible reasons:
    echo   - Python is not installed
    echo   - Python is not in PATH
    echo   - Required dependencies are missing
    echo.
    echo Please make sure:
    echo   1. Python 3.7+ is installed
    echo   2. Install numpy: pip install numpy
    echo.
    pause
)
