@echo off
python --version > nul 2>&1
if not %errorlevel% equ 0 (
    echo Python is not found
)

@echo on
python -m venv venv
.\venv\Scripts\pip.exe install -r .\requirements.txt

pause
