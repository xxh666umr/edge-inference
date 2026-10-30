@echo off
chcp 65001 >nul
title Edge Inference Web Demo - STM32F103
cd /d "%~dp0"
set "PY=python"
if not exist "%PY%" if exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" set "PY=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
echo Starting dashboard at http://127.0.0.1:8765  (close this window to stop)
"%PY%" demo\server.py --http-port 8765
pause
