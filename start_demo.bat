@echo off
chcp 65001 >nul
title Edge Inference Web Demo - STM32F103
cd /d "%~dp0"
set "PY=<本机路径>"
if not exist "%PY%" set "PY=python"
echo Starting dashboard at http://127.0.0.1:8765  (close this window to stop)
"%PY%" demo\server.py --http-port 8765
pause
