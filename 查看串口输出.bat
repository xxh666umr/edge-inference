@echo off
chcp 65001 >nul
title STM32F103 serial output (close window to quit)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0firmware\tools\view_serial.ps1"
if errorlevel 1 pause
