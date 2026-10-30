@echo off
REM Flash + verify + reset via OpenOCD (STM32F103ZET6 + CMSIS-DAP).
REM Usage: run firmware\build.bat first, then tools\flash.bat.
setlocal
cd /d "%~dp0"

set "OPENOCD=<本机路径>"
set "ELF=%~dp0..\build\firmware.elf"

if not exist "%OPENOCD%" (
  echo [FAIL] OpenOCD not found: %OPENOCD%
  exit /b 1
)
if not exist "%ELF%" (
  echo [FAIL] %ELF% not found, run build.bat first
  exit /b 1
)

"%OPENOCD%" -f interface/cmsis-dap.cfg -f target/stm32f1x.cfg ^
  -c "program %ELF% verify reset exit"

if errorlevel 1 (
  echo [FAIL] flash failed
  exit /b 1
)
echo [OK] flashed + verified + reset
