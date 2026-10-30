@echo off
REM Flash + verify + reset via OpenOCD (STM32F103ZET6 + CMSIS-DAP).
REM Usage: run firmware\build.bat first, then tools\flash.bat.
REM Set the OPENOCD environment variable to override the default path.
setlocal
cd /d "%~dp0"

if not defined OPENOCD set "OPENOCD=%LOCALAPPDATA%\Microsoft\WinGet\Packages\xpack-dev-tools.openocd-xpack_Microsoft.Winget.Source_8wekyb3d8bbwe\xpack-openocd-0.12.0-7\bin\openocd.exe"
set "ELF=%~dp0..\build\firmware.elf"

if not exist "%OPENOCD%" (
  echo [FAIL] OpenOCD not found: %OPENOCD%
  echo        Install OpenOCD or set the OPENOCD environment variable.
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
