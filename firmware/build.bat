@echo off
REM Build STM32F103ZET6 firmware (arm-none-eabi-gcc 14.2).
REM Usage: run build.bat inside the firmware directory (double-click also works).
REM Output: build\firmware.elf / .bin / .hex / .map
setlocal
cd /d "%~dp0"

set "TOOL=C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin"
set "CC=%TOOL%\arm-none-eabi-gcc.exe"
set "OBJCOPY=%TOOL%\arm-none-eabi-objcopy.exe"
set "SIZE=%TOOL%\arm-none-eabi-size.exe"

if not exist build mkdir build

"%CC%" -mcpu=cortex-m3 -mthumb -O2 -ffunction-sections -fdata-sections -Wall -DSTM32F103xE ^
  -Icmsis -I..\csrc -I..\models ^
  -Tlinker\stm32f103zet6.ld -Wl,--gc-sections -Wl,-Map=build\firmware.map ^
  -o build\firmware.elf ^
  cmsis\startup_stm32f103xe.s cmsis\system_stm32f1xx.c src\main.c ..\models\model.c ..\csrc\nn.c ^
  --specs=nano.specs --specs=nosys.specs -Wl,-u,_printf_float -lm

if errorlevel 1 (
  echo [FAIL] build failed
  exit /b 1
)

"%OBJCOPY%" -O binary build\firmware.elf build\firmware.bin
"%OBJCOPY%" -O ihex build\firmware.elf build\firmware.hex
"%SIZE%" build\firmware.elf
echo [OK] build\firmware.elf / .bin / .hex
