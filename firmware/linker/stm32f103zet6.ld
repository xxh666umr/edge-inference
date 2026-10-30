/* STM32F103ZET6 (野火霸道 V2): 512KB Flash, 64KB SRAM */
ENTRY(Reset_Handler)

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (xrw) : ORIGIN = 0x20000000, LENGTH = 64K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    /* 必须 KEEP：否则 -Wl,--gc-sections 会把 crtn.o 的 .init/.fini 收尾片段当垃圾回收，
     * 导致 crti.o 的 _init（push ...）没有 epilogue，__libc_init_array 调用后跑飞
     * （复位后卡在 Default_Handler / 总线错误）。 */
    KEEP(*(.init))
    KEEP(*(.fini))
    KEEP(*(.preinit_array))
    KEEP(*(.init_array))
    KEEP(*(.init_array.*))
    KEEP(*(.fini_array))
    KEEP(*(.fini_array.*))
    *(.text)
    *(.text*)
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >RAM AT> FLASH
  _sidata = LOADADDR(.data);

  /* 热循环代码：运行地址在 SRAM（无 Flash 等待周期），加载地址在 Flash；
   * 启动时由 main() 里的 memcpy 从 _siramfunc 拷贝到 _sramfunc。 */
  .ramfunc :
  {
    . = ALIGN(4);
    _sramfunc = .;
    KEEP(*(.ramfunc))
    KEEP(*(.ramfunc*))
    . = ALIGN(4);
    _eramfunc = .;
  } >RAM AT> FLASH
  _siramfunc = LOADADDR(.ramfunc);

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } >RAM

  end = .;
  _end = .;
}
