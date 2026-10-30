# Benchmark：int8 tiny CNN 在 PC 与 STM32F103ZET6 上的推理开销

> 模型：RadioML 11 类调制识别 tiny CNN（3,195 参数，输入 2×126 int8），详见
> [phase2_调制识别模型.md](phase2_调制识别模型.md)。
> **结论：PC ≈30 µs / 次；STM32F103ZET6 @72 MHz 实测 1,113,032 cycles ≈ 15.5 ms / 次**
> （DWT CYCCNT；优化前 33.2 ms，**降低 53.4%**）。

## 1. 硬件与工具链

| 项 | 值 |
|---|---|
| MCU | STM32F103ZET6（野火霸道 V2）：Cortex-M3 @72 MHz、512 KB Flash、64 KB SRAM、**无 FPU / 无 DSP** |
| 固件工具链 | ARM GNU Toolchain 14.2.rel1，`arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -O2` |
| 烧录 | OpenOCD 0.12.0 + 野火 DAP（CMSIS-DAP，SWD） |
| PC 编译 | MSVC 2022 `/nologo /O2 /utf-8 /W4` |
| 计时 | 板上 DWT->CYCCNT（每帧 20 次，1 次 warmup）；PC 用 `clock()`（1000 次） |
| 结果回传 | USART1（CH340，115200）→ `demo/server.py` → Web 实时看板 |

## 2. 资源占用（最终固件，arm-none-eabi-size）

```
   text    data     bss     dec     hex
  29120     460   14216   43796    ab14   firmware.elf   # 含 .ramfunc 与分阶段 benchmark
```

| 项 | 数值 | 占芯片 |
|---|---|---|
| Flash（text+data） | **29,580 B** | 5.79% / 512 KiB |
| RAM（data + bss + .ramfunc） | **15,628 B** | 23.85% / 64 KiB |
| 其中模型权重/偏置 | 3,372 B（Flash） | W_CONV1 224 + B_CONV1 64 + W_CONV2 2560 + B_CONV2 128 + W_FC 352 + B_FC 44 |
| `model_forward` 静态 buffer | 7,628 B（RAM .bss） | acc1 3840 + act1 960 + pool1 480 + acc2 1664 + act2 416 + pool2 192 + gap 32 + logits 44 |
| `.ramfunc`（SRAM 热循环代码） | **952 B**（RAM，LMA 在 Flash） | conv1d / maxpool / avgpool / fc / requantize_fx |
| 分阶段 benchmark | Flash +3.4 KB / RAM +5.1 KB | `main.c` 的 `bench_stages()` |

> Flash 只用 5.8%，**瓶颈仍在算力与取指**，不在存储。

## 3. 时延

### 3.1 PC（x86-64，MSVC `/O2`）

| 项 | 数值 |
|---|---|
| 单次 `model_forward` | **≈30 µs**（1000 次平均；多次运行 24~32 µs） |
| 输出 | argmax=0（AM-DSB），与量化参考逐位一致 |

### 3.2 STM32F103 @72 MHz（板上 DWT 实测）

| 配置 | cycles | 时延 | 相对基线 |
|---|---|---|---|
| ① 初始（float requantize，代码在 Flash） | 2,391,847 | 33.2 ms | — |
| ② ① + **requantize 定点化** | 1,731,469 | 24.0 ms | −27.6% |
| ③ ② + **热循环搬 SRAM 执行** | **1,113,032** | **15.5 ms** | **−53.4%** |

> 每帧 `min = max = avg`（误差 ±1 cycle），测量非常稳定；分类结果均为 pred=0 MATCH，
> 且与 PC/numpy 的 11 类概率完全一致（`tests/test_model_parity.py` 550/550）。

### 3.3 分阶段实测拆分（最终配置，每项 100 次平均）

| 阶段 | cycles | 占比 |
|---|---|---|
| conv2（32×13×16×5 MAC） | 652,592 | **58.6%** |
| conv1（16×60×2×7 MAC） | 274,452 | 24.7% |
| softmax（11 类，double exp） | 34,767 | 3.1% |
| requantize1（960 元素，定点） | 28,910 | 2.6% |
| requantize2（416 元素，定点） | 12,590 | 1.1% |
| avgpool / pool1 / pool2 / fc | 42,519 | 3.8% |
| 调用/循环等其余开销 | 67,202 | 6.0% |

关键对比：
- `requantize` 单算子：**326,505 → 28,904 cycles（11.3×）**，去掉了 int↔float 软浮点调用；
- conv1/conv2 搬 SRAM 后：**452,986 → 274,452**（−39.4%）、**1,068,561 → 652,592**（−38.9%）；
- 剩余大头仍是 conv2（58.6%），与 MAC 数（33,280）占比一致。

### 3.4 与静态估算的差异

上板前按「内层 5 条指令 × 7 cycles/MAC」估算 ≈0.52 M cycles / 7.2 ms：
- 未优化时实测 2.39 M / 33.2 ms，比估算慢 4.6×（Flash 取指 2 等待周期 + 软浮点）；
- 两轮优化后 1.11 M / 15.5 ms，比估算慢 2.2×，但比初始实现快 **2.15×**。
→ 结论：MCU 性能必须以实测为准，且「定点化 + 代码放 SRAM」是性价比最高的两个优化。

## 4. 优化实现要点

| 优化 | 做法 | 效果 |
|---|---|---|
| **requantize 定点化** | 离线把 `req = acc_scale/out_scale` 转成 `mul/2^24`（`mul` int32）；C 端用 int64 乘 + 算术右移，替代 int→float→int 软浮点 | 单算子 11.3×，总时延 −27.6% |
| **热循环搬 SRAM** | `nn.c` 用 `__attribute__((section(".ramfunc")))`；链接脚本 `.ramfunc` VMA=RAM / LMA=Flash（952 B）；`main()` 启动时 memcpy 到 SRAM | 卷积 −39%，总时延再 −35.7% |
| 保持一致性 | 定点参数由导出器生成（`MODEL_CONV*_FX_MUL/SHIFT`），numpy 参考同步实现整数右移语义 | 550/550 逐样本一致 |

> 定点化后 `mul` 只有 41,073 / 49,707（24 位小数），精度损失可忽略：int8 训练集精度仍为 63.27%，
> 5 折 OOF 精度与优化前一致。

## 5. 还可以继续优化（未实施）

| 方案 | 预期 | 说明 |
|---|---|---|
| 卷积内层手动 unroll ×4 + 指针自增 | conv −20~30% | 减少分支、改善取指 |
| Cortex-M3 `SMLAD`（int16 打包双乘加） | 理论卷积 ~2× | 需改数据布局与累加精度 |
| conv2 通道 32→24 | conv2 −25% | 精度可能掉 1~2 pt |
| `-O3 -funroll-loops` | −5%（实测） | 代码变大，收益有限 |

理论上限约 **9~11 ms**；对“每秒一帧 + Web 看板”的演示，当前 15.5 ms 已足够。

## 6. 复现命令

```powershell
# 1) PC 侧
cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && cd /d <仓库路径> && cl /nologo /O2 /utf-8 /W4 /Icsrc /Imodels csrc\nn.c models\model.c csrc\demo.c /Fe:demo.exe && demo.exe"

# 2) 固件 + 烧录（firmware 目录）
.\build.bat
.\tools\flash.bat

# 3) 串口看结果（USART1，115200；本机 CH340 为 COM6）
#    每 ~1.7 秒一帧；也可双击 查看串口输出.bat

# 4) 结果回传 PC 的 Web 看板
python demo\server.py --port COM6     # 浏览器 http://127.0.0.1:8765（或双击 start_demo.bat）
```

无串口时可用 OpenOCD 读 RAM 遥测（`g_telemetry` 8 个字；地址用 `arm-none-eabi-nm` 查）。

## 7. 结论

- **可用性**：固件上板实测通过——72 MHz PLL、DWT 正常、分类结果与 PC/numpy 完全一致；
- **性能**：**15.5 ms/次**（优化前 33.2 ms），对演示与每秒一帧的看板完全够用；
- **可移植性**：定点重量化 + `.ramfunc` 都是通用优化，不依赖具体模型，换模型只需重新导出；
- 后续上 H743/K210 对比时，本文档的测量方法与表格结构可直接复用。
