# edge-inference：面向 MCU 的通用开源边缘推理框架

把分类神经网络（CNN / MLP / …）量化为 **int8**，自动导出成 **C 数组 + 纯 C 前向推理**，
在资源受限的 MCU（本仓库以 **STM32F103ZET6** 为目标）上跑推理、测时延，
并把**分类结果实时回传到 PC 的 Web 看板**。

框架与任务无关：当前示例任务是 **RadioML 2016.10a 的 11 类调制识别**（22 万条），
换任务只需重训 + 重新导出。

> 当前状态：**Phase 2 / 3 完成**——tiny CNN 训练（完整 2016.10a）、int8 量化、C 数组导出、
> PC 逐位对齐、STM32F103 板载实测（**15.5 ms/次**）、结果回传 PC 的实时看板，均已跑通；
> 另已扩展 GRU / LayerNorm / attention 算子并完成 git 开源准备。

> **路径说明**：文中出现的 `<仓库路径>` 等绝对路径是本机示例，
> 在别的机器上请替换成实际仓库路径；Python/工具链路径同理。

## 当前进度

- [x] **Phase 1** PC 侧跑通：int8 推理库 + PyTorch → C 数组导出 + 输出对齐
- [x] **Phase 2** 调制识别 tiny 模型：RadioML 11 类 → tiny CNN → int8 → `models/model.c`
- [x] **C 库扩展** `conv1d` / `maxpool` / `avgpool` / `requantize`（逐位 100% 对齐）
- [x] **序列算子扩展** GRU / LayerNorm / attention / softmax（float32 参考实现，numpy 对齐）
- [x] **固件** STM32F103ZET6：USART1 + 72 MHz PLL（HSE 失败回退 HSI）+ DWT CYCCNT 时延统计
- [x] **Phase 3** 板载实测：72 MHz PLL 起振、pred MATCH、**15.5 ms/次**
- [x] **时延优化**：requantize 定点化 + 卷积热循环搬 SRAM，33.2 ms → 15.5 ms（−53.4%）
- [x] **完整数据集重训**：RadioML 2016.10a 22 万条，SNR≥-10 时 66.5%（int8 66.2%）
- [x] **结果回传 PC**：串口 → `demo/server.py` → Web 实时看板（类别/概率/时延/历史）
- [ ] **Phase 4** 开源发布（LICENSE / git 已就绪，待推送到远程仓库）

## 系统链路

```
┌──────────────────┐   USART1 (PA9/PA10)   ┌─────────────┐   HTTP    ┌──────────────┐
│ STM32F103ZET6    │ ────────────────────> │ CH340 (COM6)│ ────────> │ 浏览器实时看板 │
│ int8 CNN 推理     │  115200 8N1, 每 ~1.7 s │  PC 串口     │  demo/server.py           │
│ 15.5 ms/次        │  打印一帧分类结果       │  解析帧      │  每 500 ms 刷新            │
└──────────────────┘                        └─────────────┘           └──────────────┘
```

![Web 实时看板](docs/dashboard_demo.png)

## 结果速览

| 指标 | 数值 |
|---|---|
| 数据 | RadioML 2016.10a：11 类 × 20 SNR（-20~18 dB）× 1000 = **220,000** 条 |
| 模型 | tiny 1D-CNN：2×128 → conv(k7,s2,16) → pool2 → conv(k5,s2,32) → pool2 → GAP → FC11 |
| 参数量 | **3,195**（int8 权重 3,136 B + int32 偏置 236 B） |
| 测试集准确率（全 SNR） | **52.34%**（int8 **52.05%**，量化掉点 0.28 pt） |
| 测试集准确率（SNR ≥ -10 dB） | **66.53%**（int8 **66.16%**） |
| 高 SNR（0~18 dB） | 72~80% |
| 主要短板 | QAM16 与 QAM64 混淆（QAM16 仅 5%）、8PSK/WBFM 偏弱 |
| PC 端单次推理 | ≈50 µs（MSVC /O2，x86） |
| STM32F103 @72 MHz | **实测 15.50 ms/次**（1,115,788 cycles，DWT；优化前 33.2 ms） |
| 固件占用 | Flash **29.6 KB（5.8% / 512 KB）**，RAM **15.8 KB（24.7% / 64 KB）** |
| 板端一致性 | 11 类概率与 PC/numpy 完全一致（`tests/test_model_parity.py` 300/300） |

> 对照：旧版 550 条（2018 子集）模型在同一 2016 测试集上只有 **8.81%**（≈随机，跨数据集不迁移），
> 说明必须用目标场景数据训练——这是本轮用完整 2016.10a 重训的直接原因。

## 目录结构

```
edge-inference/
├── start_demo.bat         # 一键启动 Web 看板（板子结果回传 PC）
├── csrc/
│   ├── nn.h / nn.c        # 纯 C 推理库：int8 fc/conv1d/pool/requantize + f32 GRU/LayerNorm/attention
│   └── demo.c             # PC 端示例：内置样本 -> model_forward -> 打印概率与耗时
├── demo/                  # 板子 → PC 实时看板
│   ├── server.py          # 串口解析 + HTTP 服务（零 Web 框架，仅 pyserial）
│   └── static/index.html  # 看板前端（类别/概率/时延曲线/历史）
├── models/                # 由 tools/train_radio2016.py 生成
│   ├── model.h / model.c  # int8 权重数组 + model_forward()
│   ├── demo_sample.h      # 一条真实样本的 int8 特征（固件内置）
│   ├── modrec_meta.json   # 量化参数（scale / fx mul-shift / 形状 / 预处理说明）
│   ├── modrec_metrics.json# 测试集指标、逐 SNR/逐类、混淆矩阵
│   ├── parity_features.npz# 300 条测试特征（C/numpy 对齐测试用）
│   └── modrec_cnn.pt      # PyTorch 权重（复现/再量化用）
├── tools/
│   ├── train_radio2016.py # 完整 2016.10a 训练 + 量化 + 导出（GPU）
│   ├── train_modrec.py    # 旧 550 条 2018 子集训练脚本（保留作对比）
│   ├── quantize.py        # 量化 / 整数前向模拟 / C 代码生成（定点化）
│   ├── reexport.py        # 不重训、只重新量化导出（旧 550 模型）
│   └── export_model.py    # 旧 Phase 1 两层 MLP 导出（保留作对比）
├── tests/
│   ├── test_conv_parity.py   # C 算子 vs numpy：57 用例 / 10451 元素 / 100% 逐位一致
│   ├── test_model_parity.py  # 导出 model.c vs numpy：300/300 argmax 一致，概率误差 5e-10
│   ├── test_seq_ops_parity.py# GRU/LayerNorm/attention vs numpy：102 元素，误差 ~2e-7
│   └── test_roundtrip.py     # 旧 Phase 1 MLP 对齐测试
├── firmware/
│   ├── src/main.c         # USART1 + 72 MHz PLL + DWT 时延统计 + 分阶段 benchmark + RAM 遥测
│   ├── build.bat          # arm-none-eabi-gcc 一键编译（elf/bin/hex/size）
│   ├── tools/flash.bat    # OpenOCD + CMSIS-DAP 一键烧录
│   ├── tools/view_serial.ps1  # 串口实时查看（由 查看串口输出.bat 调用）
│   ├── cmsis/             # 设备头 + 启动文件（fetch_cmsis.py 自动下载）
│   └── linker/stm32f103zet6.ld
├── docs/                  # 模型说明 / benchmark / 序列算子 / 看板截图
└── data/                  # 数据集（需自行下载，不入库）
```

## 快速开始

### 0. 环境

- Python 3.12 + numpy/pandas/pyarrow/torch（本机在 `<本机路径>
- pyserial（Web 看板用；本机已内置在 `.deps`，`demo/server.py` 会自动加载）
- MSVC（PC 端编译 C）、ARM GNU Toolchain 14.2 + OpenOCD（固件 / 烧录）

```powershell
$py = "<本机路径>"
$env:PYTHONPATH = "<本机路径>"
cd <仓库路径>
```

### 1. 下载数据集（RadioML 2016.10a，约 203 MB）

- 页面：<https://zenodo.org/records/18397070>（CC BY-NC-SA 4.0，学术用途）
- 下载 `RML2016.10a.tar.bz2` → 解压得 `RML2016.10a_dict_optimized.pkl`（225 MB），放到 `data/` 下。

### 2. 训练 + 量化 + 导出 C 数组（GPU 约 3~5 分钟）

```powershell
& $py tools\train_radio2016.py --epochs 200 --no-noise-aug
```

输出：`models/model.{h,c}`、`demo_sample.h`、`modrec_meta.json`、`modrec_metrics.json`、
`parity_features.npz`、`modrec_cnn.pt`；脚本内含分层 80/10/10 划分、逐 SNR/逐类指标、
量化前后精度对比。

### 3. 跑 PC 端 C demo

```powershell
cmd /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && cd /d <仓库路径> && cl /nologo /O2 /utf-8 /W4 /Icsrc /Imodels csrc\nn.c models\model.c csrc\demo.c /Fe:demo.exe && demo.exe"
```

期望：`predicted : 9 (QPSK)   true: 9 (QPSK)`，11 类概率和 ≈1。

### 4. 跑对齐测试（不需要板子）

```powershell
& $py tests\test_conv_parity.py     # C 算子 vs numpy：100% 逐位一致
& $py tests\test_model_parity.py    # 导出模型 vs numpy：300/300 argmax 一致
& $py tests\test_seq_ops_parity.py  # GRU/LayerNorm/attention vs numpy：误差 ~2e-7
```

### 5. 编译 / 烧录固件

```powershell
cd firmware
.\build.bat        # -> build\firmware.elf/.bin/.hex + size
.\tools\flash.bat  # 接 CMSIS-DAP 后：program + verify + reset
```

串口 USART1（PA9=TX, PA10=RX, 115200 8N1）每 ~1.7 s 打印一帧：模型信息、11 类概率、
`latency min/avg/max cycles & us`。只想看串口：双击仓库根的 `查看串口输出.bat`。

### 6. 板子分类结果回传 PC（Web 实时看板）

双击仓库根的 **`start_demo.bat`**，浏览器自动打开 <http://127.0.0.1:8765>：
当前类别 + 置信度 + MATCH、11 类概率条、板上 DWT 实测时延与最近 60 帧曲线、
板端信息与最近识别记录。命令行等价：`python demo\server.py --port COM6`；
无板子可 `--simulate` 演示。详见 [demo/README.md](demo/README.md)。

## 量化方案（与硬件匹配）

- **int8 权重 / int8 激活 / int32 累加**：F103 无 FPU，矩阵与卷积用整数乘加；
- **逐层对称量化**：`s_w = max|W|/127`，激活 `s_out` 由校准集确定；
- **重量化改定点乘移**：`y = clamp((acc*mul + (1<<(shift-1))) >> shift)`，`mul/2^24 ≈ req`，
  比 float 软浮点快 **11.3×**（单算子实测）；
- **卷积/池化/fc/requantize 热循环搬到 SRAM 执行**（`.ramfunc`，952 B），取指无 Flash 等待；
- 最后一层 logits 留 int32，softmax 用浮点；所有 buffer 静态分配、无动态内存。

## 已知限制 / 下一步

1. **精度**：全 SNR 52.3%、SNR≥-10 66.5%、高 SNR 72~80%；
   QAM16↔QAM64、8PSK/WBFM 是主要混淆（见 [docs/radio2016_模型.md](docs/radio2016_模型.md)）。
   继续提升需要更强前端/更大模型，或换 H743 等带 FPU 的芯片。
2. **时延**：15.5 ms/次，演示够用；继续优化（unroll/SMLAD）预计 9~11 ms。
3. **序列算子**：GRU/LayerNorm/attention 目前是 float32 参考实现，F103 上较慢；
   int8 定点化（LUT sigmoid/tanh）与端到端 RNN 部署为后续工作（见 [docs/序列算子.md](docs/序列算子.md)）。
4. **开源**：LICENSE / git / 文档已就绪，待推送远程仓库。

## 发布到 GitHub / Gitee

仓库已在本地完成 `git init` 并提交（分支 `main`）：

```bash
git remote add origin <你的仓库地址>
git push -u origin main
```

说明：
- `data/*.parquet`、`data/*.pkl`、`.deps/`、`firmware/build/` 已在 `.gitignore`，不会上传；
- `models/*.pt`、`models/parity_features.npz` 会提交（几百 KB，保证测试可复现）；
- 数据集需按 `data/README.md`、`docs/radio2016_模型.md` 自行下载；
- 许可证：代码 MIT（见 `LICENSE`），数据集各自遵循其原始协议。

## 参考

- 数据：[RadioML 2016.10a](https://zenodo.org/records/18397070)（CC BY-NC-SA 4.0）
- 早期 550 条子集：[alirezaaminzadeh/radioml-2018-samples](https://huggingface.co/datasets/alirezaaminzadeh/radioml-2018-samples)（MIT）
- 板卡：野火 STM32F103ZET6「霸道 V2」——512 KB Flash / 64 KB SRAM / Cortex-M3 @72 MHz / 无 FPU
- 推理库、量化、导出、看板均为本仓库自有实现，无第三方推理框架依赖（Web 看板仅依赖 pyserial）。
