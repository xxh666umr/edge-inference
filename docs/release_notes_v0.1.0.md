# v0.1.0 发布说明

> 首个可运行版本：把 tiny CNN 调制识别模型部署到 STM32F103ZET6，
> 端到端跑通「训练 → int8 量化 → C 数组 → 板载推理 → Web 实时看板」。

## 亮点

| 项目 | 结果 |
|---|---|
| 数据集 | RadioML 2016.10a：11 类 × 20 SNR（-20~18 dB）× 1000 = **220,000** 条 |
| 模型 | tiny 1D-CNN，**3,195 参数**（2×128 int8 输入） |
| 精度 | 测试集全 SNR **52.3%**，**SNR ≥ -10 dB 66.5%**，高 SNR 72~80% |
| int8 量化 | 掉点仅 **0.28 pt**；定点重量化（mul/shift）替代软浮点 |
| MCU 时延 | **15.50 ms/次** @72 MHz（DWT 实测，优化前 33.2 ms，−53%） |
| 资源占用 | Flash **29.6 KB (5.8%)**、RAM **15.8 KB (24.7%)** |
| 结果回传 | USART1 → `demo/server.py` → 浏览器实时看板 |
| 序列算子 | GRU / LayerNorm / attention（float32 参考实现，numpy 对齐） |
| 测试 | C 算子 10,451 元素 100% 逐位一致；模型 300/300；序列算子误差 ~2e-7 |

## 主要内容

- `csrc/`：纯 C 推理库（int8 conv1d / maxpool / avgpool / fc / softmax / 定点重量化；
  float32 GRU / LayerNorm / attention）
- `firmware/`：STM32F103ZET6 固件（USART1 + 72 MHz PLL + DWT 计时 + 分阶段 benchmark + RAM 遥测）
- `models/`：导出的 int8 C 数组、量化参数、指标、parity 特征
- `demo/`：板子 → PC 的 Web 实时看板（零 Web 框架，仅 pyserial）
- `tools/`：训练 / 量化 / 导出脚本（RadioML 2016.10a 与早期 550 条子集两套）
- `tests/`：三套 C vs numpy 对齐测试（CI 自动运行）
- `docs/`：模型说明、benchmark、序列算子

## 复现

```powershell
# 数据：https://zenodo.org/records/18397070 （CC BY-NC-SA 4.0，学术用途）
python tools/train_radio2016.py --epochs 200 --no-noise-aug   # GPU 约 3~5 分钟
python tests/test_conv_parity.py
python tests/test_model_parity.py
python tests/test_seq_ops_parity.py
```

固件：`firmware/build.bat` 编译，`firmware/tools/flash.bat` 烧录（OpenOCD + CMSIS-DAP）。
看板：双击 `start_demo.bat`，浏览器打开 http://127.0.0.1:8765。

## 已知限制

- 全 SNR 精度 52.3%，低 SNR（≤-12 dB）接近随机；QAM16 与 QAM64 混淆明显（QAM16 仅 5%）；
- 序列算子目前是 float32 参考实现，F103 上较慢，int8 定点化为后续工作；
- 时延 15.5 ms 适合演示，连续实时解调需进一步优化或换 HPA（H743 等）。

## 许可证

代码 MIT；数据集 RadioML 2016.10a 为 CC BY-NC-SA 4.0（仅学术/非商业用途，不随仓库分发）。
