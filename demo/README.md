# Web 实时看板：板子分类结果 → PC

把 STM32F103 的分类结果实时显示在浏览器里（类似上一阶段的语义通信 Demo）。

## 链路

```
STM32F103ZET6 --USART1/CH340(COM6,115200)--> PC 串口 --> demo/server.py --> HTTP --> 浏览器看板
      int8 CNN 推理（33 ms/次）                 解析帧               每 500 ms 刷新
```

## 一键启动

双击仓库根目录的 **`start_demo.bat`**（或命令行 `python demo\server.py`），
浏览器会自动打开 <http://127.0.0.1:8765>。

## 命令行

```powershell
python demo\server.py                     # 自动找 CH340 串口
python demo\server.py --port COM6         # 指定串口
python demo\server.py --simulate          # 无板子演示（生成模拟数据）
python demo\server.py --replay log.txt    # 回放串口日志
python demo\server.py --http-port 8765 --no-browser
```

依赖：`pyserial`（`pip install pyserial`）。本机已内置在项目 `.deps` 里，
`server.py` 会自动把它加入 `sys.path`，无需手动设置 PYTHONPATH。

## 看板内容

- 当前识别类别 + 置信度 + 与 PC 参考是否一致（MATCH/MISMATCH）；
- 11 类概率条（板端 int8 推理输出）；
- 单次推理时延：板上 DWT 实测的 cycles / ms，附最近 60 帧曲线；
- 板端信息：模型、输入、时钟、帧数、更新时刻、链路；
- 最近 8 条识别记录。

![看板截图](../docs/dashboard_demo.png)

## 串口帧格式（server.py 解析的协议）

固件每 ~1.7 秒打印一帧（`TIMING_RUNS=20` 时延统计 + 1 秒周期）：

```
========================================
model  : radio_ml_cnn
input  : C=2 L=126 int8, classes=11, params=3195
clock  : 72000000 Hz (PLL), dwt=ok
demo   : true=0, pc_ref_pred=0
  class  0: prob=0.4114
  ...
pred   : 0 (ref 0) MATCH
latency: min=2391847 avg=2391847 max=2391847 cycles (N=20)
latency: min=33220.10 avg=33220.10 max=33220.10 us
========================================
```

> 注意：CH340 的 DTR 常接板子复位脚，`server.py` 打开串口后会立即把
> DTR/RTS 置低，否则板子会被一直按在复位状态（已踩坑修复）。
