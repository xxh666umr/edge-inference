"""把两层的 MLP 分类网络量化成 int8，并导出为 C 头文件。

【旧版 Phase 1 参考实现】当前主线是 CNN 调制识别：tools/train_modrec.py + tools/quantize.py。
本脚本保留用于对比/教学，运行后生成 models/legacy_mlp_demo.h（不会覆盖 models/model.h）。

量化方案（对称、逐张量、逐层 scale）：
  - 输入：x_q = clamp(round(x / s_in), -128, 127)
  - 每层 Linear：W_q = clamp(round(W / s_w), -128, 127)，s_w = max|W|/127
                 b_q = round(b / (s_in * s_w))
                 acc = W_q · x_q + b_q   (int32)
  - 输出重新量化为 int8：s_out 由校准数据确定，req = (s_in*s_w)/s_out
    y_q = clamp(round(acc * req), 0, 127)   （带 ReLU）
  - 最后层 logits 保持 int32，softmax 用浮点：logit = acc * (s_in*s_w)

只支持 nn.Sequential(Linear, ReLU, Linear) 这种两层 MLP（框架第一步，后续扩展）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


def _q(x: np.ndarray, scale: float, lo=-128, hi=127) -> np.ndarray:
    return np.clip(np.round(x / scale), lo, hi).astype(np.int64)


def quantize_mlp(model, calib: np.ndarray, input_scale: float | None = None):
    """model: nn.Sequential(Linear(in,h), ReLU, Linear(h,C))。返回量化参数 dict。"""
    with torch.no_grad():
        l1 = model[0]
        l2 = model[2]
        W1 = l1.weight.detach().cpu().numpy()  # (h, in)
        b1 = l1.bias.detach().cpu().numpy()    # (h,)
        W2 = l2.weight.detach().cpu().numpy()  # (C, h)
        b2 = l2.bias.detach().cpu().numpy()    # (C,)

    in_dim = W1.shape[1]
    h = W1.shape[0]
    C = W2.shape[0]
    calib = np.asarray(calib, dtype=np.float32)

    # 输入 scale
    s_in = input_scale if input_scale else float(np.abs(calib).max() / 127.0)
    s_in = max(s_in, 1e-6)

    # 层 1 权重/偏置
    s_w1 = float(np.abs(W1).max() / 127.0) or 1e-6
    W1q = _q(W1, s_w1)
    acc1_s = s_in * s_w1
    b1q = np.clip(np.round(b1 / acc1_s), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int64)

    # 校准层 1 输出，得到 h 的 output scale
    xq = _q(calib, s_in)
    h_acc = (W1q @ xq.T + b1q[:, None]).astype(np.int64)   # (h, B)
    h_relu = np.maximum(h_acc, 0).astype(np.int64)
    h_float = h_relu * acc1_s
    s_out1 = float(np.abs(h_float).max() / 127.0) or 1e-6
    req1 = acc1_s / s_out1

    # 层 2
    s_w2 = float(np.abs(W2).max() / 127.0) or 1e-6
    W2q = _q(W2, s_w2)
    acc2_s = s_out1 * s_w2
    b2q = np.clip(np.round(b2 / acc2_s), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int64)

    return {
        "in_dim": in_dim, "hidden": h, "classes": C,
        "input_scale": s_in,
        "layer1": {"w": W1q, "b": b1q, "w_scale": s_w1,
                   "acc_scale": acc1_s, "out_scale": s_out1, "req_scale": req1},
        "layer2": {"w": W2q, "b": b2q, "w_scale": s_w2,
                   "acc_scale": acc2_s},
    }


def quantized_forward_numpy(q, x: np.ndarray):
    """用与 C 相同的整数运算重算前向，返回 (argmax, probs)。"""
    x = np.asarray(x, dtype=np.float32)
    l1 = q["layer1"]
    l2 = q["layer2"]
    xq = _q(x, q["input_scale"])
    h_acc = (l1["w"] @ xq + l1["b"]).astype(np.int64)
    h_relu = np.maximum(h_acc, 0)
    hq = _q(h_relu * l1["req_scale"], 1.0, lo=-128, hi=127)  # round(acc*req)
    logits = (l2["w"] @ hq + l2["b"]).astype(np.float64) * l2["acc_scale"]
    logits -= logits.max()
    e = np.exp(logits)
    probs = e / e.sum()
    return int(probs.argmax()), probs.astype(np.float64)


def _c_arr(name, a, dtype="int8_t", per=12):
    rows = []
    for i in range(0, len(a), per):
        rows.append(", ".join(str(int(v)) for v in a[i:i + per]))
    body = ",\n  ".join(rows)
    return f"static const {dtype} {name}[{len(a)}] = {{\n  {body}\n}};"


def write_model_h(q, path):
    l1, l2 = q["layer1"], q["layer2"]
    lines = []
    lines.append("/* 自动生成：int8 两层 MLP（见 tools/export_model.py） */")
    lines.append("#ifndef MODEL_H")
    lines.append("#define MODEL_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define IN_DIM    {q['in_dim']}")
    lines.append(f"#define HIDDEN    {q['hidden']}")
    lines.append(f"#define NUM_CLASSES {q['classes']}")
    lines.append(f"#define INPUT_SCALE {q['input_scale']:.9f}f")
    lines.append(f"#define L1_REQ_SCALE {l1['req_scale']:.9f}f")
    lines.append(f"#define L2_ACC_SCALE {l2['acc_scale']:.9f}f")
    lines.append("")
    lines.append(_c_arr("W1", l1["w"].reshape(-1)))
    lines.append(_c_arr("B1", l1["b"], dtype="int32_t"))
    lines.append(_c_arr("W2", l2["w"].reshape(-1)))
    lines.append(_c_arr("B2", l2["b"], dtype="int32_t"))
    lines.append("")
    lines.append("#endif")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _to_py(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_py(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    # 生成一个演示用的两层 MLP（8→16→4，随机权重）
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    calib = np.random.default_rng(0).normal(size=(256, 8)).astype(np.float32)
    q = quantize_mlp(model, calib)
    path = write_model_h(q, os.path.join("models", "legacy_mlp_demo.h"))
    with open(os.path.join("models", "legacy_mlp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(_to_py(q), f, indent=2)
    print("wrote", path)
    print(f"mlp {q['in_dim']}->{q['hidden']}->{q['classes']}, "
          f"W1 {q['layer1']['w'].size} int8, W2 {q['layer2']['w'].size} int8")
    # 与 csrc/demo.c 中固定的演示输入保持一致，用于核对 C 输出
    demo_x = np.array([20, -30, 5, 120, -60, 0, 80, -10], dtype=np.int64)
    arg, probs = quantized_forward_numpy(q, demo_x * q["input_scale"])
    print("expected demo probs (numpy):", np.round(probs, 4).tolist(), "argmax", arg)


if __name__ == "__main__":
    main()
