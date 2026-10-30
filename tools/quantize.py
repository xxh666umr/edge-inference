"""int8 量化与 C 数组导出（Phase 2：tiny CNN 调制识别）。

量化方案：对称、逐张量（per-tensor）、逐层 scale，int32 累加：
  - 输入特征 x：x_q = clamp(round(x / s_in), -128, 127)，s_in = max|x|/127
  - 卷积/全连接权重：s_w = max|W|/127，b_q = round(b / (s_in * s_w))
  - int32 累加：acc = sum(w_q * x_q) + b_q
  - 输出重量化：s_out 由校准集激活最大值确定，req = (s_in * s_w) / s_out
    y_q = clamp(round(acc * req), 0, 127)（带 ReLU）
  - 最后一层 logits 保持 int32，softmax 用浮点（与 csrc/nn.c 一致）

舍入规则：round half away from zero（与 C 端 `(int)(v + (v>=0?0.5f:-0.5f))` 一致）；
重量化/softmax 的浮点运算用 float32 模拟 C 的 `float`，保证 Python 与 C 逐位对齐。

布局约定（与 csrc/nn.h 完全一致）：
  - 特征张量：[C][L] 行主序（C 通道在前），本模型 C=2（I、Q）
  - 卷积权重：[C_out][C_in][K]
  - 卷积输出：[C_out][L_out]，L_out = (L_in - K) / stride + 1（valid padding）
"""
from __future__ import annotations

import json
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------- 基础工具
def round_away(v):
    """round half away from zero，返回整数值的浮点数组。"""
    v = np.asarray(v, dtype=np.float64)
    return np.trunc(v + np.copysign(0.5, v))


def quantize_sym(x, scale, lo=-128, hi=127):
    """对称量化 + clamp（round away from zero）。"""
    return np.clip(round_away(np.asarray(x, np.float64) / float(scale)), lo, hi).astype(np.int64)


def requantize_f32(acc, scale, relu=True):
    """模拟 C 端 requantize(_relu)：float32 乘 + round away + clamp。"""
    v = (np.asarray(acc, np.int64).astype(np.float32) * np.float32(scale)).astype(np.float32)
    # C: (int32_t)(v + (v >= 0 ? 0.5f : -0.5f))，加法在 float32 中进行
    r = np.trunc((v + np.copysign(np.float32(0.5), v)).astype(np.float32)).astype(np.int64)
    r = np.clip(r, -128, 127 if not relu else 127)
    if relu:
        r = np.maximum(r, 0)
    return r.astype(np.int64)


FX_SHIFT = 24   # 定点重量化的小数位数：req ≈ mul / 2^24
FX_MUL_MIN = 1


def fx_scale(req):
    """把 float 重量化系数转成定点 (mul, shift)，与 C 的 requantize_*_fx 对应。"""
    mul = int(round(float(req) * (1 << FX_SHIFT)))
    return max(mul, FX_MUL_MIN), FX_SHIFT


def requantize_fx(acc, mul, shift, relu=True):
    """模拟 C 端 requantize_relu_fx / requantize_fx（int64 乘 + 算术右移）。"""
    p = np.asarray(acc, np.int64) * np.int64(mul) + (np.int64(1) << np.int64(shift - 1))
    r = p >> np.int64(shift)
    if relu:
        return np.clip(r, 0, 127)
    return np.clip(r, -128, 127)


def avgpool_global_int(pooled):
    """全局平均池化（int8 输入，k=L，stride=1），与 C 整数除法语义一致。

    C 参考实现：s = sum(x); r = (s >= 0) ? (s + k/2) / k : (s - k/2) / k;
    其中 `/` 为 C 的向零截断整除。
    """
    x = np.asarray(pooled, np.int64)
    n, c, l = x.shape
    s = x.sum(axis=2)  # (N, C)
    k = l
    num = s + np.where(s >= 0, k // 2, -(k // 2))
    r = np.trunc(num / k).astype(np.int64)  # 向零截断
    return np.clip(r, -128, 127)


# ---------------------------------------------------------------- 整数前向（可批量）
def conv1d_acc(xq, wq, bq, stride):
    """xq: (N, C_in, L_in) int；wq: (C_out, C_in, K)；bq: (C_out,)；返回 int32 acc (N,C_out,L_out)。"""
    xq = np.asarray(xq, np.int64)
    wq = np.asarray(wq, np.int64)
    k = wq.shape[2]
    win = sliding_window_view(xq, k, axis=2)[:, :, ::stride]  # (N,C_in,L_out,K)
    # 注意：x 的通道标号必须与 w 的输入通道标号一致的字母（c），否则 c/i 会各自独立求和
    acc = np.einsum("nctk,ock->not", win, wq, optimize=True)
    return acc + np.asarray(bq, np.int64)[None, :, None]


def maxpool1d_int(xq, k, stride):
    """xq: (N,C,L) int8 -> (N,C,L_out) int8。"""
    xq = np.asarray(xq, np.int64)
    win = sliding_window_view(xq, k, axis=2)[:, :, ::stride]
    return win.max(axis=3)


def softmax_int8(logits, acc_scale):
    """与 csrc/nn.c:softmax_argmax 一致（float32 运算 + 双精度 exp 累加）。"""
    logits = np.asarray(logits, np.int64)
    scale = np.float32(acc_scale)
    z = (logits.astype(np.float32) * scale).astype(np.float32)  # (N,Cls)
    mx = z.max(axis=1, keepdims=True)
    e = np.exp((z - mx).astype(np.float64)).astype(np.float32)  # C: (float)exp(double)
    s = e.astype(np.float64).sum(axis=1, keepdims=True)
    probs = (e / s).astype(np.float32)
    return probs


# ---------------------------------------------------------------- CNN 量化
def quantize_cnn(model, calib):
    """model: CNN(2->c1->c2->fc)，输入 calib: (N,2,L) float32。返回量化参数 dict。"""
    import torch

    calib = np.asarray(calib, np.float32)
    if calib.ndim == 2:
        calib = calib.reshape(-1, 2, 126)
    n, c_in, l_in = calib.shape

    with torch.no_grad():
        w1 = model.c1.weight.detach().cpu().numpy().astype(np.float64)   # (c1,2,7)
        b1 = model.c1.bias.detach().cpu().numpy().astype(np.float64)
        w2 = model.c2.weight.detach().cpu().numpy().astype(np.float64)   # (c2,c1,5)
        b2 = model.c2.bias.detach().cpu().numpy().astype(np.float64)
        w3 = model.fc.weight.detach().cpu().numpy().astype(np.float64)   # (cls,c2)
        b3 = model.fc.bias.detach().cpu().numpy().astype(np.float64)
        k1, s1 = model.c1.kernel_size[0], model.c1.stride[0]
        k2, s2 = model.c2.kernel_size[0], model.c2.stride[0]
        p1, p2 = int(model.pool1.kernel_size), int(model.pool2.kernel_size)
    c_out1, c_out2 = w1.shape[0], w2.shape[0]
    l1 = (l_in - k1) // s1 + 1
    l1p = (l1 - p1) // p1 + 1
    l2 = (l1p - k2) // s2 + 1
    l2p = (l2 - p2) // p2 + 1

    # 输入
    s_in = max(float(np.abs(calib).max()) / 127.0, 1e-6)
    xq = quantize_sym(calib, s_in)  # (N,2,L)

    # conv1
    s_w1 = max(float(np.abs(w1).max()) / 127.0, 1e-6)
    w1q = quantize_sym(w1, s_w1)
    acc1_scale = s_in * s_w1
    b1q = np.clip(round_away(b1 / acc1_scale), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int64)
    acc1 = conv1d_acc(xq, w1q, b1q, s1)  # (N,c1,l1) int
    relu1 = np.maximum(acc1, 0)
    out1_scale = max(float((relu1 * acc1_scale).max()) / 127.0, 1e-6)
    req1 = acc1_scale / out1_scale
    mul1, shift1 = fx_scale(req1)
    h1 = requantize_fx(acc1, mul1, shift1, relu=True)  # (N,c1,l1) int8
    p1q = maxpool1d_int(h1, p1, p1)             # (N,c1,l1p)

    # conv2
    s_w2 = max(float(np.abs(w2).max()) / 127.0, 1e-6)
    w2q = quantize_sym(w2, s_w2)
    acc2_scale = out1_scale * s_w2
    b2q = np.clip(round_away(b2 / acc2_scale), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int64)
    acc2 = conv1d_acc(p1q, w2q, b2q, s2)
    relu2 = np.maximum(acc2, 0)
    out2_scale = max(float((relu2 * acc2_scale).max()) / 127.0, 1e-6)
    req2 = acc2_scale / out2_scale
    mul2, shift2 = fx_scale(req2)
    h2 = requantize_fx(acc2, mul2, shift2, relu=True)
    p2q = maxpool1d_int(h2, p2, p2)             # (N,c2,l2p)

    # 全局平均池化 + 全连接
    gap = avgpool_global_int(p2q)               # (N,c2) int8，scale = out2_scale
    s_w3 = max(float(np.abs(w3).max()) / 127.0, 1e-6)
    w3q = quantize_sym(w3, s_w3)
    fc_scale = out2_scale * s_w3
    b3q = np.clip(round_away(b3 / fc_scale), -2 ** 31 + 1, 2 ** 31 - 1).astype(np.int64)
    logits = (gap.astype(np.int64) @ w3q.T + b3q)  # (N,cls) int32

    return {
        "arch": "cnn",
        "classes": int(w3.shape[0]),
        "input": {"c": int(c_in), "l": int(l_in), "scale": float(s_in)},
        "conv1": {"c_in": int(c_in), "c_out": int(c_out1), "k": int(k1), "stride": int(s1),
                  "l_in": int(l_in), "l_out": int(l1), "w": w1q, "b": b1q,
                  "w_scale": float(s_w1), "acc_scale": float(acc1_scale),
                  "out_scale": float(out1_scale), "req_scale": float(req1),
                  "fx_mul": int(mul1), "fx_shift": int(shift1)},
        "pool1": {"k": int(p1), "stride": int(p1), "l_in": int(l1), "l_out": int(l1p)},
        "conv2": {"c_in": int(c_out1), "c_out": int(c_out2), "k": int(k2), "stride": int(s2),
                  "l_in": int(l1p), "l_out": int(l2), "w": w2q, "b": b2q,
                  "w_scale": float(s_w2), "acc_scale": float(acc2_scale),
                  "out_scale": float(out2_scale), "req_scale": float(req2),
                  "fx_mul": int(mul2), "fx_shift": int(shift2)},
        "pool2": {"k": int(p2), "stride": int(p2), "l_in": int(l2), "l_out": int(l2p)},
        "gap": {"k": int(l2p), "stride": 1},
        "fc": {"c_in": int(c_out2), "c_out": int(w3.shape[0]), "w": w3q, "b": b3q,
               "w_scale": float(s_w3), "acc_scale": float(fc_scale)},
        "param_count": int(w1q.size + b1q.size + w2q.size + b2q.size + w3q.size + b3q.size),
        "weight_bytes": int(w1q.size + w2q.size + w3q.size),
        "bias_bytes": int(4 * (b1q.size + b2q.size + b3q.size)),
    }


def quantized_forward_cnn(q, features, return_logits=False):
    """按 C 端完全相同的整数流程重算，返回 (argmax, probs) 或 logits。"""
    x = np.asarray(features, np.float32)
    if x.ndim == 2:
        x = x.reshape(-1, q["input"]["c"], q["input"]["l"])
    xq = quantize_sym(x, q["input"]["scale"])          # (N,2,L)
    c1, p1 = q["conv1"], q["pool1"]
    acc1 = conv1d_acc(xq, c1["w"], c1["b"], c1["stride"])
    h1 = requantize_fx(acc1, c1["fx_mul"], c1["fx_shift"], relu=True)
    p1q = maxpool1d_int(h1, p1["k"], p1["stride"])
    c2, p2 = q["conv2"], q["pool2"]
    acc2 = conv1d_acc(p1q, c2["w"], c2["b"], c2["stride"])
    h2 = requantize_fx(acc2, c2["fx_mul"], c2["fx_shift"], relu=True)
    p2q = maxpool1d_int(h2, p2["k"], p2["stride"])
    gap = avgpool_global_int(p2q)
    fc = q["fc"]
    logits = gap.astype(np.int64) @ fc["w"].T + fc["b"]
    if return_logits:
        return logits
    probs = softmax_int8(logits, fc["acc_scale"])
    return probs.argmax(axis=1), probs


# ---------------------------------------------------------------- C 导出
def _c_array(name, a, dtype="int8_t", per=12):
    flat = np.asarray(a).reshape(-1)
    rows = []
    for i in range(0, len(flat), per):
        rows.append(", ".join(str(int(v)) for v in flat[i:i + per]))
    body = ",\n  ".join(rows)
    return f"static const {dtype} {name}[{len(flat)}] = {{\n  {body}\n}};"


def _c_float(x):
    return f"{float(x):.9g}f"


def write_model(q, out_dir, name="radio_ml_cnn"):
    """生成 models/model.h + models/model.c（无需 Python 运行时即可推理）。"""
    os.makedirs(out_dir, exist_ok=True)
    c1, c2, fc = q["conv1"], q["conv2"], q["fc"]
    h = []
    h.append("/* 自动生成，请勿手改：tools/train_modrec.py -> tools/quantize.py */")
    h.append("#ifndef MODEL_H")
    h.append("#define MODEL_H")
    h.append("#include <stdint.h>")
    h.append("")
    h.append(f'#define MODEL_NAME "{name}"')
    h.append(f"#define MODEL_INPUT_C     {q['input']['c']}")
    h.append(f"#define MODEL_INPUT_LEN   {q['input']['l']}")
    h.append(f"#define MODEL_NUM_CLASSES {q['classes']}")
    h.append(f"#define MODEL_INPUT_SCALE {_c_float(q['input']['scale'])}")
    h.append(f"#define MODEL_PARAM_COUNT {q['param_count']}")
    h.append(f"#define MODEL_CONV1_FX_MUL   {c1['fx_mul']}")
    h.append(f"#define MODEL_CONV1_FX_SHIFT {c1['fx_shift']}")
    h.append(f"#define MODEL_CONV2_FX_MUL   {c2['fx_mul']}")
    h.append(f"#define MODEL_CONV2_FX_SHIFT {c2['fx_shift']}")
    h.append("")
    h.append("#ifdef __cplusplus")
    h.append('extern "C" {')
    h.append("#endif")
    h.append("")
    h.append("/* 输入: int8[MODEL_INPUT_C][MODEL_INPUT_LEN] 行主序（I 通道在前）。")
    h.append(" * 返回 argmax 类别，probs 填 MODEL_NUM_CLASSES 个概率。 */")
    h.append("int model_forward(const int8_t *input, float *probs);")
    h.append("")
    h.append("#ifdef __cplusplus")
    h.append("}")
    h.append("#endif")
    h.append("")
    h.append("#endif /* MODEL_H */")

    c = []
    c.append("/* 自动生成，请勿手改：tools/train_modrec.py -> tools/quantize.py */")
    c.append('#include "model.h"')
    c.append('#include "nn.h"')
    c.append("")
    c.append(f"#define CONV1_C_IN  {c1['c_in']}")
    c.append(f"#define CONV1_C_OUT {c1['c_out']}")
    c.append(f"#define CONV1_K     {c1['k']}")
    c.append(f"#define CONV1_STRIDE {c1['stride']}")
    c.append(f"#define CONV1_L_IN  {c1['l_in']}")
    c.append(f"#define CONV1_L_OUT {c1['l_out']}")
    c.append(f"/* 定点重量化：scale ≈ mul / 2^shift（F103 无 FPU，比软浮点快约 20 倍） */")

    c.append(f"#define POOL1_K     {q['pool1']['k']}")
    c.append(f"#define POOL1_L_OUT {q['pool1']['l_out']}")
    c.append(f"#define CONV2_C_IN  {c2['c_in']}")
    c.append(f"#define CONV2_C_OUT {c2['c_out']}")
    c.append(f"#define CONV2_K     {c2['k']}")
    c.append(f"#define CONV2_STRIDE {c2['stride']}")
    c.append(f"#define CONV2_L_IN  {c2['l_in']}")
    c.append(f"#define CONV2_L_OUT {c2['l_out']}")

    c.append(f"#define POOL2_K     {q['pool2']['k']}")
    c.append(f"#define POOL2_L_OUT {q['pool2']['l_out']}")
    c.append(f"#define FC_C_IN     {fc['c_in']}")
    c.append(f"#define FC_C_OUT    {fc['c_out']}")
    c.append(f"#define FC_ACC_SCALE {_c_float(fc['acc_scale'])}")
    c.append("")
    c.append(_c_array("W_CONV1", c1["w"]))
    c.append(_c_array("B_CONV1", c1["b"], dtype="int32_t"))
    c.append(_c_array("W_CONV2", c2["w"]))
    c.append(_c_array("B_CONV2", c2["b"], dtype="int32_t"))
    c.append(_c_array("W_FC", fc["w"]))
    c.append(_c_array("B_FC", fc["b"], dtype="int32_t"))
    c.append("")
    c.append("int model_forward(const int8_t *input, float *probs) {")
    c.append("    static int32_t acc1[CONV1_C_OUT * CONV1_L_OUT];")
    c.append("    static int8_t  act1[CONV1_C_OUT * CONV1_L_OUT];")
    c.append("    static int8_t  pool1[CONV1_C_OUT * POOL1_L_OUT];")
    c.append("    static int32_t acc2[CONV2_C_OUT * CONV2_L_OUT];")
    c.append("    static int8_t  act2[CONV2_C_OUT * CONV2_L_OUT];")
    c.append("    static int8_t  pool2[CONV2_C_OUT * POOL2_L_OUT];")
    c.append("    static int8_t  gap[FC_C_IN];")
    c.append("    static int32_t logits[FC_C_OUT];")
    c.append("")
    c.append("    conv1d_accumulate(W_CONV1, B_CONV1, input, acc1, CONV1_C_IN, CONV1_C_OUT,")
    c.append("                      CONV1_L_IN, CONV1_K, CONV1_STRIDE);")
    c.append("    requantize_relu_fx(acc1, act1, CONV1_C_OUT * CONV1_L_OUT,")
    c.append("                        MODEL_CONV1_FX_MUL, MODEL_CONV1_FX_SHIFT);")
    c.append("    maxpool1d(act1, pool1, CONV1_C_OUT, CONV1_L_OUT, POOL1_K, POOL1_K);")
    c.append("")
    c.append("    conv1d_accumulate(W_CONV2, B_CONV2, pool1, acc2, CONV2_C_IN, CONV2_C_OUT,")
    c.append("                      CONV2_L_IN, CONV2_K, CONV2_STRIDE);")
    c.append("    requantize_relu_fx(acc2, act2, CONV2_C_OUT * CONV2_L_OUT,")
    c.append("                        MODEL_CONV2_FX_MUL, MODEL_CONV2_FX_SHIFT);")
    c.append("    maxpool1d(act2, pool2, CONV2_C_OUT, CONV2_L_OUT, POOL2_K, POOL2_K);")
    c.append("")
    c.append("    avgpool1d(pool2, gap, FC_C_IN, POOL2_L_OUT, POOL2_L_OUT, 1);")
    c.append("    matmul_accumulate(W_FC, B_FC, gap, logits, FC_C_IN, FC_C_OUT);")
    c.append("    return softmax_argmax(logits, FC_C_OUT, FC_ACC_SCALE, probs);")
    c.append("}")

    h_path = os.path.join(out_dir, "model.h")
    c_path = os.path.join(out_dir, "model.c")
    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h) + "\n")
    with open(c_path, "w", encoding="utf-8") as f:
        f.write("\n".join(c) + "\n")
    return h_path, c_path


def write_demo_sample(path, x_int8, true_cls, pred_cls, class_name="", true_name="", pred_name=""):
    """生成 models/demo_sample.h：内含一条真实样本的 int8 特征，供 C demo / 固件使用。"""
    flat = np.asarray(x_int8, dtype=np.int64).reshape(-1)
    rows = []
    for i in range(0, len(flat), 16):
        rows.append(", ".join(str(int(v)) for v in flat[i:i + 16]))
    body = ",\n  ".join(rows)
    lines = [
        "/* 自动生成：一条来自 RadioML 数据集的调制识别样本（int8 特征，布局 [2][126]，I 在前） */",
        "#ifndef DEMO_SAMPLE_H",
        "#define DEMO_SAMPLE_H",
        "#include <stdint.h>",
        "",
        f"#define DEMO_TRUE_CLASS {int(true_cls)}",
        f"#define DEMO_PRED_CLASS {int(pred_cls)}",
        f'#define DEMO_TRUE_NAME  "{true_name}"',
        f'#define DEMO_PRED_NAME  "{pred_name}"',
        f"#define DEMO_INPUT_LEN  {len(flat)}",
        "",
        f"static const int8_t DEMO_INPUT[{len(flat)}] = {{",
        f"  {body}",
        "};",
        "",
        "#endif /* DEMO_SAMPLE_H */",
    ]
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


def save_meta(q, path, extra=None):
    meta = _to_py(q)
    if extra:
        meta.update(_to_py(extra))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path




