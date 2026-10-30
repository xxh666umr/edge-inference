"""int8 推理库（csrc/nn.c）与 numpy 参考实现的逐位对齐测试。

测试内容
--------
1. 单算子逐位比对：
   - conv1d_accumulate：int32 acc，随机多组 (c_in, c_out, l_in, k, stride)，含 NULL bias；
   - requantize：clamp(round(acc*scale), -128, 127)，含 .5 边界、饱和、随机 scale；
   - maxpool1d / avgpool1d：int8 输出，含负数、重复值、.5 取整边界；
2. 整链逐位比对：
   conv1d -> requantize_relu -> maxpool1d -> matmul_accumulate -> softmax_argmax，
   比对 int32 acc、int8 激活、int8 池化结果、int32 logits 与最终 argmax。

C 侧做法
--------
脚本把测试数据直接嵌入生成的 C 探针（tmp/clib_probe/probe.c），用本机 MSVC 编译
（vcvars64.bat + cl），运行探针并解析 stdout。.c/.obj/.exe 只写工作区临时目录，
不写入仓库。探针 include 仓库的 "nn.h"，链接仓库的 nn.c，因此测的是真实仓库代码。

运行
----
    python tests/test_conv_parity.py

可选环境变量
------------
    EDGE_INFERENCE_ROOT  仓库根目录（默认本文件所在 tests/ 的上一级）
    CLIB_PROBE_DIR       探针中间产物目录（默认系统临时目录下的 edge_inference_clib_probe）
    VCVARS64             vcvars64.bat 的绝对路径（默认自动搜索 VS 2022）

退出码：0 = PASS（100% 逐位一致），1 = FAIL。
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Windows 控制台 / 管道默认编码常是 GBK(cp936)，统一成 UTF-8，避免中文输出乱码。
# 仅影响本脚本的打印编码，不影响任何比对逻辑；老版本 Python 或已替换的流会静默跳过。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

# ---------------------------------------------------------------- 路径 / 工具链

REPO_ROOT = Path(
    os.environ.get("EDGE_INFERENCE_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
CSRC_DIR = REPO_ROOT / "csrc"

PROBE_DIR = Path(
    os.environ.get(
        "CLIB_PROBE_DIR",
        os.path.join(tempfile.gettempdir(), "edge_inference_clib_probe"),
    )
).resolve()

# 比对标签 -> 可读名称
LABELS = {
    "CONV": "conv1d_accumulate（int32 acc）",
    "REQ": "requantize（int8）",
    "MAXP": "maxpool1d（int8）",
    "AVGP": "avgpool1d（int8）",
    "CHAIN_ACC": "整链 conv1d 累加（int32）",
    "CHAIN_RELU": "整链 requantize_relu（int8）",
    "CHAIN_POOL": "整链 maxpool1d（int8）",
    "CHAIN_LOGITS": "整链 matmul_accumulate（int32 logits）",
}


def find_vcvars64() -> Path:
    """定位 MSVC 的 vcvars64.bat。"""
    env = os.environ.get("VCVARS64")
    if env and Path(env).is_file():
        return Path(env)
    roots = []
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        v = os.environ.get(var)
        if v:
            roots.append(v)
    candidates = []
    for root in roots:
        for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
            candidates.append(
                Path(root) / "Microsoft Visual Studio" / "2022" / edition
                / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            )
    for path in candidates:
        if path.is_file():
            return path
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    raise RuntimeError(
        "找不到 MSVC vcvars64.bat，请设置环境变量 VCVARS64 指向该文件"
    )


# ---------------------------------------------------------------- numpy 参考实现
#
# 所有取整规则与 C 实现严格对应：
#   round_away：四舍五入、远离零（0.5 -> 1，-0.5 -> -1），
#   用 float32 运算 + 向零截断模拟 C 的 (int32_t)(v ± 0.5f)。

def ref_round_away(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    half = np.float32(0.5)
    r = np.where(v >= np.float32(0.0), np.floor(v + half), np.ceil(v - half))
    return r.astype(np.int64)


def ref_clamp_i8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -128, 127).astype(np.int8)


def ref_conv1d(x: np.ndarray, w: np.ndarray, bias, k: int, stride: int) -> np.ndarray:
    """x [c_in][l_in] int8, w [c_out][c_in][k] int8, bias [c_out] 或 None。"""
    c_in, l_in = x.shape
    c_out = w.shape[0]
    l_out = (l_in - k) // stride + 1
    acc = np.zeros((c_out, l_out), dtype=np.int64)
    if bias is not None:
        acc += np.asarray(bias, dtype=np.int64)[:, None]
    wmat = np.asarray(w, dtype=np.int64).reshape(c_out, -1)  # [c_out][c_in*k]，与 C 权重布局一致
    for t in range(l_out):
        patch = np.asarray(x[:, t * stride:t * stride + k], dtype=np.int64).reshape(-1)
        acc[:, t] += wmat @ patch
    return acc


def ref_requantize(acc: np.ndarray, scale: np.float32) -> np.ndarray:
    v = np.asarray(acc, dtype=np.int32).astype(np.float32) * np.float32(scale)
    return ref_clamp_i8(ref_round_away(v))


def ref_requantize_relu(acc: np.ndarray, scale: np.float32) -> np.ndarray:
    v = np.asarray(acc, dtype=np.int32).astype(np.float32) * np.float32(scale)
    r = ref_round_away(v)
    r = np.maximum(r, 0)
    return np.clip(r, -128, 127).astype(np.int8)


def ref_maxpool1d(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    c, l_in = x.shape
    l_out = (l_in - k) // stride + 1
    y = np.empty((c, l_out), dtype=np.int8)
    for t in range(l_out):
        y[:, t] = x[:, t * stride:t * stride + k].max(axis=1)
    return y


def ref_avgpool1d(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    c, l_in = x.shape
    l_out = (l_in - k) // stride + 1
    y = np.empty((c, l_out), dtype=np.int8)
    for t in range(l_out):
        s = np.asarray(x[:, t * stride:t * stride + k], dtype=np.int64).sum(axis=1)
        mean = s.astype(np.float32) / np.float32(k)  # 与 C 的 (float)sum / (float)k 对应
        y[:, t] = ref_clamp_i8(ref_round_away(mean))
    return y


def ref_matmul(w: np.ndarray, bias, x: np.ndarray) -> np.ndarray:
    out = np.asarray(w, dtype=np.int64) @ np.asarray(x, dtype=np.int64).reshape(-1)
    if bias is not None:
        out += np.asarray(bias, dtype=np.int64)
    return out


def ref_softmax_argmax(logits: np.ndarray, acc_scale: np.float32):
    """与 C 的 softmax_argmax 相同的 float/double 混合运算（只比 argmax）。"""
    v = np.asarray(logits, dtype=np.int32).astype(np.float32) * np.float32(acc_scale)
    mx = np.float32(v.max())
    e = np.exp((v.astype(np.float64) - float(mx))).astype(np.float32)
    total = float(np.sum(e.astype(np.float64)))
    probs = (e.astype(np.float64) / total).astype(np.float32)
    return int(np.argmax(probs)), probs.astype(np.float64)


# ---------------------------------------------------------------- 用例生成

def _rand_i8(rng, shape) -> np.ndarray:
    return rng.integers(-128, 128, size=shape, dtype=np.int64).astype(np.int8)


def gen_conv_cases(rng) -> list[dict]:
    # (c_in, c_out, l_in, k, stride)
    shapes = [
        (1, 4, 16, 3, 1),
        (1, 4, 16, 7, 1),
        (2, 8, 16, 5, 1),
        (2, 8, 61, 7, 2),
        (8, 16, 61, 7, 2),
        (8, 16, 64, 5, 1),
        (8, 16, 32, 3, 1),
        (4, 4, 21, 3, 2),
        (1, 1, 5, 3, 2),
        (2, 16, 61, 3, 1),
        (3, 8, 40, 5, 2),
        (1, 1, 7, 7, 1),   # k == l_in，l_out = 1
        (5, 4, 17, 5, 2),
        (2, 4, 10, 3, 2),  # l_in-k 不整除，验证 floor 行为仍与 numpy 一致
    ]
    cases = []
    for idx, (c_in, c_out, l_in, k, stride) in enumerate(shapes):
        x = _rand_i8(rng, (c_in, l_in))
        w = _rand_i8(rng, (c_out, c_in, k))
        bias = None if idx % 5 == 3 else rng.integers(
            -8000, 8001, size=c_out, dtype=np.int64).astype(np.int32)
        cases.append({
            "x": x, "w": w, "bias": bias, "k": k, "stride": stride,
            "c_in": c_in, "c_out": c_out, "l_in": l_in,
        })
    return cases


def gen_requant_cases(rng) -> list[dict]:
    cases = []

    def add(acc, scale):
        cases.append({"acc": np.asarray(acc, dtype=np.int32),
                      "scale": np.float32(scale)})

    # 舍入边界：正好 x.5（四舍五入远离零）
    add([0, 1, -1, 3, -3, 7, -7, 127, -127], 0.5)
    add([1, -1, 3, -3], 2.5)  # 2.5->3, 7.5->8, -2.5->-3, -7.5->-8
    add([1, 2, 3, 4, -1, -2, -3, -4], 1.0)
    add([5, -5], 0.1)  # 0.5 / -0.5 附近
    # 饱和到 [-128, 127]
    add([1000, -1000, 127, 128, -128, -129, 10 ** 7, -(10 ** 7)], 127.0)
    add([127, -128, 0, 1, -1], 1.0)
    # 随机 int32 累加值 + 随机 scale
    for i in range(10):
        n = [1, 5, 37, 128, 513][i % 5]
        acc = rng.integers(-(2 ** 31), 2 ** 31, size=n, dtype=np.int64).astype(np.int32)
        scale = np.float32(rng.uniform(1e-7, 1e-4))
        add(acc, scale)
    return cases


def gen_pool_cases(rng) -> tuple[list[dict], list[dict]]:
    # (c, l_in, k, stride)
    shapes = [
        (1, 16, 3, 1),
        (1, 16, 2, 2),
        (4, 61, 7, 2),
        (8, 32, 5, 1),
        (2, 20, 4, 2),
        (3, 17, 3, 2),
        (2, 9, 9, 1),   # k == l_in
        (5, 40, 3, 2),
        (2, 30, 6, 3),
    ]
    maxp, avgp = [], []
    for idx, (c, l_in, k, stride) in enumerate(shapes):
        x = _rand_i8(rng, (c, l_in))
        if idx == 0:  # 全负 + 含 -128，测 max 的负数比较
            x = np.full((c, l_in), -100, dtype=np.int8)
            x[0, 3] = -128
            x[0, 7] = -1
        maxp.append({"x": x, "k": k, "stride": stride, "c": c, "l_in": l_in})
        avgp.append({"x": x, "k": k, "stride": stride, "c": c, "l_in": l_in})

    # avgpool 的 .5 取整边界（手工构造，逐通道覆盖）
    x = np.array([
        [1, 2, 5, 6, -1, -2, -5, -6, 127, 126, -127, -126, 0, 1, -1, 0],
        [3, 4, -3, -4, 7, 8, -7, -8, 0, 1, 0, -1, 126, 127, -126, -127],
    ], dtype=np.int8)
    avgp.append({"x": x, "k": 2, "stride": 2, "c": 2, "l_in": 16})

    return maxp, avgp


def gen_chain_cases(rng) -> list[dict]:
    # (c_in, l_in, c_out, k, stride, pool_k, pool_stride, classes)
    specs = [
        (1, 32, 4, 3, 1, 2, 2, 4),
        (2, 64, 8, 5, 1, 2, 2, 8),
        (8, 61, 16, 7, 2, 2, 1, 4),
        (2, 40, 8, 3, 2, 2, 2, 4),
        (4, 24, 8, 5, 1, 3, 2, 6),
        (1, 16, 4, 7, 1, 2, 2, 3),
        (3, 32, 8, 5, 2, 3, 1, 5),
        (2, 48, 4, 3, 1, 4, 2, 8),
    ]
    cases = []
    for idx, (c_in, l_in, c_out, k, stride, pk, ps, classes) in enumerate(specs):
        x = _rand_i8(rng, (c_in, l_in))
        w1 = _rand_i8(rng, (c_out, c_in, k))
        b1 = None if idx % 4 == 2 else rng.integers(
            -8000, 8001, size=c_out, dtype=np.int64).astype(np.int32)
        acc = ref_conv1d(x, w1, b1, k, stride)
        max_abs = max(1.0, float(np.abs(acc).max()))
        for _attempt in range(50):
            # 让 ReLU 后的激活大致铺满 int8 范围
            req = np.float32(127.0 / max_abs * rng.uniform(0.6, 1.0))
            y1 = ref_requantize_relu(acc, req)
            y2 = ref_maxpool1d(y1, pk, ps)
            in_dim = int(y2.size)
            w2 = _rand_i8(rng, (classes, in_dim))
            b2 = None if idx % 3 == 1 else rng.integers(
                -5000, 5001, size=classes, dtype=np.int64).astype(np.int32)
            logits = ref_matmul(w2, b2, y2)
            # logits 的 float32 scale：压到 ~10 以内
            acc_scale = np.float32(10.0 / (127.0 * 127.0 * in_dim)
                                   * rng.uniform(0.5, 2.0))
            _arg, probs = ref_softmax_argmax(logits, acc_scale)
            order = np.sort(probs)[::-1]
            margin = float(order[0] - order[1]) if order.size > 1 else 1.0
            if margin > 1e-4:  # 保证 argmax 不落在浮点噪声上，比对才有意义
                break
        else:
            raise RuntimeError("整链用例无法生成足够大的 softmax 间隔")
        cases.append({
            "x": x, "w1": w1, "b1": b1, "k": k, "stride": stride,
            "req": req, "pool_k": pk, "pool_stride": ps,
            "w2": w2, "b2": b2, "acc_scale": acc_scale,
            "c_in": c_in, "l_in": l_in, "c_out": c_out, "classes": classes,
        })
    return cases


# ---------------------------------------------------------------- C 探针生成

def _c_ints(values, per_line: int = 16) -> str:
    vals = [str(int(v)) for v in np.asarray(values).reshape(-1)]
    rows = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return ",\n    ".join(rows)


def _arr(kind: str, name: str, values) -> list[str]:
    arr = np.asarray(values)
    return [f"static const {kind} {name}[{arr.size}] = {{",
            "    " + _c_ints(arr),
            "};"]


def _float_bits(v: np.float32) -> int:
    return int(np.float32(v).view(np.uint32))


def build_probe_source(conv_cases, req_cases, mp_cases, ap_cases, chain_cases) -> str:
    L: list[str] = []

    def add(*lines):
        L.extend(lines)

    add("/* 自动生成，勿手改：由 tests/test_conv_parity.py 生成的对齐探针。 */",
        '#include <stdint.h>',
        '#include <stdio.h>',
        '#include <string.h>',
        '#include "nn.h"',
        "",
        "/* 把 IEEE-754 float32 的位模式还原成 float（保证 scale 与 numpy 参考值逐位相同） */",
        "static float bits2f(uint32_t u) {",
        "    float f;",
        "    memcpy(&f, &u, sizeof f);",
        "    return f;",
        "}",
        "",
        "static void emit_i32(const char *tag, int id, const int32_t *v, int n) {",
        "    printf(\"%s %d %d\", tag, id, n);",
        "    for (int i = 0; i < n; ++i) {",
        "        printf(\" %d\", (int)v[i]);",
        "    }",
        "    printf(\"\\n\");",
        "}",
        "",
        "static void emit_i8(const char *tag, int id, const int8_t *v, int n) {",
        "    printf(\"%s %d %d\", tag, id, n);",
        "    for (int i = 0; i < n; ++i) {",
        "        printf(\" %d\", (int)v[i]);",
        "    }",
        "    printf(\"\\n\");",
        "}",
        "",
        "int main(void) {",
        "")

    # ---- conv1d_accumulate
    add("    /* ===== conv1d_accumulate：随机多形状，含 NULL bias ===== */")
    for idx, c in enumerate(conv_cases):
        l_out = (c["l_in"] - c["k"]) // c["stride"] + 1
        acc_n = c["c_out"] * l_out
        add("    {")
        add(*["        " + s for s in _arr("int8_t", f"cx{idx}", c["x"])])
        add(*["        " + s for s in _arr("int8_t", f"cw{idx}", c["w"])])
        if c["bias"] is not None:
            add(*["        " + s for s in _arr("int32_t", f"cb{idx}", c["bias"])])
        add(f"        int32_t cacc{idx}[{acc_n}];")
        bias_arg = f"cb{idx}" if c["bias"] is not None else "NULL"
        add(f"        conv1d_accumulate(cw{idx}, {bias_arg}, cx{idx}, cacc{idx},"
            f" {c['c_in']}, {c['c_out']}, {c['l_in']}, {c['k']}, {c['stride']});")
        add(f"        emit_i32(\"CONV\", {idx}, cacc{idx}, {acc_n});")
        add("    }", "")

    # ---- requantize
    add("    /* ===== requantize：.5 边界 / 饱和 / 随机 scale ===== */")
    for idx, c in enumerate(req_cases):
        n = int(c["acc"].size)
        add("    {")
        add(*["        " + s for s in _arr("int32_t", f"ra{idx}", c["acc"])])
        add(f"        static const uint32_t rs{idx}[1] = {{{_float_bits(c['scale'])}u}};")
        add(f"        int8_t ry{idx}[{n}];")
        add(f"        requantize(ra{idx}, ry{idx}, {n}, bits2f(rs{idx}[0]));")
        add(f"        emit_i8(\"REQ\", {idx}, ry{idx}, {n});")
        add("    }", "")

    # ---- maxpool1d
    add("    /* ===== maxpool1d：含负数 / 重复值 / k == l_in ===== */")
    for idx, c in enumerate(mp_cases):
        l_out = (c["l_in"] - c["k"]) // c["stride"] + 1
        n = c["c"] * l_out
        add("    {")
        add(*["        " + s for s in _arr("int8_t", f"mx{idx}", c["x"])])
        add(f"        int8_t my{idx}[{n}];")
        add(f"        maxpool1d(mx{idx}, my{idx}, {c['c']}, {c['l_in']},"
            f" {c['k']}, {c['stride']});")
        add(f"        emit_i8(\"MAXP\", {idx}, my{idx}, {n});")
        add("    }", "")

    # ---- avgpool1d
    add("    /* ===== avgpool1d：四舍五入远离零（含 .5 边界） ===== */")
    for idx, c in enumerate(ap_cases):
        l_out = (c["l_in"] - c["k"]) // c["stride"] + 1
        n = c["c"] * l_out
        add("    {")
        add(*["        " + s for s in _arr("int8_t", f"ax{idx}", c["x"])])
        add(f"        int8_t ay{idx}[{n}];")
        add(f"        avgpool1d(ax{idx}, ay{idx}, {c['c']}, {c['l_in']},"
            f" {c['k']}, {c['stride']});")
        add(f"        emit_i8(\"AVGP\", {idx}, ay{idx}, {n});")
        add("    }", "")

    # ---- 整链
    add("    /* ===== 整链：conv1d -> requantize_relu -> maxpool1d -> matmul -> softmax ===== */")
    for idx, c in enumerate(chain_cases):
        l_out = (c["l_in"] - c["k"]) // c["stride"] + 1
        acc_n = c["c_out"] * l_out
        pool_n = c["c_out"] * ((l_out - c["pool_k"]) // c["pool_stride"] + 1)
        in_dim = pool_n
        add("    {")
        add(*["        " + s for s in _arr("int8_t", f"tx{idx}", c["x"])])
        add(*["        " + s for s in _arr("int8_t", f"tw1_{idx}", c["w1"])])
        if c["b1"] is not None:
            add(*["        " + s for s in _arr("int32_t", f"tb1_{idx}", c["b1"])])
        add(*["        " + s for s in _arr("int8_t", f"tw2_{idx}", c["w2"])])
        if c["b2"] is not None:
            add(*["        " + s for s in _arr("int32_t", f"tb2_{idx}", c["b2"])])
        add(f"        static const uint32_t trs{idx}[1] = {{{_float_bits(c['req'])}u}};")
        add(f"        static const uint32_t tas{idx}[1] = {{{_float_bits(c['acc_scale'])}u}};")
        add(f"        int32_t tacc{idx}[{acc_n}];",
            f"        int8_t ty1_{idx}[{acc_n}];",
            f"        int8_t ty2_{idx}[{in_dim}];",
            f"        int32_t tlogits{idx}[{c['classes']}];",
            f"        float tprobs{idx}[{c['classes']}];",
            f"        int targ{idx};")
        b1_arg = f"tb1_{idx}" if c["b1"] is not None else "NULL"
        add(f"        conv1d_accumulate(tw1_{idx}, {b1_arg}, tx{idx}, tacc{idx},"
            f" {c['c_in']}, {c['c_out']}, {c['l_in']}, {c['k']}, {c['stride']});")
        add(f"        emit_i32(\"CHAIN_ACC\", {idx}, tacc{idx}, {acc_n});")
        add(f"        requantize_relu(tacc{idx}, ty1_{idx}, {acc_n}, bits2f(trs{idx}[0]));")
        add(f"        emit_i8(\"CHAIN_RELU\", {idx}, ty1_{idx}, {acc_n});")
        add(f"        maxpool1d(ty1_{idx}, ty2_{idx}, {c['c_out']}, {l_out},"
            f" {c['pool_k']}, {c['pool_stride']});")
        add(f"        emit_i8(\"CHAIN_POOL\", {idx}, ty2_{idx}, {in_dim});")
        b2_arg = f"tb2_{idx}" if c["b2"] is not None else "NULL"
        add(f"        matmul_accumulate(tw2_{idx}, {b2_arg}, ty2_{idx}, tlogits{idx},"
            f" {in_dim}, {c['classes']});")
        add(f"        emit_i32(\"CHAIN_LOGITS\", {idx}, tlogits{idx}, {c['classes']});")
        add(f"        targ{idx} = softmax_argmax(tlogits{idx}, {c['classes']},"
            f" bits2f(tas{idx}[0]), tprobs{idx});")
        add(f"        printf(\"CHAIN_ARGMAX %d %d\\n\", {idx}, targ{idx});")
        add("    }", "")

    add("    printf(\"PROBE_DONE\\n\");", "    return 0;", "}", "")
    return "\n".join(L)


def compile_and_run_probe() -> str:
    vcvars = find_vcvars64()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_c = PROBE_DIR / "probe.c"
    build_bat = PROBE_DIR / "build_probe.bat"
    probe_exe = PROBE_DIR / "probe.exe"
    if probe_exe.exists():
        probe_exe.unlink()

    build_bat.write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  echo [build] vcvars64 failed\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        f'cd /d "{PROBE_DIR}"\r\n'
        f'cl /nologo /O2 /W3 /utf-8 /I"{CSRC_DIR}" probe.c "{CSRC_DIR / "nn.c"}"'
        " /Fe:probe.exe\r\n"
        "if errorlevel 1 exit /b 1\r\n"
        "exit /b 0\r\n",
        encoding="ascii",
    )

    print(f"[build] cl: {' '.join([str(vcvars), str(probe_c)])}")
    build = subprocess.run(
        ["cmd", "/c", str(build_bat)],
        cwd=str(PROBE_DIR), capture_output=True, text=True,
        errors="replace", timeout=600,
    )
    if build.returncode != 0 or not probe_exe.is_file():
        sys.stdout.write(build.stdout or "")
        sys.stderr.write(build.stderr or "")
        raise RuntimeError("C 探针编译失败（见上方 cl 输出）")

    print("[run] probe.exe")
    run = subprocess.run(
        [str(probe_exe)], cwd=str(PROBE_DIR),
        capture_output=True, text=True, errors="replace", timeout=300,
    )
    if run.returncode != 0:
        sys.stderr.write(run.stdout or "")
        sys.stderr.write(run.stderr or "")
        raise RuntimeError(f"探针运行失败，exit={run.returncode}")
    return run.stdout


# ---------------------------------------------------------------- 结果解析 / 比对

PROBE_TAGS = ("CONV", "REQ", "MAXP", "AVGP",
              "CHAIN_ACC", "CHAIN_RELU", "CHAIN_POOL", "CHAIN_LOGITS")


def parse_probe_output(text: str) -> tuple[dict, dict]:
    values: dict[str, dict[int, np.ndarray]] = {tag: {} for tag in PROBE_TAGS}
    argmax: dict[int, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "PROBE_DONE":
            done = True
            continue
        if tag in PROBE_TAGS:
            cid, n = int(parts[1]), int(parts[2])
            vals = [int(p) for p in parts[3:]]
            if len(vals) != n:
                raise RuntimeError(f"探针输出行字段数不符: {line[:80]}")
            values[tag][cid] = np.asarray(vals, dtype=np.int64)
        elif tag == "CHAIN_ARGMAX":
            argmax[int(parts[1])] = int(parts[2])
        else:
            raise RuntimeError(f"无法识别的探针输出: {line[:80]}")
    if not done:
        raise RuntimeError("探针未输出 PROBE_DONE，可能中途崩溃")
    return values, argmax


def compare_arrays(tag, expected, actual) -> tuple[int, list[str]]:
    total = 0
    errors: list[str] = []
    for cid, exp in expected.items():
        total += int(exp.size)
        act = actual.get(cid)
        if act is None:
            errors.append(f"{tag} 用例 {cid}: 探针缺少该用例输出")
            continue
        if act.size != exp.size:
            errors.append(f"{tag} 用例 {cid}: 长度不符 C={act.size} ref={exp.size}")
            continue
        diff = np.flatnonzero(act != exp)
        if diff.size:
            i = int(diff[0])
            errors.append(
                f"{tag} 用例 {cid}: {int(diff.size)} 个元素不一致，"
                f"首个位置 {i}: C={int(act[i])} ref={int(exp[i])}")
    for cid in actual:
        if cid not in expected:
            errors.append(f"{tag} 用例 {cid}: 探针多出未定义用例")
    return total, errors


def main() -> int:
    print("=" * 72)
    print("int8 推理库逐位对齐测试（C 探针 vs numpy 参考实现）")
    print(f"仓库根目录 : {REPO_ROOT}")
    print(f"探针临时目录: {PROBE_DIR}")

    if not (CSRC_DIR / "nn.h").is_file() or not (CSRC_DIR / "nn.c").is_file():
        print(f"FAIL: 找不到 {CSRC_DIR}\\nn.h 或 nn.c")
        return 1
    try:
        vcvars = find_vcvars64()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"MSVC       : {vcvars}")

    # 1) 生成用例 + numpy 参考结果
    rng = np.random.default_rng(20240601)
    conv_cases = gen_conv_cases(rng)
    req_cases = gen_requant_cases(rng)
    mp_cases, ap_cases = gen_pool_cases(rng)
    chain_cases = gen_chain_cases(rng)

    expected: dict[str, dict[int, np.ndarray]] = {tag: {} for tag in PROBE_TAGS}
    expected_argmax: dict[int, int] = {}
    for idx, c in enumerate(conv_cases):
        expected["CONV"][idx] = ref_conv1d(
            c["x"], c["w"], c["bias"], c["k"], c["stride"]).reshape(-1)
    for idx, c in enumerate(req_cases):
        expected["REQ"][idx] = ref_requantize(c["acc"], c["scale"]).reshape(-1)
    for idx, c in enumerate(mp_cases):
        expected["MAXP"][idx] = ref_maxpool1d(c["x"], c["k"], c["stride"]).reshape(-1)
    for idx, c in enumerate(ap_cases):
        expected["AVGP"][idx] = ref_avgpool1d(c["x"], c["k"], c["stride"]).reshape(-1)
    for idx, c in enumerate(chain_cases):
        acc = ref_conv1d(c["x"], c["w1"], c["b1"], c["k"], c["stride"])
        y1 = ref_requantize_relu(acc, c["req"])
        y2 = ref_maxpool1d(y1, c["pool_k"], c["pool_stride"])
        logits = ref_matmul(c["w2"], c["b2"], y2)
        arg, _probs = ref_softmax_argmax(logits, c["acc_scale"])
        expected["CHAIN_ACC"][idx] = acc.reshape(-1)
        expected["CHAIN_RELU"][idx] = y1.reshape(-1).astype(np.int64)
        expected["CHAIN_POOL"][idx] = y2.reshape(-1).astype(np.int64)
        expected["CHAIN_LOGITS"][idx] = logits.reshape(-1)
        expected_argmax[idx] = arg

    # 2) 生成并编译探针
    probe_c = PROBE_DIR / "probe.c"
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_c.write_text(
        build_probe_source(conv_cases, req_cases, mp_cases, ap_cases, chain_cases),
        encoding="utf-8",  # 探针里有中文注释，cl 用 /utf-8 编译
    )
    print(f"[gen] {probe_c} "
          f"({probe_c.stat().st_size} 字节, {len(conv_cases)} conv / "
          f"{len(req_cases)} requant / {len(mp_cases)} maxpool / "
          f"{len(ap_cases)} avgpool / {len(chain_cases)} 整链用例)")
    stdout = compile_and_run_probe()
    values, actual_argmax = parse_probe_output(stdout)

    # 3) 逐位比对
    print("-" * 72)
    total_elements = 0
    all_errors: list[str] = []
    for tag in PROBE_TAGS:
        n, errors = compare_arrays(tag, expected[tag], values[tag])
        total_elements += n
        all_errors.extend(errors)
        status = "OK" if not errors else f"{len(errors)} 处不一致"
        print(f"[{LABELS[tag]}] {len(expected[tag])} 用例, "
              f"比对 {n} 个元素, {status}")

    argmax_total = len(expected_argmax)
    argmax_errors = [f"整链用例 {cid}: argmax C={actual_argmax.get(cid)} ref={ref}"
                     for cid, ref in expected_argmax.items()
                     if actual_argmax.get(cid) != ref]
    all_errors.extend(argmax_errors)
    print(f"[{LABELS['CONV']} 之外的整链 argmax] {argmax_total} 用例, "
          f"比对 {argmax_total} 个 argmax, "
          f"{'OK' if not argmax_errors else f'{len(argmax_errors)} 处不一致'}")

    print("-" * 72)
    total_cases = (len(conv_cases) + len(req_cases) + len(mp_cases)
                   + len(ap_cases) + len(chain_cases))
    print(f"用例总数: {total_cases}（整链 {len(chain_cases)} 条）")
    print(f"比对元素总数: {total_elements + argmax_total} "
          f"（其中整数/整型 {total_elements}，argmax {argmax_total}）")
    if all_errors:
        print("不一致明细（最多 10 条）:")
        for msg in all_errors[:10]:
            print("  -", msg)
        mismatch = sum(1 for _ in all_errors)
        rate = 100.0 * (total_elements + argmax_total - mismatch) / max(
            1, total_elements + argmax_total)
        print(f"一致率: {rate:.4f}%  （{mismatch} 个比对单元不一致）")
        print("FAIL: C 实现与 numpy 参考实现未逐位对齐")
        return 1
    print("一致率: 100.0000%")
    print("PASS: C 实现与 numpy 参考实现逐位一致")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - 测试脚本需给出明确 FAIL 与非零退出码
        print(f"FAIL: {exc}")
        sys.exit(1)
