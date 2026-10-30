"""验证导出模型 models/model.c（C 推理）与 numpy 量化参考实现逐样本一致。

测试内容
--------
1. 从 models/modrec_meta.json 读回量化参数，用与 tools/quantize.py 相同的整数流程
   重算 550 条样本的 argmax 与 11 类概率；
2. 生成 C 探针（读 features.bin，逐条调用 model_forward），用 MSVC 编译
   models/model.c + csrc/nn.c，运行并解析输出；
3. 比对：argmax 必须 100% 一致；概率最大绝对误差 < 1e-4；
4. 自证 avgpool 的 numpy 整数公式与 C 的 float32 mean + round-away 在全范围一致
   （k=2/3/6，sum 覆盖 int8 池化所有可能值）。

运行（仓库根目录）：
    python tests/test_model_parity.py
退出码：0 = PASS，1 = FAIL。
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
MODELS_DIR = REPO_ROOT / "models"
CSRC_DIR = REPO_ROOT / "csrc"
PROBE_DIR = Path(os.environ.get(
    "MODEL_PROBE_DIR", os.path.join(tempfile.gettempdir(), "edge_inference_model_probe")))

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from quantize import quantized_forward_cnn  # noqa: E402

DATA_DEFAULT = REPO_ROOT / "data" / "train-00000-of-00001.parquet"


# ---------------------------------------------------------------- 工具链
def find_vcvars64() -> Path:
    env = os.environ.get("VCVARS64")
    if env and Path(env).is_file():
        return Path(env)
    roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)")]
    for root in filter(None, roots):
        for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
            p = Path(root) / "Microsoft Visual Studio" / "2022" / edition / \
                "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if p.is_file():
                return p
    for pattern in (r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
                    r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    raise RuntimeError("找不到 MSVC vcvars64.bat，请设置环境变量 VCVARS64")


def compile_and_run(vcvars: Path, probe_c: Path, exe: Path, features_bin: Path) -> str:
    """生成一个 build.bat（call vcvars 后再 cl），在探针目录里编译并运行。"""
    bat = PROBE_DIR / "build_probe.bat"
    content = (
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        "if errorlevel 1 exit /b 2\r\n"
        f'cl /nologo /O2 /utf-8 /W4 /I "{CSRC_DIR}" /I "{MODELS_DIR}" '
        f'"{probe_c}" "{MODELS_DIR / "model.c"}" "{CSRC_DIR / "nn.c"}" '
        f'/Fe:"{exe}"\r\n'
    )
    bat.write_text(content, encoding="ascii")
    r = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(PROBE_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"MSVC 编译失败:\n{r.stdout}\n{r.stderr}")
    run = subprocess.run([str(exe), str(features_bin)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=str(PROBE_DIR))
    if run.returncode != 0:
        raise RuntimeError(f"探针运行失败: {run.returncode}\n{run.stdout}\n{run.stderr}")
    return run.stdout


# ---------------------------------------------------------------- 量化参数 / 特征
def load_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    for layer in ("conv1", "conv2", "fc"):
        meta[layer]["w"] = np.asarray(meta[layer]["w"], dtype=np.int64)
        meta[layer]["b"] = np.asarray(meta[layer]["b"], dtype=np.int64)
    return meta


def build_features() -> tuple[np.ndarray, np.ndarray]:
    """复用训练脚本的预处理：截 1008 -> 功率归一化 -> 平均抽取 126。"""
    from train_modrec import decimate, load_iq

    data = Path(os.environ.get("MODREC_DATA", str(DATA_DEFAULT)))
    iq, y, _snr, _classes = load_iq(str(data))
    return decimate(iq), y


def check_avgpool_equivalence() -> tuple[int, int]:
    """numpy 整数公式 vs C 语义（float32 mean + round-away），全枚举比对。"""
    mismatches = 0
    total = 0
    for k in (2, 3, 6):
        for s in range(-128 * k, 127 * k + 1):
            # C 语义
            mean = np.float32(s) / np.float32(k)
            c_val = int(np.trunc(mean + np.copysign(np.float32(0.5), mean)))
            # numpy 整数公式
            num = s + (k // 2 if s >= 0 else -(k // 2))
            n_val = int(np.trunc(num / k))
            total += 1
            if c_val != n_val:
                mismatches += 1
    return total, mismatches


def build_probe_source() -> str:
    return r'''/* 自动生成：验证 model_forward 与 numpy 参考一致 */
#include <stdio.h>
#include <stdint.h>
#include "model.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: probe features.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    static int8_t x[MODEL_INPUT_C * MODEL_INPUT_LEN];
    static float probs[MODEL_NUM_CLASSES];
    int n = 0;
    while (fread(x, 1, sizeof(x), f) == sizeof(x)) {
        int a = model_forward(x, probs);
        printf("%d %d", n, a);
        for (int i = 0; i < MODEL_NUM_CLASSES; ++i) printf(" %.9g", (double)probs[i]);
        printf("\n");
        ++n;
    }
    fclose(f);
    fprintf(stderr, "samples=%d\n", n);
    return 0;
}
'''


# ---------------------------------------------------------------- 主流程
def main() -> int:
    meta_path = MODELS_DIR / "modrec_meta.json"
    model_c = MODELS_DIR / "model.c"
    model_h = MODELS_DIR / "model.h"
    for p in (meta_path, model_c, model_h):
        if not p.is_file():
            print(f"FAIL: 缺少 {p}")
            return 1

    print("=" * 72)
    print("导出模型 C 推理 vs numpy 量化参考：逐样本对齐测试")
    print(f"仓库     : {REPO_ROOT}")
    print(f"模型     : {meta_path.relative_to(REPO_ROOT)}")

    meta = load_meta(meta_path)
    feats, y = build_features()
    n_samples = len(feats)
    print(f"样本     : {n_samples} 条, 特征 {feats.shape[1]}x{feats.shape[2]}")

    # numpy 参考
    ref_pred, ref_probs = quantized_forward_cnn(meta, feats)

    # int8 特征（与训练/导出完全相同的量化）
    from quantize import quantize_sym
    xq = quantize_sym(feats, meta["input"]["scale"]).astype(np.int8)
    flat = np.ascontiguousarray(xq).reshape(n_samples, -1)

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    features_bin = PROBE_DIR / "features.bin"
    flat.tofile(features_bin)
    probe_c = PROBE_DIR / "probe_model.c"
    probe_c.write_text(build_probe_source(), encoding="utf-8")
    exe = PROBE_DIR / "probe_model.exe"

    try:
        vcvars = find_vcvars64()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"MSVC     : {vcvars}")
    print(f"[gen] {probe_c} ({probe_c.stat().st_size} B), features.bin "
          f"({features_bin.stat().st_size} B)")

    stdout = compile_and_run(vcvars, probe_c, exe, features_bin)
    rows = [line.split() for line in stdout.strip().splitlines() if line.strip()]
    if len(rows) != n_samples:
        print(f"FAIL: 探针输出 {len(rows)} 行，期望 {n_samples}")
        return 1
    c_pred = np.array([int(r[1]) for r in rows], np.int64)
    c_probs = np.array([[float(v) for v in r[2:]] for r in rows], np.float64)

    agree = int((c_pred == ref_pred).sum())
    max_diff = float(np.abs(c_probs - ref_probs.astype(np.float64)).max())
    print("-" * 72)
    print(f"[argmax] C vs numpy 一致 {agree}/{n_samples} = "
          f"{100.0 * agree / n_samples:.4f}%")
    print(f"[probs ] 最大绝对误差 {max_diff:.3e}（阈值 1e-4）")

    # 量化模型在数据集上的精度（只作信息展示）
    acc_int8 = float((ref_pred == y).mean())
    print(f"[info  ] 量化模型在全部 {n_samples} 条样本上的精度 "
          f"{acc_int8 * 100:.2f}%（训练集，仅供参考）")

    total, mismatch = check_avgpool_equivalence()
    print(f"[avgpool] numpy 整数公式 vs C float32 mean+round-away: "
          f"{total - mismatch}/{total} 一致, {mismatch} 不一致")

    print("-" * 72)
    ok = (agree == n_samples) and (max_diff < 1e-4) and (mismatch == 0)
    if ok:
        print("PASS: 导出模型 C 推理与 numpy 参考逐样本一致")
        return 0
    print("FAIL: 导出模型与 numpy 参考不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
