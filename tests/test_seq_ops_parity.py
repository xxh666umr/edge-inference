"""序列模型算子（GRU / LayerNorm / attention / softmax）C 实现 vs numpy 对齐测试。

覆盖：
  - softmax_probs：int32 logits + float scale -> 概率
  - gru_cell_f32：单步 GRU（PyTorch 语义，gate 顺序 r,z,n）
  - layernorm_f32：LayerNorm（有偏方差）
  - attention_f32：单头 scaled dot-product attention（含行 softmax）

C 探针由本脚本生成并编译（MSVC），数据直接嵌入探针源码，结果用 %.9g 打印后解析。
容差：1e-5（float32 与 numpy float64 的舍入差）。

运行（仓库根目录）：python tests/test_seq_ops_parity.py
退出码：0 = PASS，1 = FAIL。
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CSRC = REPO_ROOT / "csrc"
PROBE_DIR = Path(os.environ.get(
    "SEQ_PROBE_DIR", os.path.join(tempfile.gettempdir(), "edge_inference_seq_probe")))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def find_vcvars64() -> Path:
    env = os.environ.get("VCVARS64")
    if env and Path(env).is_file():
        return Path(env)
    for root in filter(None, (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"))):
        for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
            p = Path(root) / "Microsoft Visual Studio" / "2022" / edition / \
                "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if p.is_file():
                return p
    for pat in (r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
                r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"):
        m = sorted(glob.glob(pat))
        if m:
            return Path(m[-1])
    raise RuntimeError("找不到 MSVC vcvars64.bat")


def c_array(name, arr, dtype="float"):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(f"{v:.9g}f" if dtype == "float" else str(int(v)) for v in flat)
    return f"static const {dtype} {name}[{len(flat)}] = {{{vals}}};"


def build_probe(cases):
    lines = [
        "#include <stdio.h>",
        "#include <stdint.h>",
        '#include "nn.h"',
        "",
    ]
    body = []
    for ci, c in enumerate(cases):
        kind = c["kind"]
        p = f"c{ci}"
        if kind == "softmax":
            lines.append(c_array(f"{p}_logits", c["logits"], "int32_t"))
            body.append(f'    {{ float probs[{c["n"]}]; '
                        f'softmax_probs({p}_logits, {c["n"]}, {c["scale"]:.9g}f, probs); '
                        f'printf("{ci}"); for (int i=0;i<{c["n"]};++i) printf(" %.9g", probs[i]); '
                        f'printf("\\n"); }}')
        elif kind == "gru":
            H, I = c["hidden"], c["input_dim"]
            lines += [c_array(f"{p}_wih", c["w_ih"]), c_array(f"{p}_whh", c["w_hh"]),
                      c_array(f"{p}_bih", c["b_ih"]), c_array(f"{p}_bhh", c["b_hh"]),
                      c_array(f"{p}_x", c["x"]), c_array(f"{p}_h", c["h"])]
            body.append(f'    {{ float hn[{H}]; '
                        f'gru_cell_f32({p}_wih, {p}_whh, {p}_bih, {p}_bhh, {p}_x, {p}_h, hn, {I}, {H}); '
                        f'printf("{ci}"); for (int i=0;i<{H};++i) printf(" %.9g", hn[i]); '
                        f'printf("\\n"); }}')
        elif kind == "ln":
            n = c["n"]
            lines += [c_array(f"{p}_x", c["x"]), c_array(f"{p}_g", c["gamma"]),
                      c_array(f"{p}_b", c["beta"])]
            body.append(f'    {{ float y[{n}]; '
                        f'layernorm_f32({p}_x, {p}_g, {p}_b, y, {n}, {c["eps"]:.9g}f); '
                        f'printf("{ci}"); for (int i=0;i<{n};++i) printf(" %.9g", y[i]); '
                        f'printf("\\n"); }}')
        elif kind == "attn":
            S, D = c["seq"], c["d"]
            lines += [c_array(f"{p}_q", c["q"]), c_array(f"{p}_k", c["k"]),
                      c_array(f"{p}_v", c["v"])]
            body.append(f'    {{ float sc[{S * S}], out[{S * D}]; '
                        f'attention_f32({p}_q, {p}_k, {p}_v, out, {S}, {D}, sc); '
                        f'printf("{ci}"); for (int i=0;i<{S * D};++i) printf(" %.9g", out[i]); '
                        f'printf("\\n"); }}')
    lines += ["int main(void) {"] + body + ["    return 0;", "}", ""]
    return "\n".join(lines)


def run_probe(cases):
    vcvars = find_vcvars64()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    src = PROBE_DIR / "probe_seq.c"
    src.write_text(build_probe(cases), encoding="ascii")
    exe = PROBE_DIR / "probe_seq.exe"
    bat = PROBE_DIR / "build_seq.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        f'cl /nologo /O2 /W4 /I "{CSRC}" "{src}" "{CSRC / "nn.c"}" /Fe:"{exe}"\r\n',
        encoding="ascii")
    r = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(PROBE_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"编译失败:\n{r.stdout}\n{r.stderr}")
    run = subprocess.run([str(exe)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(PROBE_DIR))
    if run.returncode != 0:
        raise RuntimeError(f"运行失败:\n{run.stdout}\n{run.stderr}")
    out = {}
    for line in run.stdout.strip().splitlines():
        parts = line.split()
        if parts:
            out[int(parts[0])] = np.array([float(v) for v in parts[1:]], np.float64)
    return out


def ref_softmax(logits, scale):
    z = np.asarray(logits, np.float64) * scale
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def ref_gru(w_ih, w_hh, b_ih, b_hh, x, h, I, H):
    w_ih = np.asarray(w_ih).reshape(3 * H, I)
    w_hh = np.asarray(w_hh).reshape(3 * H, H)
    b_ih = np.asarray(b_ih)
    b_hh = np.asarray(b_hh)
    x = np.asarray(x, np.float64)
    h = np.asarray(h, np.float64)
    r = 1 / (1 + np.exp(-(w_ih[:H] @ x + b_ih[:H] + w_hh[:H] @ h + b_hh[:H])))
    z = 1 / (1 + np.exp(-(w_ih[H:2 * H] @ x + b_ih[H:2 * H] + w_hh[H:2 * H] @ h + b_hh[H:2 * H])))
    n = np.tanh(w_ih[2 * H:] @ x + b_ih[2 * H:] + r * (w_hh[2 * H:] @ h + b_hh[2 * H:]))
    return (1 - z) * n + z * h


def ref_layernorm(x, gamma, beta, eps):
    x = np.asarray(x, np.float64)
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * np.asarray(gamma) + np.asarray(beta)


def ref_attn(q, k, v, S, D):
    q = np.asarray(q, np.float64).reshape(S, D)
    k = np.asarray(k, np.float64).reshape(S, D)
    v = np.asarray(v, np.float64).reshape(S, D)
    s = q @ k.T / np.sqrt(D)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return (p @ v).reshape(-1)


def main():
    rng = np.random.default_rng(20261030)
    cases, expected = [], []

    # softmax
    for _ in range(4):
        n = int(rng.integers(3, 12))
        logits = rng.integers(-500, 500, size=n).tolist()
        scale = float(rng.uniform(1e-3, 5e-3))
        cases.append({"kind": "softmax", "logits": logits, "n": n, "scale": scale})
        expected.append(ref_softmax(logits, scale))

    # GRU
    for _ in range(4):
        I = int(rng.integers(3, 9))
        H = int(rng.integers(3, 7))
        c = {"kind": "gru", "input_dim": I, "hidden": H,
             "w_ih": rng.normal(size=(3 * H, I)).tolist(),
             "w_hh": rng.normal(size=(3 * H, H)).tolist(),
             "b_ih": rng.normal(size=3 * H).tolist(),
             "b_hh": rng.normal(size=3 * H).tolist(),
             "x": rng.normal(size=I).tolist(),
             "h": rng.normal(size=H).tolist()}
        cases.append(c)
        expected.append(ref_gru(c["w_ih"], c["w_hh"], c["b_ih"], c["b_hh"],
                                c["x"], c["h"], I, H))

    # LayerNorm
    for _ in range(3):
        n = int(rng.integers(4, 12))
        c = {"kind": "ln", "n": n, "eps": 1e-5,
             "x": rng.normal(size=n).tolist(),
             "gamma": (1 + rng.normal(size=n) * 0.1).tolist(),
             "beta": (rng.normal(size=n) * 0.1).tolist()}
        cases.append(c)
        expected.append(ref_layernorm(c["x"], c["gamma"], c["beta"], c["eps"]))

    # attention
    for _ in range(3):
        S = int(rng.integers(3, 6))
        D = int(rng.integers(3, 8))
        c = {"kind": "attn", "seq": S, "d": D,
             "q": rng.normal(size=(S, D)).tolist(),
             "k": rng.normal(size=(S, D)).tolist(),
             "v": rng.normal(size=(S, D)).tolist()}
        cases.append(c)
        expected.append(ref_attn(c["q"], c["k"], c["v"], S, D))

    print("=" * 72)
    print("序列算子 C 实现 vs numpy 对齐测试")
    print(f"仓库: {REPO_ROOT}")
    actual = run_probe(cases)

    total, bad = 0, []
    names = ["softmax_probs", "gru_cell_f32", "layernorm_f32", "attention_f32"]
    for ci, c in enumerate(cases):
        exp = expected[ci]
        got = actual.get(ci)
        if got is None or got.shape != exp.shape:
            bad.append(f"case {ci} ({c['kind']}): 输出缺失/形状不符")
            total += len(exp)
            continue
        diff = float(np.abs(got - exp).max())
        total += len(exp)
        ok = diff < 1e-5
        print(f"[{names[0] if c['kind']=='softmax' else names[1] if c['kind']=='gru' else names[2] if c['kind']=='ln' else names[3]}] "
              f"case {ci}: max|diff|={diff:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"case {ci} ({c['kind']}): max|diff|={diff:.2e}")

    print("-" * 72)
    print(f"比对元素总数: {total}, 失败 {len(bad)}")
    if bad:
        for b in bad[:10]:
            print("  -", b)
        print("FAIL")
        return 1
    print("PASS: 序列算子 C 实现与 numpy 一致（容差 1e-5）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
