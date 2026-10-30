"""在不重新训练的前提下，用已有 models/modrec_cnn.pt + 数据重新量化/导出 C 数组。

用途：量化算法（如 float → 定点重量化）变更后，快速重建 models/model.{h,c}、
demo_sample.h、modrec_meta.json，并更新 modrec_metrics.json 里的 int8 训练集精度。

用法：
    python tools/reexport.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import (  # noqa: E402
    quantize_cnn, quantize_sym, quantized_forward_cnn, save_meta,
    write_demo_sample, write_model,
)
from train_modrec import TinyCNN, decimate, load_iq  # noqa: E402


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(root, "models")
    data_path = os.path.join(root, "data", "train-00000-of-00001.parquet")

    ckpt = torch.load(os.path.join(models, "modrec_cnn.pt"), map_location="cpu")
    arch = ckpt["arch"]
    model = TinyCNN(arch["classes"], arch["c1"], arch["c2"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    iq, y, _snr, classes = load_iq(data_path)
    feats = decimate(iq)
    q = quantize_cnn(model, feats)
    pred_q, probs_q = quantized_forward_cnn(q, feats)
    acc_int8 = float((pred_q == y).mean())

    # 与旧 meta 对比权重，确认只改了重量化算法、权重/scale 一致
    meta_path = os.path.join(models, "modrec_meta.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        for layer in ("conv1", "conv2", "fc"):
            for key in ("w", "b"):
                a = np.asarray(old[layer][key])
                b = np.asarray(q[layer][key])
                assert a.shape == b.shape and np.array_equal(a, b), f"{layer}.{key} 变了"
        print("[check] 量化权重与旧 meta 完全一致（只改重量化算术）")
    except FileNotFoundError:
        pass

    h_path, c_path = write_model(q, models)
    correct = np.where(pred_q == y)[0]
    demo_idx = int(correct[0]) if len(correct) else 0
    demo_x = quantize_sym(feats[demo_idx:demo_idx + 1], q["input"]["scale"])[0]
    demo_path = write_demo_sample(
        os.path.join(models, "demo_sample.h"), demo_x,
        int(y[demo_idx]), int(pred_q[demo_idx]),
        true_name=classes[int(y[demo_idx])], pred_name=classes[int(pred_q[demo_idx])])
    save_meta(q, meta_path, extra={"classes": classes, "preprocess": {
        "raw_len": 1008, "bins": 126,
        "normalize": "per-window RMS (mean power = 1)",
        "layout": "[C=2][L=126], I then Q",
        "requant": "fixed-point mul/shift (int64)"}})

    metrics_path = os.path.join(models, "modrec_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics.setdefault("final", {})["train_acc_int8_fx"] = acc_int8
        metrics["final"]["requant"] = "fixed-point (mul=%d/%d, shift=%d)" % (
            q["conv1"]["fx_mul"], q["conv2"]["fx_mul"], q["conv1"]["fx_shift"])
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[export] {os.path.relpath(h_path, root)}")
    print(f"[export] {os.path.relpath(c_path, root)}")
    print(f"[export] {os.path.relpath(demo_path, root)}")
    print(f"[fx] conv1 mul={q['conv1']['fx_mul']} shift={q['conv1']['fx_shift']}  "
          f"conv2 mul={q['conv2']['fx_mul']} shift={q['conv2']['fx_shift']}")
    print(f"[int8] 全 550 条训练集精度 {acc_int8 * 100:.2f}%")
    print(f"[demo] true={y[demo_idx]} pred={pred_q[demo_idx]} "
          f"({classes[int(y[demo_idx])]})")


if __name__ == "__main__":
    main()
