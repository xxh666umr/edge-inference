"""用完整 RadioML 2016.10a（22 万条）重训调制识别 tiny CNN，并导出 int8 C 数组。

数据：RML2016.10a（Zenodo 18397070，CC BY-NC-SA 4.0，学术用途；数据不入库）
  - 11 类：8PSK / AM-DSB / AM-SSB / BPSK / CPFSK / GFSK / PAM4 / QAM16 / QAM64 / QPSK / WBFM
  - 20 个 SNR：-20 ~ +18 dB（2 dB 步进），每个 (mod, snr) 1000 条
  - 每条 (2, 128) float32；总 220,000 条；按 (mod, snr) 分层 80/10/10

模型与部署版一致：2×128 → conv(k7,s2,16) → pool2 → conv(k5,s2,32) → pool2 → GAP → FC11
（3,195 参数）；导出 models/model.{h,c}、demo_sample.h、meta、checkpoint、parity_features.npz。

训练加速：数据与增强都放在 GPU 上（相位旋转/循环时移/幅度缩放/加噪全部向量化），
避免 CPU numpy 逐样本循环成为瓶颈。

用法：
    python tools/train_radio2016.py [--epochs 100] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import (  # noqa: E402
    quantize_cnn, quantize_sym, quantized_forward_cnn, save_meta,
    write_demo_sample, write_model,
)
from train_modrec import TinyCNN  # noqa: E402

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def load_rml2016(path):
    """读取 RML2016.10a pickle，返回 X (N,2,128) float32、y (N,)、snr (N,)。"""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    xs, ys, snrs = [], [], []
    for key, val in d.items():
        mod, snr = key
        if isinstance(mod, bytes):
            mod = mod.decode()
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (2, 128):
            raise ValueError(f"unexpected shape for {key}: {arr.shape}")
        xs.append(arr)
        ys.append(np.full(len(arr), CLASSES.index(mod), np.int64))
        snrs.append(np.full(len(arr), int(snr), np.int64))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys)
    snr = np.concatenate(snrs)
    p = (X ** 2).mean(axis=(1, 2), keepdims=True)
    X = X / np.sqrt(np.maximum(p, 1e-12))
    return X, y, snr


def stratified_split(y, snr, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    part = np.zeros(n, np.int8)
    for cls in np.unique(y):
        for s in np.unique(snr):
            idx = np.where((y == cls) & (snr == s))[0]
            rng.shuffle(idx)
            k = len(idx)
            n_tr = int(round(k * 0.8))
            n_va = int(round(k * 0.1))
            part[idx[:n_tr]] = 0
            part[idx[n_tr:n_tr + n_va]] = 1
            part[idx[n_tr + n_va:]] = 2
    return part


def augment_torch(x, gen, noise_lo=-20.0, noise_hi=10.0, use_noise=True):
    """GPU 上向量化增强：相位旋转 + 循环时移 + 幅度缩放 + 可选加噪。"""
    b, _, length = x.shape
    ph = torch.rand(b, 1, device=x.device, generator=gen) * (2 * math.pi)
    c, s = torch.cos(ph), torch.sin(ph)
    i2 = x[:, 0] * c - x[:, 1] * s
    q2 = x[:, 0] * s + x[:, 1] * c
    x = torch.stack([i2, q2], dim=1)
    sh = torch.randint(0, length, (b,), device=x.device, generator=gen)
    idx = (torch.arange(length, device=x.device)[None, :] + sh[:, None]) % length
    x = torch.gather(x, 2, idx[:, None, :].expand(-1, 2, -1))
    gain = 0.7 + torch.rand(b, 1, 1, device=x.device, generator=gen) * 0.6
    x = x * gain
    if not use_noise:
        return x
    target = noise_lo + torch.rand(b, 1, 1, device=x.device, generator=gen) * (noise_hi - noise_lo)
    sig = (x ** 2).mean(dim=(1, 2), keepdim=True)
    noise_p = sig / torch.pow(torch.tensor(10.0, device=x.device), target / 10.0)
    noise = torch.randn(x.shape, device=x.device, generator=gen) * torch.sqrt(noise_p / 2)
    return x + noise


@torch.no_grad()
def predict(model, X, batch=1024):
    model.eval()
    pred = np.zeros(len(X), np.int64)
    for i in range(0, len(X), batch):
        pred[i:i + batch] = model(X[i:i + batch]).argmax(1).cpu().numpy()
    return pred


def metrics(pred, y, snr):
    per_snr = {}
    for s in sorted(np.unique(snr)):
        m = snr == s
        per_snr[str(int(s))] = {"n": int(m.sum()), "acc": float((pred[m] == y[m]).mean())}
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per_class = {CLASSES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(CLASSES))}
    return {"oa": float((pred == y).mean()), "per_snr": per_snr,
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(root, "data", "RML2016.10a_dict_optimized.pkl"))
    ap.add_argument("--out-dir", default=os.path.join(root, "models"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c1", type=int, default=16)
    ap.add_argument("--c2", type=int, default=32)
    ap.add_argument("--no-noise-aug", action="store_true", help="关闭额外加噪增强")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    print(f"[data] {args.data}")
    X, y, snr = load_rml2016(args.data)
    print(f"[data] {X.shape}, {len(CLASSES)} classes, {time.time()-t0:.1f}s")
    part = stratified_split(y, snr, args.seed)
    tr, va, te = part == 0, part == 1, part == 2
    print(f"[split] train {tr.sum()} val {va.sum()} test {te.sum()}")

    Xtr = torch.from_numpy(X[tr]).to(dev)
    ytr = torch.from_numpy(y[tr]).to(dev)
    Xva = torch.from_numpy(X[va]).to(dev)
    Xte = torch.from_numpy(X[te]).to(dev)

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    model = TinyCNN(len(CLASSES), args.c1, args.c2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(Xtr)
    swa, n_swa = None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = augment_torch(Xtr[idx], gen, use_noise=not args.no_noise_aug)
            yb = ytr[idx]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep >= args.epochs - 5:
            sd = model.state_dict()
            if swa is None:
                swa = {k: v.detach().clone().float() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    swa[k] += v.detach().float()
            n_swa += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            va_pred = predict(model, Xva)
            print(f"[ep {ep:3d}] loss {tot/n:.4f}  val_acc "
                  f"{(va_pred == y[va]).mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    if swa is not None:
        for k in swa:
            swa[k] /= n_swa
        model.load_state_dict({k: v.clone() for k, v in swa.items()})
    model.eval()

    pred_f = predict(model, Xte)
    m_float = metrics(pred_f, y[te], snr[te])
    calib = X[tr][:2000]
    q = quantize_cnn(model, calib)
    pred_q, _ = quantized_forward_cnn(q, X[te])
    m_int8 = metrics(pred_q, y[te], snr[te])
    m10 = snr[te] >= -10
    acc10_f = float((pred_f[m10] == y[te][m10]).mean())
    acc10_q = float((pred_q[m10] == y[te][m10]).mean())
    print(f"[test] float {m_float['oa']*100:.2f}%  int8 {m_int8['oa']*100:.2f}%  "
          f"(drop {(m_float['oa']-m_int8['oa'])*100:.2f} pt)")
    print(f"[test] SNR>=-10 ({int(m10.sum())} 条): float {acc10_f*100:.2f}%  "
          f"int8 {acc10_q*100:.2f}%")
    print("[test] per-SNR(float): " + ", ".join(
        f"{k}:{v['acc']*100:.0f}%" for k, v in m_float["per_snr"].items()))

    h_path, c_path = write_model(q, args.out_dir, name="radio_ml_cnn_2016")
    correct = np.where(pred_q == y[te])[0]
    di = int(correct[0]) if len(correct) else 0
    demo_x = quantize_sym(X[te][di:di + 1], q["input"]["scale"])[0]
    demo_path = write_demo_sample(
        os.path.join(args.out_dir, "demo_sample.h"), demo_x,
        int(y[te][di]), int(pred_q[di]),
        true_name=CLASSES[int(y[te][di])], pred_name=CLASSES[int(pred_q[di])])
    save_meta(q, os.path.join(args.out_dir, "modrec_meta.json"), extra={
        "classes": CLASSES,
        "preprocess": {"raw_len": 128, "bins": 128,
                       "normalize": "per-window RMS (mean power = 1)",
                       "layout": "[C=2][L=128], I then Q",
                       "requant": "fixed-point mul/shift (int64)"},
        "dataset": "RadioML 2016.10a (Zenodo 18397070, CC BY-NC-SA 4.0)",
    })
    np.savez_compressed(os.path.join(args.out_dir, "parity_features.npz"),
                        features=X[te][:300].astype(np.float32),
                        labels=y[te][:300], snr=snr[te][:300])
    torch.save({"state_dict": model.state_dict(),
                "arch": {"c1": args.c1, "c2": args.c2, "classes": len(CLASSES), "input_len": 128}},
               os.path.join(args.out_dir, "modrec_cnn.pt"))
    with open(os.path.join(args.out_dir, "modrec_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": {"name": "RadioML 2016.10a", "samples": int(len(y)),
                               "classes": CLASSES, "snr_range": [-20, 18],
                               "split": {"train": int(tr.sum()), "val": int(va.sum()),
                                         "test": int(te.sum())}},
                   "model": {"arch": "tiny_cnn", "c1": args.c1, "c2": args.c2,
                             "input": [2, 128], "params": q["param_count"],
                             "int8_weight_bytes": q["weight_bytes"],
                             "bias_bytes": q["bias_bytes"]},
                   "test_float": m_float, "test_int8": m_int8,
                   "test_snr_ge_m10": {"n": int(m10.sum()),
                                       "float": acc10_f, "int8": acc10_q},
                   "quant_drop": m_float["oa"] - m_int8["oa"],
                   "fx": {"conv1_mul": q["conv1"]["fx_mul"],
                          "conv2_mul": q["conv2"]["fx_mul"], "shift": 24}},
                  f, indent=2, ensure_ascii=False)
    print(f"[export] {h_path}")
    print(f"[export] {c_path}")
    print(f"[export] {demo_path}")
    print(f"[model] params={q['param_count']} int8={q['weight_bytes']}B")
    print(f"[time] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
