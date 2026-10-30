"""Phase 2：训练 tiny CNN 调制识别模型（RadioML 11 类），int8 量化并导出 C 数组。

流程：
  1. 读 data/train-00000-of-00001.parquet（550 条 I/Q，11 类，SNR -10~30 dB）；
  2. 预处理：截到 1008 点 -> 每样本功率归一化 -> 平均抽取到 126 点（2×126 输入）；
  3. 数据增强：随机相位旋转、循环时移（8 的倍数）、幅度缩放、加高斯噪声（SNR -10~25 dB）；
  4. 5 折交叉验证（按 类内 SNR 排序轮转划分），评估浮点精度；
  5. 每折模型用训练折校准集做 int8 量化，评估量化后精度（OOF），统计掉点；
  6. 用全部 550 条训练最终模型 -> int8 量化 -> 生成 models/model.h / model.c / modrec_meta.json。

用法（Windows PowerShell）：
  $py = "python"
  $env:PYTHONPATH = "<你的依赖目录>"
  & $py tools\\train_modrec.py --data data\\train-00000-of-00001.parquet --out-dir models
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import (  # noqa: E402
    quantize_cnn, quantize_sym, quantized_forward_cnn, save_meta, write_demo_sample,
    write_model,
)

# 数据固定参数（数据集内 400 条 1020 点、150 条 1024 点，统一截到 1008）
RAW_LEN = 1008
BINS = 126           # 平均抽取后的每通道长度（1008 / 8）
C1, C2 = 16, 32      # 两个卷积层通道数


class TinyCNN(nn.Module):
    """2 x 126 -> conv k7 s2 -> pool2 -> conv k5 s2 -> pool2 -> 全局平均 -> fc。"""

    def __init__(self, n_cls=11, c1=C1, c2=C2):
        super().__init__()
        self.c1 = nn.Conv1d(2, c1, 7, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        self.c2 = nn.Conv1d(c1, c2, 5, stride=2)
        self.pool2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(c2, n_cls)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.pool1(x)
        x = torch.relu(self.c2(x))
        x = self.pool2(x)
        x = x.mean(dim=2)
        return self.fc(x)


class TinyMLP(nn.Module):
    """基线：126 维池化 I/Q 特征 -> 64 -> 11。参数更少，用于对比。"""

    def __init__(self, n_cls=11, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * BINS, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 数据
def load_iq(path):
    df = pd.read_parquet(path)
    iq = np.stack([
        np.stack([np.asarray(i, np.float32)[:RAW_LEN], np.asarray(q, np.float32)[:RAW_LEN]])
        for i, q in zip(df["iq_i"], df["iq_q"])
    ])
    y = df["label"].to_numpy(np.int64)
    snr = df["snr_db"].to_numpy(np.int64)
    classes = []
    for lab in range(int(y.max()) + 1):
        names = df.loc[df["label"] == lab, "modulation"].unique()
        classes.append(str(names[0]))
    # 每样本功率归一化（I/Q 平均功率为 1）
    p = (iq ** 2).mean(axis=(1, 2), keepdims=True)
    iq = iq / np.sqrt(p)
    return iq, y, snr, classes


def decimate(iq, bins=BINS):
    n, c, length = iq.shape
    f = length // bins
    return iq.reshape(n, c, bins, f).mean(axis=3).astype(np.float32)


def make_folds(y, snr, k=5):
    fold = np.zeros(len(y), np.int64)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        order = idx[np.argsort(snr[idx])]
        for j, i in enumerate(order):
            fold[i] = j % k
    return fold


def augment_iq(iq, rng):
    b = iq.shape[0]
    ph = rng.uniform(0, 2 * np.pi, size=(b, 1)).astype(np.float32)
    cos, sin = np.cos(ph), np.sin(ph)
    i2 = iq[:, 0] * cos - iq[:, 1] * sin
    q2 = iq[:, 0] * sin + iq[:, 1] * cos
    iq = np.stack([i2, q2], axis=1)
    shift = rng.integers(0, BINS, size=b) * (RAW_LEN // BINS)
    iq = np.stack([
        np.stack([np.roll(iq[j, 0], shift[j]), np.roll(iq[j, 1], shift[j])]) for j in range(b)
    ])
    iq = iq * rng.uniform(0.7, 1.3, size=(b, 1, 1)).astype(np.float32)
    target = rng.uniform(-10, 25, size=(b, 1, 1)).astype(np.float32)
    sig = (iq ** 2).mean(axis=(1, 2), keepdims=True)
    noise_p = sig / (10 ** (target / 10))
    noise = rng.normal(size=iq.shape).astype(np.float32) * np.sqrt(noise_p / 2)
    return iq + noise


# ---------------------------------------------------------------- 训练 / 评估
def train_model(model, x_iq, y, epochs, lr, augment, seed, batch=64, swa_last=10):
    """固定 epoch 训练 + 最后 swa_last 个 epoch 的权重平均（无验证集泄漏）。

    交叉验证时不能拿"在验证折上挑最好 epoch"来选模型，否则精度会被高估；
    这里用 cosine 退火 + 末段权重平均（SWA-lite）稳定最终权重。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    yt = torch.from_numpy(np.asarray(y).copy())
    n = len(yt)
    swa = None
    n_swa = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch].numpy()
            xb = x_iq[idx].copy()
            if augment:
                xb = augment_iq(xb, rng)
            xb = torch.from_numpy(decimate(xb))
            opt.zero_grad()
            loss = lossf(model(xb), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if ep >= epochs - swa_last:
            sd = model.state_dict()
            if swa is None:
                swa = {k: v.detach().clone().float() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    swa[k] += v.detach().float()
            n_swa += 1
    if swa is not None:
        for k in swa:
            swa[k] /= n_swa
        model.load_state_dict({k: v.clone() for k, v in swa.items()})
    return model


@torch.no_grad()
def predict_float(model, x_iq):
    model.eval()
    x = torch.from_numpy(decimate(x_iq))
    return model(x).argmax(1).numpy()


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "train-00000-of-00001.parquet"))
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--quick", action="store_true", help="跳过交叉验证，只训练导出模型")
    ap.add_argument("--no-mlp-baseline", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(root, args.data)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)

    iq, y, snr, classes = load_iq(data_path)
    fold = make_folds(y, snr, args.folds)
    print(f"[data] {iq.shape[0]} samples, {len(classes)} classes, folds={np.bincount(fold).tolist()}")
    print(f"[data] classes: {classes}")

    metrics = {
        "dataset": {"path": os.path.basename(data_path), "samples": int(len(y)),
                    "classes": classes, "snr_range": [int(snr.min()), int(snr.max())],
                    "raw_len": RAW_LEN, "bins": BINS},
        "model": {"arch": "tiny_cnn", "c1": C1, "c2": C2, "input": [2, BINS]},
    }

    if not args.quick:
        oof_float = np.full(len(y), -1, np.int64)
        oof_int8 = np.full(len(y), -1, np.int64)
        fold_f_float, fold_f_int8 = [], []
        for f in range(args.folds):
            tr, va = fold != f, fold == f
            model = TinyCNN(len(classes), C1, C2)
            train_model(model, iq[tr], y[tr], args.epochs, args.lr,
                        not args.no_augment, args.seed + f)
            pred_f = predict_float(model, iq[va])
            q = quantize_cnn(model, decimate(iq[tr]))
            pred_q, _ = quantized_forward_cnn(q, decimate(iq[va]))
            oof_float[va] = pred_f
            oof_int8[va] = pred_q
            acc_f = float((pred_f == y[va]).mean())
            acc_q = float((pred_q == y[va]).mean())
            fold_f_float.append(acc_f)
            fold_f_int8.append(acc_q)
            print(f"[cv] fold {f}: float {acc_f*100:.2f}%  int8 {acc_q*100:.2f}%")

        acc_f = float((oof_float == y).mean())
        acc_q = float((oof_int8 == y).mean())
        per_snr = {}
        for s in sorted(np.unique(snr)):
            m = snr == s
            per_snr[str(int(s))] = {"n": int(m.sum()),
                                    "float": float((oof_float[m] == y[m]).mean()),
                                    "int8": float((oof_int8[m] == y[m]).mean())}
        cm = np.zeros((len(classes), len(classes)), np.int64)
        for t, p in zip(y, oof_int8):
            cm[t, p] += 1
        per_class = {classes[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(classes))}
        metrics["cv"] = {
            "folds": args.folds, "epochs": args.epochs, "augment": not args.no_augment,
            "oa_float": acc_f, "oa_int8": acc_q, "quant_drop": acc_f - acc_q,
            "per_fold_float": fold_f_float, "per_fold_int8": fold_f_int8,
            "per_snr_int8": per_snr, "per_class_int8": per_class,
            "confusion_matrix": cm.tolist(),
        }
        print(f"[cv] OOF float {acc_f*100:.2f}%  int8 {acc_q*100:.2f}%  "
              f"(drop {100*(acc_f-acc_q):.2f} pt)")

        if not args.no_mlp_baseline:
            mlp_pred = np.zeros(len(y), np.int64)
            for f in range(args.folds):
                tr, va = fold != f, fold == f
                mlp = TinyMLP(len(classes))
                torch.manual_seed(args.seed + f)
                rng = np.random.default_rng(args.seed + f)
                opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
                lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
                xtr = decimate(iq[tr]).reshape(tr.sum(), -1)
                ytr = torch.from_numpy(y[tr])
                for _ in range(args.epochs):
                    mlp.train()
                    perm = torch.randperm(len(ytr))
                    for i in range(0, len(ytr), 64):
                        b = perm[i:i + 64].numpy()
                        xb = xtr[b]
                        if not args.no_augment:
                            raw = augment_iq(iq[tr][b].copy(), rng)
                            xb = decimate(raw).reshape(len(b), -1)
                        opt.zero_grad()
                        loss = lossf(mlp(torch.from_numpy(xb)), ytr[b])
                        loss.backward()
                        opt.step()
                    sched.step()
                with torch.no_grad():
                    xv = torch.from_numpy(decimate(iq[va]).reshape(va.sum(), -1))
                    mlp_pred[va] = mlp(xv).argmax(1).numpy()
            acc_mlp = float((mlp_pred == y).mean())
            metrics["mlp_baseline"] = {"oa_float": acc_mlp, "params": int(
                sum(p.numel() for p in TinyMLP(len(classes)).parameters()))}
            print(f"[cv] MLP baseline float {acc_mlp*100:.2f}%")

    # ---- 最终模型：全部数据训练 -> 量化 -> 导出
    model = TinyCNN(len(classes), C1, C2)
    train_model(model, iq, y, args.epochs, args.lr, not args.no_augment, args.seed)
    feats_all = decimate(iq)
    q = quantize_cnn(model, feats_all)
    pred_q, _ = quantized_forward_cnn(q, feats_all)
    train_acc_q = float((pred_q == y).mean())
    train_acc_f = float((predict_float(model, iq) == y).mean())
    metrics["final"] = {
        "trained_on": "all 550", "epochs": args.epochs,
        "train_acc_float": train_acc_f, "train_acc_int8": train_acc_q,
        "param_count": q["param_count"], "weight_bytes": q["weight_bytes"],
        "bias_bytes": q["bias_bytes"],
        "flash_estimate_bytes": q["weight_bytes"] + q["bias_bytes"] + 4096,
    }
    os.makedirs(out_dir, exist_ok=True)
    h_path, c_path = write_model(q, out_dir)
    # 选一条 int8 预测正确的样本作为 C demo / 固件的内置输入
    correct = np.where(pred_q == y)[0]
    demo_idx = int(correct[0]) if len(correct) else 0
    demo_x = quantize_sym(feats_all[demo_idx:demo_idx + 1], q["input"]["scale"])[0]
    demo_path = write_demo_sample(
        os.path.join(out_dir, "demo_sample.h"), demo_x,
        int(y[demo_idx]), int(pred_q[demo_idx]),
        true_name=classes[int(y[demo_idx])], pred_name=classes[int(pred_q[demo_idx])])
    save_meta(q, os.path.join(out_dir, "modrec_meta.json"),
              extra={"classes": classes, "preprocess": {
                  "raw_len": RAW_LEN, "bins": BINS,
                  "normalize": "per-window RMS (mean power = 1)",
                  "layout": "[C=2][L=126], I then Q"}})
    torch.save({"state_dict": model.state_dict(),
                "arch": {"c1": C1, "c2": C2, "classes": len(classes)}},
               os.path.join(out_dir, "modrec_cnn.pt"))
    with open(os.path.join(out_dir, "modrec_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[export] {h_path}")
    print(f"[export] {c_path}")
    print(f"[export] {demo_path}")
    print(f"[model] params={q['param_count']} int8_weights={q['weight_bytes']}B "
          f"bias={q['bias_bytes']}B")
    print(f"[train-all] float {train_acc_f*100:.2f}%  int8 {train_acc_q*100:.2f}%")


if __name__ == "__main__":
    main()




