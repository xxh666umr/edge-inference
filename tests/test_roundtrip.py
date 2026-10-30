"""验证 int8 量化后的整数前向与 PyTorch 浮点前向的分类结果一致。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from export_model import quantize_mlp, quantized_forward_numpy  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
model.eval()

calib = np.random.default_rng(0).normal(size=(512, 8)).astype(np.float32)
q = quantize_mlp(model, calib)

agree = 0
n = 1000
for _ in range(n):
    x = np.random.default_rng().normal(size=8).astype(np.float32)
    with torch.no_grad():
        t_arg = int(model(torch.from_numpy(x)).argmax().item())
    q_arg, _ = quantized_forward_numpy(q, x)
    if t_arg == q_arg:
        agree += 1

print(f"agreement: {agree}/{n} = {100.0 * agree / n:.2f}%")
print(f"input_scale={q['input_scale']:.6f} "
      f"W1={q['layer1']['w'].size} int8  W2={q['layer2']['w'].size} int8")
print(f"h_scale={q['layer1']['out_scale']:.6f}  "
      f"req1={q['layer1']['req_scale']:.6f}  acc2_scale={q['layer2']['acc_scale']:.6f}")
if agree / n >= 0.9:
    print("PASS: int8 前向与 PyTorch 分类基本一致")
else:
    print("WARN: 一致率偏低，检查量化 scale")
