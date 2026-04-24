"""
export_weights.py — 从 mnist_model.pth 导出量化权重
输出: model_weights.h

PyTorch 权重布局（与 Keras 不同）：
  Conv2d.weight : [C_out, C_in, kH, kW]  → 转为 [C_out, kH, kW, C_in]
  Linear.weight : [out, in]              → 直接使用，无需转置
  BatchNorm2d   : weight(γ), bias(β), running_mean, running_var
"""

import os
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH  = "mnist_model.pth"
OUTPUT_PATH = "model_weights.h"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到 {MODEL_PATH}，请先运行 train_mcu.py")

# ─────────────────────────────────────────────
# 重建模型结构（与 train_mcu.py 保持一致）
# ─────────────────────────────────────────────
class MCU_CNN(nn.Module):
    def __init__(self, drop_conv1=0.10, drop_conv2=0.15, drop_fc=0.40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv2),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = MCU_CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print(f"已加载模型: {MODEL_PATH}")

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def quantize(arr: np.ndarray):
    """per-tensor 对称量化，zero_point=0"""
    scale = float(np.max(np.abs(arr))) / 127.0 if np.max(np.abs(arr)) != 0 else 1.0
    q     = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale

def to_numpy(tensor):
    return tensor.detach().float().numpy()

def fmt_int8(name, shape_str, arr):
    """int8 多维数组，每行16个值"""
    flat  = arr.flatten()
    rows  = [flat[i:i+16] for i in range(0, len(flat), 16)]
    body  = "\n".join(
        "  " + ", ".join(f"{int(v):4d}" for v in row) + ","
        for row in rows
    ).rstrip(",")
    return f"const int8_t {name}{shape_str} = {{\n{body}\n}};\n"

def fmt_float(name, n, arr):
    """float 一维数组，每行8个值"""
    flat  = arr.flatten()
    rows  = [flat[i:i+8] for i in range(0, len(flat), 8)]
    body  = "\n".join(
        "  " + ", ".join(f"{float(v):13.8f}f" for v in row) + ","
        for row in rows
    ).rstrip(",")
    return f"const float {name}[{n}] = {{\n{body}\n}};\n"

# ─────────────────────────────────────────────
# 提取权重
# ─────────────────────────────────────────────
# features 层索引：[0]Conv1 [1]BN1 [2]ReLU [3]Drop2d [4]Pool
#                  [5]Conv2 [6]BN2 [7]ReLU [8]Drop2d [9]Pool
# classifier 层索引：[0]Flatten [1]Linear1 [2]ReLU [3]Dropout [4]Linear2

# ── Conv1 ──────────────────────────────────
# PyTorch: (C_out=8, C_in=1, kH=3, kW=3) → 目标: [8][3][3][1]
w1_pt = to_numpy(model.features[0].weight)               # (8,1,3,3)
w1    = np.transpose(w1_pt, (0, 2, 3, 1))               # (8,3,3,1)
w1q, s1 = quantize(w1)

# ── BN1 折叠 ───────────────────────────────
bn1       = model.features[1]
eps1      = bn1.eps
gamma1    = to_numpy(bn1.weight)
beta1     = to_numpy(bn1.bias)
mean1     = to_numpy(bn1.running_mean)
var1      = to_numpy(bn1.running_var)
bn1_scale = gamma1 / np.sqrt(var1 + eps1)
bn1_bias  = beta1 - mean1 * bn1_scale

# ── Conv2 ──────────────────────────────────
# PyTorch: (16,8,3,3) → 目标: [16][3][3][8]
w2_pt = to_numpy(model.features[5].weight)               # (16,8,3,3)
w2    = np.transpose(w2_pt, (0, 2, 3, 1))               # (16,3,3,8)
w2q, s2 = quantize(w2)

# ── BN2 折叠 ───────────────────────────────
bn2       = model.features[6]
eps2      = bn2.eps
gamma2    = to_numpy(bn2.weight)
beta2     = to_numpy(bn2.bias)
mean2     = to_numpy(bn2.running_mean)
var2      = to_numpy(bn2.running_var)
bn2_scale = gamma2 / np.sqrt(var2 + eps2)
bn2_bias  = beta2 - mean2 * bn2_scale

# ── Dense1 ─────────────────────────────────
# PyTorch Linear.weight: [out=64, in=784] → 目标: [64][784] ✓ 无需转置
d1w = to_numpy(model.classifier[1].weight)               # (64,784)
d1b = to_numpy(model.classifier[1].bias)                 # (64,)
d1q, sd1 = quantize(d1w)

# ── Dense2 ─────────────────────────────────
# PyTorch: [out=10, in=64] → 目标: [10][64] ✓
d2w = to_numpy(model.classifier[4].weight)               # (10,64)
d2b = to_numpy(model.classifier[4].bias)                 # (10,)
d2q, sd2 = quantize(d2w)

# 打印摘要
print(f"\n{'层':<12} {'形状':^18} {'最大值':>8} {'scale':>12} {'最大误差':>10}")
print("─" * 64)
for name, w, wq, sc in [
    ("CONV1_W",  w1,  w1q, s1),
    ("CONV2_W",  w2,  w2q, s2),
    ("DENSE1_W", d1w, d1q, sd1),
    ("DENSE2_W", d2w, d2q, sd2),
]:
    dq  = wq.astype(np.float32) * sc
    err = np.max(np.abs(dq - w))
    print(f"  {name:<10} {str(w.shape):^18} {np.max(np.abs(w)):>8.4f} {sc:>12.8f} {err:>10.6f}")

# ─────────────────────────────────────────────
# 生成 model_weights.h
# ─────────────────────────────────────────────
lines = []
lines.append(f"""\
/**
 * model_weights.h — 由 export_weights.py 自动生成
 *
 * 源模型   : {MODEL_PATH}  (PyTorch)
 * 量化方式 : per-tensor 对称量化，zero_point = 0
 *            scale = max(|W|) / 127
 *            W_q   = clip(round(W / scale), -128, 127)
 *
 * 权重尺寸：
 *   CONV1_W     [8][3][3][1]   int8  =    72 B
 *   CONV1_BN_*  [8]            float =    64 B
 *   CONV2_W     [16][3][3][8]  int8  =  1152 B
 *   CONV2_BN_*  [16]           float =   128 B
 *   DENSE1_W    [64][784]      int8  = 50176 B (~49 KB)
 *   DENSE1_BIAS [64]           float =   256 B
 *   DENSE2_W    [10][64]       int8  =   640 B
 *   DENSE2_BIAS [10]           float =    40 B
 */

#pragma once
#include <stdint.h>

""")

sep = "// " + "═" * 63 + "\n"

lines += [sep, "//  Block 1 — Conv2D(8, 3×3, same, no_bias)\n",
          "//  布局: [C_out=8][kH=3][kW=3][C_in=1]  共 72 个 int8\n", sep, "\n"]
lines.append(fmt_int8("CONV1_W", "[8][3][3][1]", w1q))
lines.append(f"const float  CONV1_W_SCALE = {s1:.8f}f;\n")
lines.append(f"const int8_t CONV1_W_ZP    = 0;\n\n")
lines.append("// BN1 折叠参数  bn_scale=γ/√(var+ε)  bn_bias=β-mean·bn_scale\n")
lines.append(fmt_float("CONV1_BN_SCALE", 8,  bn1_scale))
lines.append(fmt_float("CONV1_BN_BIAS",  8,  bn1_bias))

lines += ["\n", sep, "//  Block 2 — Conv2D(16, 3×3, same, no_bias)\n",
          "//  布局: [C_out=16][kH=3][kW=3][C_in=8]  共 1152 个 int8\n", sep, "\n"]
lines.append(fmt_int8("CONV2_W", "[16][3][3][8]", w2q))
lines.append(f"const float  CONV2_W_SCALE = {s2:.8f}f;\n")
lines.append(f"const int8_t CONV2_W_ZP    = 0;\n\n")
lines.append("// BN2 折叠参数\n")
lines.append(fmt_float("CONV2_BN_SCALE", 16, bn2_scale))
lines.append(fmt_float("CONV2_BN_BIAS",  16, bn2_bias))

lines += ["\n", sep, "//  Dense1 — 784→64，带 bias，后接 ReLU\n",
          "//  布局: [out=64][in=784]  共 50176 个 int8\n", sep, "\n"]
lines.append(fmt_int8("DENSE1_W", "[64][784]", d1q))
lines.append(f"const float  DENSE1_W_SCALE = {sd1:.8f}f;\n")
lines.append(f"const int8_t DENSE1_W_ZP    = 0;\n\n")
lines.append(fmt_float("DENSE1_BIAS", 64, d1b))

lines += ["\n", sep, "//  Dense2 — 64→10，带 bias，后接 Softmax\n",
          "//  布局: [out=10][in=64]  共 640 个 int8\n", sep, "\n"]
lines.append(fmt_int8("DENSE2_W", "[10][64]", d2q))
lines.append(f"const float  DENSE2_W_SCALE = {sd2:.8f}f;\n")
lines.append(f"const int8_t DENSE2_W_ZP    = 0;\n\n")
lines.append(fmt_float("DENSE2_BIAS", 10, d2b))

with open(OUTPUT_PATH, "w") as f:
    f.writelines(lines)

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\n已生成: {OUTPUT_PATH}  ({size_kb:.1f} KB)")
