"""
compare_models.py — float32 vs int8 量化模型在测试集上的对比（PyTorch）
========================================================================
原理：
  量化模型 = 将各层权重量化为 int8 再反量化回 float32
  → 注入同一 MCU_CNN 架构中运行
  → 与原始 float32 权重模型在完全相同的测试数据上对比

输出图：
  compare_models.png  (2×3 共6个子图)
    1. 总体准确率对比
    2. 各类别准确率对比
    3. 置信度分布对比
    4. 两模型预测一致性分析
    5. 量化误差 vs 置信度变化散点图
    6. 分歧样本展示
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 0. 中文字体 & 设备
# ─────────────────────────────────────────────
_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
zh_font    = fm.FontProperties(fname=_FONT_PATH)
matplotlib.rcParams['font.family']        = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'mnist_model.pth'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ─────────────────────────────────────────────
# 1. 模型定义（与 train_mcu.py 保持一致）
# ─────────────────────────────────────────────
class MCU_CNN(nn.Module):
    def __init__(self, drop_conv1=0.10, drop_conv2=0.15, drop_fc=0.40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv1), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv2), nn.MaxPool2d(2),
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

# ─────────────────────────────────────────────
# 2. 加载数据集（测试集，不做增强）
# ─────────────────────────────────────────────
_MEAN, _STD = (0.1307,), (0.3081,)
test_transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
test_ds     = torchvision.datasets.MNIST('./data', train=False,
                                         download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
print(f"测试集: {len(test_ds):,} 张")

# ─────────────────────────────────────────────
# 3. 加载 float32 模型
# ─────────────────────────────────────────────
fp32_model = MCU_CNN()
fp32_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
fp32_model.eval()
print(f"已加载 float32 模型: {MODEL_PATH}")

# ─────────────────────────────────────────────
# 4. 量化工具函数（per-tensor 对称，zero_point=0）
# ─────────────────────────────────────────────
def quantize(arr: np.ndarray):
    """int8 对称量化"""
    scale = float(np.max(np.abs(arr))) / 127.0 if np.max(np.abs(arr)) != 0 else 1.0
    q     = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale

def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale

def to_np(tensor):
    return tensor.detach().float().numpy()

# ─────────────────────────────────────────────
# 5. 构建量化模型（权重量化→反量化后注入）
# ─────────────────────────────────────────────
print("\n量化权重（per-tensor 对称 int8）...")
int8_model = copy.deepcopy(fp32_model)

def quant_conv_weight(module_fp32, module_int8):
    """Conv2d 权重: (C_out,C_in,kH,kW) → [C_out,kH,kW,C_in] 量化 → 还原"""
    w    = to_np(module_fp32.weight)              # (C_out,C_in,kH,kW)
    w_t  = np.transpose(w, (0, 2, 3, 1))         # [C_out,kH,kW,C_in]
    wq, s = quantize(w_t)
    w_dq = np.transpose(dequantize(wq, s), (0, 3, 1, 2))   # 还原 PyTorch 格式
    module_int8.weight.data = torch.from_numpy(w_dq)
    err = float(np.max(np.abs(w_dq - w)))
    return s, err

def quant_linear_weight(module_fp32, module_int8):
    """Linear 权重: (out,in) 直接量化"""
    w    = to_np(module_fp32.weight)              # (out,in)
    wq, s = quantize(w)
    w_dq = dequantize(wq, s)
    module_int8.weight.data = torch.from_numpy(w_dq)
    err = float(np.max(np.abs(w_dq - w)))
    return s, err

s1,  e1  = quant_conv_weight(fp32_model.features[0],    int8_model.features[0])
s2,  e2  = quant_conv_weight(fp32_model.features[5],    int8_model.features[5])
sd1, ed1 = quant_linear_weight(fp32_model.classifier[1], int8_model.classifier[1])
sd2, ed2 = quant_linear_weight(fp32_model.classifier[4], int8_model.classifier[4])

print(f"  {'层':<12} {'scale':>14} {'最大权重误差':>14}")
print(f"  {'─'*42}")
for name, s, e in [('CONV1_W', s1, e1), ('CONV2_W', s2, e2),
                    ('DENSE1_W', sd1, ed1), ('DENSE2_W', sd2, ed2)]:
    print(f"  {name:<12} {s:>14.8f} {e:>14.8f}")

int8_model.eval()

# ─────────────────────────────────────────────
# 6. 推理（获取 softmax 概率）
# ─────────────────────────────────────────────
print("\n推理中（float32 & int8 模型各一次）...")

def infer(model, loader):
    all_probs, all_preds, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.numpy())
            all_preds.append(logits.argmax(1).numpy())
            all_labels.append(labels.numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))

fp32_prob, fp32_pred, y_test = infer(fp32_model, test_loader)
int8_prob, int8_pred, _      = infer(int8_model,  test_loader)

fp32_acc = np.mean(fp32_pred == y_test) * 100
int8_acc = np.mean(int8_pred == y_test) * 100
acc_loss = fp32_acc - int8_acc

print(f"\nfloat32 准确率: {fp32_acc:.2f}%")
print(f"int8    准确率: {int8_acc:.2f}%")
print(f"准确率损失:     {acc_loss:.3f}%")

# 预测一致性统计
mask_bc = (fp32_pred == y_test) & (int8_pred == y_test)
mask_fo = (fp32_pred == y_test) & (int8_pred != y_test)
mask_io = (fp32_pred != y_test) & (int8_pred == y_test)
mask_bw = (fp32_pred != y_test) & (int8_pred != y_test)
both_correct = mask_bc.sum()
fp32_only    = mask_fo.sum()
int8_only    = mask_io.sum()
both_wrong   = mask_bw.sum()

print(f"\n两者都对   : {both_correct} 张 ({both_correct/100:.1f}%)")
print(f"仅 float32 对: {fp32_only} 张")
print(f"仅 int8 对   : {int8_only} 张")
print(f"两者都错   : {both_wrong} 张")

# ─────────────────────────────────────────────
# 7. 绘图（2×3，共6个子图）
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle("float32 模型 vs int8 量化模型 — 测试集对比（10,000张）",
             fontsize=15, fontweight="bold", fontproperties=zh_font, y=0.98)

gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
COLORS = {"fp32": "#4C8BF5", "int8": "#F5824C"}

# ── 子图1：总体准确率 ──────────────────────────
ax1   = fig.add_subplot(gs[0, 0])
bars  = ax1.bar(["float32", "int8 量化"],
                [fp32_acc, int8_acc],
                color=[COLORS["fp32"], COLORS["int8"]],
                width=0.45, edgecolor="white", linewidth=1.2)
y_lo  = max(96, min(fp32_acc, int8_acc) - 1.5)
ax1.set_ylim(y_lo, 100.8)
ax1.set_ylabel("准确率 (%)", fontproperties=zh_font)
ax1.set_title("总体准确率对比", fontproperties=zh_font, fontsize=12)
ax1.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, [fp32_acc, int8_acc]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
             f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
if abs(acc_loss) < 0.05:
    ax1.text(0.5, (fp32_acc + int8_acc) / 2 - 1.0,
             f"差异 {abs(acc_loss):.3f}%（近似无损）",
             ha="center", fontproperties=zh_font, color="green", fontsize=9,
             transform=ax1.transData)
else:
    ax1.annotate(f"损失 {acc_loss:.3f}%",
                 xy=(1, int8_acc), xytext=(0.4, int8_acc - 0.6),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontproperties=zh_font, color="red", fontsize=9)

# ── 子图2：各类别准确率 ────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
class_acc_fp32 = [np.mean(fp32_pred[y_test == d] == d) * 100 for d in range(10)]
class_acc_int8 = [np.mean(int8_pred[y_test == d] == d) * 100 for d in range(10)]
xp  = np.arange(10)
bw  = 0.38
ax2.bar(xp - bw/2, class_acc_fp32, bw, label="float32",
        color=COLORS["fp32"], edgecolor="white")
ax2.bar(xp + bw/2, class_acc_int8, bw, label="int8",
        color=COLORS["int8"], edgecolor="white")
bot = max(90, min(class_acc_fp32 + class_acc_int8) - 2)
ax2.set_ylim(bot, 101.5)
ax2.set_xticks(xp)
ax2.set_xlabel("数字类别", fontproperties=zh_font)
ax2.set_ylabel("准确率 (%)", fontproperties=zh_font)
ax2.set_title("各类别准确率对比", fontproperties=zh_font, fontsize=12)
ax2.legend(prop=zh_font, loc="lower right")
ax2.grid(True, alpha=0.3, axis="y")
diffs_cls = [abs(a - b) for a, b in zip(class_acc_fp32, class_acc_int8)]
worst     = int(np.argmax(diffs_cls))
if diffs_cls[worst] > 0.2:
    ax2.annotate(f"最大差异\n数字 {worst}: {diffs_cls[worst]:.2f}%",
                 xy=(worst, min(class_acc_fp32[worst], class_acc_int8[worst])),
                 xytext=(worst + 1.2, bot + 1.5),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontproperties=zh_font, color="red", fontsize=8)

# ── 子图3：置信度分布直方图 ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
conf_fp32_ok = np.max(fp32_prob[fp32_pred == y_test], axis=1)
conf_int8_ok = np.max(int8_prob[int8_pred == y_test], axis=1)
bins = np.linspace(0.5, 1.0, 40)
ax3.hist(conf_fp32_ok, bins=bins, alpha=0.6, color=COLORS["fp32"],
         label="float32 正确预测", density=True)
ax3.hist(conf_int8_ok, bins=bins, alpha=0.6, color=COLORS["int8"],
         label="int8 正确预测",   density=True)
ax3.set_xlabel("最高类别置信度", fontproperties=zh_font)
ax3.set_ylabel("概率密度",       fontproperties=zh_font)
ax3.set_title("正确预测的置信度分布", fontproperties=zh_font, fontsize=12)
ax3.legend(prop=zh_font, fontsize=9)
ax3.grid(True, alpha=0.3)
ylim3 = ax3.get_ylim()[1]
ax3.text(0.52, ylim3 * 0.85,
         f"fp32 均值: {conf_fp32_ok.mean():.4f}\nint8 均值: {conf_int8_ok.mean():.4f}",
         fontproperties=zh_font, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── 子图4：预测一致性饼图 ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
sizes_pie   = [both_correct, fp32_only, int8_only, both_wrong]
labels_pie  = [f"两者都对\n{both_correct}张",
               f"仅float32对\n{fp32_only}张",
               f"仅int8对\n{int8_only}张",
               f"两者都错\n{both_wrong}张"]
colors_pie  = ["#4CAF50", COLORS["fp32"], COLORS["int8"], "#BDBDBD"]
explode_pie = (0, 0.06, 0.06, 0.04)
wedges, texts, autotexts = ax4.pie(
    sizes_pie, labels=labels_pie, colors=colors_pie, explode=explode_pie,
    autopct="%1.1f%%", startangle=90,
    textprops={"fontproperties": zh_font, "fontsize": 8}
)
for at in autotexts:
    at.set_fontsize(8)
ax4.set_title("预测一致性分析", fontproperties=zh_font, fontsize=12)

# ── 子图5：置信度变化散点图 ───────────────────
ax5  = fig.add_subplot(gs[1, 1])
conf_fp32_all = np.max(fp32_prob, axis=1)
conf_int8_all = np.max(int8_prob, axis=1)
conf_diff     = conf_int8_all - conf_fp32_all

rng = np.random.default_rng(42)
if mask_bc.sum() > 0:
    idx_bc = rng.choice(np.where(mask_bc)[0], min(1000, mask_bc.sum()), replace=False)
    ax5.scatter(conf_fp32_all[idx_bc], conf_diff[idx_bc],
                alpha=0.15, s=5, color="#4CAF50", label=f"两者都对({mask_bc.sum()})")
if fp32_only > 0:
    ax5.scatter(conf_fp32_all[mask_fo], conf_diff[mask_fo],
                alpha=0.8, s=22, color=COLORS["fp32"],
                label=f"仅float32对({fp32_only})", zorder=4)
if int8_only > 0:
    ax5.scatter(conf_fp32_all[mask_io], conf_diff[mask_io],
                alpha=0.8, s=22, color=COLORS["int8"],
                label=f"仅int8对({int8_only})", zorder=4)
if mask_bw.sum() > 0:
    idx_bw = rng.choice(np.where(mask_bw)[0], min(200, mask_bw.sum()), replace=False)
    ax5.scatter(conf_fp32_all[idx_bw], conf_diff[idx_bw],
                alpha=0.5, s=10, color="#BDBDBD", label=f"两者都错({mask_bw.sum()})")

ax5.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax5.set_xlabel("float32 置信度", fontproperties=zh_font)
ax5.set_ylabel("量化后置信度变化\n(int8 − float32)", fontproperties=zh_font)
ax5.set_title("量化对置信度的影响", fontproperties=zh_font, fontsize=12)
ax5.legend(prop=zh_font, fontsize=8, markerscale=2)
ax5.grid(True, alpha=0.3)

# ── 子图6：分歧样本展示 ───────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

fo_idx = np.where(mask_fo)[0]
io_idx = np.where(mask_io)[0]
n_fo   = min(4, len(fo_idx))
n_io   = min(4, len(io_idx))

# 测试集原始图片（uint8，用于可视化）
raw_imgs = test_ds.data.numpy()   # (10000, 28, 28) uint8

if n_fo + n_io > 0:
    n_cols   = max(n_fo, n_io, 1)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, n_cols, subplot_spec=gs[1, 2], hspace=0.35, wspace=0.08
    )
    # 第一行：仅 float32 预测对
    for i in range(n_fo):
        idx = fo_idx[i]
        axi = fig.add_subplot(inner_gs[0, i])
        axi.imshow(raw_imgs[idx], cmap="gray")
        axi.axis("off")
        if i == 0:
            axi.set_title("仅 fp32 对 ↓", fontproperties=zh_font,
                          fontsize=7, color=COLORS["fp32"])
        axi.set_xlabel(f"真:{y_test[idx]} f:{fp32_pred[idx]} q:{int8_pred[idx]}",
                       fontproperties=zh_font, fontsize=6.5)
    # 第二行：仅 int8 预测对
    for i in range(n_io):
        idx = io_idx[i]
        axi = fig.add_subplot(inner_gs[1, i])
        axi.imshow(raw_imgs[idx], cmap="gray")
        axi.axis("off")
        if i == 0:
            axi.set_title("仅 int8 对 ↓", fontproperties=zh_font,
                          fontsize=7, color=COLORS["int8"])
        axi.set_xlabel(f"真:{y_test[idx]} f:{fp32_pred[idx]} q:{int8_pred[idx]}",
                       fontproperties=zh_font, fontsize=6.5)
    fig.text(0.895, 0.27,
             "分歧样本  真:真实标签\nf:fp32预测  q:int8预测",
             ha="center", fontproperties=zh_font, fontsize=8, color="gray")
else:
    ax6.text(0.5, 0.5, "无分歧样本\n（两模型预测完全一致）",
             ha="center", va="center", fontproperties=zh_font, fontsize=12)

plt.savefig("compare_models.png", dpi=150, bbox_inches="tight")
print("\n图表已保存: compare_models.png")
plt.show()

print(f"""
╔══════════════════════════════════════════════════════╗
║              量化对比完成                            ║
╠══════════════════════════════════════════════════════╣
║  float32 准确率 : {fp32_acc:.2f}%{'':<34}║
║  int8    准确率 : {int8_acc:.2f}%{'':<34}║
║  准确率损失     : {acc_loss:.3f}%{'':<35}║
╠══════════════════════════════════════════════════════╣
║  量化权重误差 (max|W_dq - W|):                      ║
║    CONV1_W  : {e1:.6f}{'':<38}║
║    CONV2_W  : {e2:.6f}{'':<38}║
║    DENSE1_W : {ed1:.6f}{'':<38}║
║    DENSE2_W : {ed2:.6f}{'':<38}║
╚══════════════════════════════════════════════════════╝
""")
