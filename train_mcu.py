"""
MNIST 轻量 CNN 训练（PyTorch + GPU 优化版）
============================================
改进项：
  ✓ PyTorch + CUDA GPU 训练
  ✓ 数据增强（旋转 / 平移 / 缩放 / RandomErasing）
  ✓ Conv 层 Dropout2d + FC 层 Dropout
  ✓ ReduceLROnPlateau 自适应学习率
  ✓ EarlyStopping（监控 val_loss）
  ✓ 模型检查点（保存 val_acc 最高的权重）

输出：
  mnist_model.pth          ← 最佳模型权重
  training_curves.png      ← 训练过程曲线（loss / acc / lr）
  loss_function_design.png ← 损失函数原理图
  confusion_matrix.png     ← 混淆矩阵
  wrong_samples.png        ← 错误样本
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import itertools

# ─────────────────────────────────────────────
# 0. 中文字体 & 设备
# ─────────────────────────────────────────────
_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
zh_font    = fm.FontProperties(fname=_FONT_PATH)
matplotlib.rcParams['font.family']        = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}", end="")
if DEVICE.type == 'cuda':
    print(f"  ({torch.cuda.get_device_name(0)})")
else:
    print()

# ─────────────────────────────────────────────
# 1. 超参数
# ─────────────────────────────────────────────
CFG = dict(
    batch_size   = 1024,
    max_epochs   = 100,
    lr           = 1e-3,
    weight_decay = 1e-4,
    # ReduceLROnPlateau
    lr_factor    = 0.5,
    lr_patience  = 5,
    lr_min       = 1e-6,
    # EarlyStopping
    es_patience  = 15,
    es_min_delta = 1e-4,
    # 数据增强
    aug_rotation  = 10,       # 随机旋转 ±10°
    aug_translate = 0.10,     # 随机平移 ±10%
    aug_scale     = (0.9, 1.1),
    aug_erase_p   = 0.15,     # RandomErasing 概率
    # Dropout
    drop_conv1 = 0.10,
    drop_conv2 = 0.15,
    drop_fc    = 0.40,
)

# ─────────────────────────────────────────────
# 2. 数据集（带增强）
# ─────────────────────────────────────────────
_MEAN, _STD = (0.1307,), (0.3081,)

train_transform = T.Compose([
    T.RandomRotation(CFG['aug_rotation']),
    T.RandomAffine(degrees=0,
                   translate=(CFG['aug_translate'], CFG['aug_translate']),
                   scale=CFG['aug_scale']),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
    T.RandomErasing(p=CFG['aug_erase_p'], scale=(0.02, 0.12), value=0),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

train_ds = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=train_transform)
test_ds  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"训练集: {len(train_ds):,}  测试集: {len(test_ds):,}")
print(f"数据增强: 旋转±{CFG['aug_rotation']}° | "
      f"平移±{int(CFG['aug_translate']*100)}% | "
      f"缩放{CFG['aug_scale']} | "
      f"RandomErasing p={CFG['aug_erase_p']}")

# ─────────────────────────────────────────────
# 3. 模型（Conv 层加入 Dropout2d）
# ─────────────────────────────────────────────
class MCU_CNN(nn.Module):
    """
    轻量 CNN，专为单片机设计
    Input  [1, 28, 28]
      ↓ Conv(1→8)  + BN + ReLU + Dropout2d + MaxPool → [8, 14, 14]
      ↓ Conv(8→16) + BN + ReLU + Dropout2d + MaxPool → [16, 7, 7]
      ↓ Flatten → Linear(784→64) + ReLU + Dropout
      ↓ Linear(64→10)
    """
    def __init__(self, drop_conv1=0.10, drop_conv2=0.15, drop_fc=0.40):
        super().__init__()
        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv1),          # 随机将整个 feature map 置零
            nn.MaxPool2d(2),                   # 28→14

            # ── Block 2 ──────────────────────
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_conv2),
            nn.MaxPool2d(2),                   # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 16×7×7 = 784
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MCU_CNN(CFG['drop_conv1'], CFG['drop_conv2'], CFG['drop_fc']).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {total_params:,}  ({total_params*4/1024:.1f} KB float32)")
print(model)

# ─────────────────────────────────────────────
# 4. 损失 / 优化器 / 调度器
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)   # label smoothing 防过拟合

optimizer = optim.AdamW(model.parameters(),
                        lr=CFG['lr'],
                        weight_decay=CFG['weight_decay'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',
    factor=CFG['lr_factor'],
    patience=CFG['lr_patience'],
    min_lr=CFG['lr_min'],
)

# ─────────────────────────────────────────────
# 5. EarlyStopping
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

early_stopping = EarlyStopping(CFG['es_patience'], CFG['es_min_delta'])

# ─────────────────────────────────────────────
# 6. 训练 / 验证函数
# ─────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = total_correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                           labels.to(DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out  = model(imgs)
            loss = criterion(out, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += out.argmax(1).eq(labels).sum().item()
            total         += imgs.size(0)

    return total_loss / total, total_correct / total * 100


# ─────────────────────────────────────────────
# 7. 训练主循环
# ─────────────────────────────────────────────
history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
best_acc   = 0.0
MODEL_PATH = 'mnist_model.pth'

print(f"\n{'═'*62}")
print(f"  开始训练（最多 {CFG['max_epochs']} epochs，EarlyStopping patience={CFG['es_patience']}）")
print(f"{'═'*62}")
print(f"{'Epoch':>6}  {'TrainLoss':>9}  {'ValLoss':>9}  {'TrainAcc':>9}  {'ValAcc':>9}  {'LR':>9}  {'Ckpt':>4}")
print(f"{'─'*62}")

for epoch in range(1, CFG['max_epochs'] + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(test_loader,  train=False)
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    # 检查点：val_acc 最高时保存
    ckpt = ''
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        ckpt = '✓'

    print(f"{epoch:>6}  {train_loss:>9.4f}  {val_loss:>9.4f}  "
          f"{train_acc:>8.2f}%  {val_acc:>8.2f}%  {current_lr:>9.2e}  {ckpt:>4}")

    # 调度器 & EarlyStopping（都依赖 val_loss）
    scheduler.step(val_loss)
    early_stopping(val_loss)
    if early_stopping.stop:
        print(f"\n  EarlyStopping 触发（连续 {CFG['es_patience']} epoch val_loss 无改善）")
        break

actual_epochs = len(history['train_loss'])
print(f"\n  实际训练轮次: {actual_epochs}")
print(f"  最佳 val_acc : {best_acc:.2f}%")
print(f"  模型已保存   : {MODEL_PATH}")

# 加载最佳权重，评估最终测试准确率
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_, final_acc = run_epoch(test_loader, train=False)
print(f"  最终测试准确率: {final_acc:.2f}%")

# ─────────────────────────────────────────────
# 8. 图1：训练过程曲线（Loss / Accuracy / LR）
# ─────────────────────────────────────────────
epochs_range = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("训练过程曲线（PyTorch + GPU + 数据增强）",
             fontsize=14, fontweight='bold', fontproperties=zh_font)

# Loss
ax = axes[0]
ax.plot(epochs_range, history['train_loss'], 'b-o', markersize=3, linewidth=1.5, label='训练 Loss')
ax.plot(epochs_range, history['val_loss'],   'r-s', markersize=3, linewidth=1.5, label='验证 Loss')
best_ep = int(np.argmin(history['val_loss'])) + 1
ax.axvline(best_ep, color='green', linestyle='--', alpha=0.6, label=f'最佳 epoch={best_ep}')
ax.set_xlabel('Epoch', fontproperties=zh_font)
ax.set_ylabel('Loss（含 label smoothing）', fontproperties=zh_font)
ax.set_title('损失曲线', fontproperties=zh_font)
ax.legend(prop=zh_font, fontsize=9)
ax.grid(True, alpha=0.3)

# Accuracy
ax = axes[1]
ax.plot(epochs_range, history['train_acc'], 'b-o', markersize=3, linewidth=1.5, label='训练准确率')
ax.plot(epochs_range, history['val_acc'],   'r-s', markersize=3, linewidth=1.5, label='验证准确率')
ax.axvline(best_ep, color='green', linestyle='--', alpha=0.6)
ax.annotate(f"最高 {best_acc:.2f}%\n(第{best_ep}轮)",
            xy=(best_ep, best_acc),
            xytext=(best_ep + max(1, actual_epochs*0.05), best_acc - 0.8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontproperties=zh_font, color='red', fontsize=8)
ax.set_xlabel('Epoch', fontproperties=zh_font)
ax.set_ylabel('准确率 (%)', fontproperties=zh_font)
ax.set_title('准确率曲线', fontproperties=zh_font)
ax.legend(prop=zh_font, fontsize=9)
ax.grid(True, alpha=0.3)

# Learning Rate
ax = axes[2]
ax.semilogy(epochs_range, history['lr'], 'g-o', markersize=3, linewidth=1.5)
# 标注 LR 下降点
lr_arr = np.array(history['lr'])
drop_eps = np.where(np.diff(lr_arr) < 0)[0] + 2
for ep in drop_eps:
    ax.axvline(ep, color='orange', linestyle=':', alpha=0.8)
    ax.annotate(f'×{CFG["lr_factor"]}', xy=(ep, lr_arr[ep-1]),
                xytext=(ep+0.5, lr_arr[ep-1]*1.5),
                fontproperties=zh_font, fontsize=7, color='orange')
ax.set_xlabel('Epoch', fontproperties=zh_font)
ax.set_ylabel('Learning Rate (log)', fontproperties=zh_font)
ax.set_title('学习率曲线（ReduceLROnPlateau）', fontproperties=zh_font)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print('\n[图1] 训练曲线已保存: training_curves.png')
plt.show()

# ─────────────────────────────────────────────
# 9. 图2：损失函数原理（CrossEntropy + Label Smoothing）
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('损失函数设计：交叉熵 + Label Smoothing (ε=0.05)',
             fontsize=13, fontweight='bold', fontproperties=zh_font)

# 子图A：-log(p) 对比 label smoothing 后的 loss
ax = axes[0]
p   = np.linspace(0.001, 1.0, 300)
eps = 0.05
K   = 10
# 标准交叉熵: L = -log(p)
# Label smoothing: 真实标签变成 (1-ε) + ε/K，其余变成 ε/K
# L_ls = -(1-ε+ε/K)*log(p) - (K-1)*(ε/K)*log((1-p)/(K-1)+1e-9)  ← 近似
# 简化显示：对正确类的 loss
smooth_target = 1 - eps + eps / K   # ≈ 0.955
loss_ce       = -np.log(p)
loss_ls       = -smooth_target * np.log(p)   # 主项（简化）

ax.plot(p, loss_ce, 'b-',  linewidth=2, label='标准交叉熵 -log(p)')
ax.plot(p, loss_ls, 'r--', linewidth=2, label=f'Label Smoothing ε=0.05\n目标概率≈{smooth_target:.3f}')
ax.set_xlabel('正确类别预测概率 p', fontproperties=zh_font)
ax.set_ylabel('Loss 值', fontproperties=zh_font)
ax.set_title('Label Smoothing 降低过度自信', fontproperties=zh_font)
ax.legend(prop=zh_font, fontsize=9)
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)
ax.fill_between(p, loss_ce, loss_ls, alpha=0.15, color='green',
                label='差值（惩罚减少）')

# 子图B：Dropout2d 示意（feature map 随机置零）
ax = axes[1]
np.random.seed(42)
fmap = np.random.rand(8, 4, 4)   # 8个 feature map，4×4
dropped = fmap.copy()
drop_channels = np.random.choice(8, 2, replace=False)  # 随机 drop 2个通道
dropped[drop_channels] = 0.0

# 展示前4个通道
for i in range(4):
    ax_sub = plt.axes([0.375 + (i%2)*0.065, 0.55 - (i//2)*0.22, 0.055, 0.18])
    color = 'gray' if i in drop_channels else None
    ax_sub.imshow(dropped[i], cmap='Blues' if i not in drop_channels else 'Greys',
                  vmin=0, vmax=1)
    ax_sub.set_title(f'ch{i}' + (' ✗' if i in drop_channels else ''),
                     fontsize=7, color='red' if i in drop_channels else 'black')
    ax_sub.axis('off')

ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.5, 0.92, 'Dropout2d：整通道置零',
        ha='center', fontproperties=zh_font, fontsize=11, fontweight='bold')
ax.text(0.5, 0.78, '→ 强迫其他通道学习冗余特征\n→ 防止特征图间共适应',
        ha='center', fontproperties=zh_font, fontsize=9, color='gray')
ax.text(0.5, 0.12, f'Conv1 p={CFG["drop_conv1"]}  Conv2 p={CFG["drop_conv2"]}  FC p={CFG["drop_fc"]}',
        ha='center', fontproperties=zh_font, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 子图C：数据增强效果展示
ax = axes[2]
sample_img, sample_lbl = test_ds[0]
sample_np = (sample_img.squeeze().numpy() * _STD[0] + _MEAN[0])  # 反归一化

# 生成4张增强图
aug_imgs = []
raw_pil  = torchvision.datasets.MNIST('./data', train=False, download=False,
                                       transform=None)[0][0]  # PIL image
for _ in range(6):
    t = train_transform(raw_pil)
    aug_imgs.append((t.squeeze().numpy() * _STD[0] + _MEAN[0]).clip(0, 1))

axes[2].axis('off')
gs_inner = fig.add_gridspec(2, 4,
    left=0.69, right=0.99, top=0.88, bottom=0.12, hspace=0.05, wspace=0.05)
ax_orig = fig.add_subplot(gs_inner[0, 0])
ax_orig.imshow(sample_np.clip(0,1), cmap='gray')
ax_orig.set_title('原图', fontproperties=zh_font, fontsize=8)
ax_orig.axis('off')
for i in range(6):
    r, c = divmod(i + 1, 4)
    ax_a = fig.add_subplot(gs_inner[r, c])
    ax_a.imshow(aug_imgs[i], cmap='gray', vmin=0, vmax=1)
    ax_a.set_title(f'增强{i+1}', fontproperties=zh_font, fontsize=8)
    ax_a.axis('off')
ax_blank = fig.add_subplot(gs_inner[1, 3])
ax_blank.axis('off')

plt.savefig('loss_function_design.png', dpi=150, bbox_inches='tight')
print('[图2] 损失函数原理图已保存: loss_function_design.png')
plt.show()

# ─────────────────────────────────────────────
# 10. 全量推理获取预测结果（用于混淆矩阵 & 错误样本）
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(DEVICE))
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# ─────────────────────────────────────────────
# 11. 图3：混淆矩阵
# ─────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_title(f'混淆矩阵（测试集 {len(test_ds):,} 张）\n对角线=正确，其余=误判',
             fontproperties=zh_font, fontsize=13)
ax.set_xlabel('预测标签', fontproperties=zh_font)
ax.set_ylabel('真实标签', fontproperties=zh_font)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
thresh = cm.max() / 2
for i, j in itertools.product(range(10), range(10)):
    ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
            color='white' if cm[i,j] > thresh else 'black',
            fontsize=10, fontweight='bold' if i == j else 'normal')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print('[图3] 混淆矩阵已保存: confusion_matrix.png')
plt.show()

# ─────────────────────────────────────────────
# 12. 图4：错误样本
# ─────────────────────────────────────────────
wrong_idx = np.where(all_preds != all_labels)[0]

# 从测试集取原始（未归一化）图片用于显示
raw_test_imgs = test_ds.data.numpy()   # (10000, 28, 28) uint8

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
fig.suptitle(f'识别错误的样本（共 {len(wrong_idx)} 张，错误率 {len(wrong_idx)/100:.1f}%）',
             fontproperties=zh_font, fontsize=13, fontweight='bold')
for ax, idx in zip(axes.flat, wrong_idx[:20]):
    ax.imshow(raw_test_imgs[idx], cmap='gray')
    ax.set_title(f'真实:{all_labels[idx]}  预测:{all_preds[idx]}',
                 fontproperties=zh_font, fontsize=10, color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig('wrong_samples.png', dpi=150, bbox_inches='tight')
print('[图4] 错误样本已保存: wrong_samples.png')
plt.show()

print(f"""
╔══════════════════════════════════════════════════╗
║                  训练完成                        ║
╠══════════════════════════════════════════════════╣
║  设备     : {str(DEVICE):<38}║
║  实际轮次 : {actual_epochs:<38}║
║  最佳准确率: {best_acc:.2f}%{'':<34}║
║  模型文件 : mnist_model.pth{'':<26}║
╠══════════════════════════════════════════════════╣
║  下一步: python export_weights.py                ║
║  → 从 .pth 导出量化权重 model_weights.h         ║
╚══════════════════════════════════════════════════╝
""")
