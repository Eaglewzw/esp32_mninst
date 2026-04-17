"""
MNIST 数据集下载 & 可视化脚本
用途: 下载数据到 ./data/，展示样本图像，打印数据集统计信息
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np

# ── 中文字体配置 ──────────────────────────────
_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
zh_font    = fm.FontProperties(fname=_FONT_PATH)
matplotlib.rcParams['font.family']       = 'WenQuanYi Micro Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 下载 MNIST 数据集到当前目录 ./data/
# ─────────────────────────────────────────────
print("=" * 50)
print("  正在下载 MNIST 数据集 ...")
print("=" * 50)

transform = transforms.Compose([
    transforms.ToTensor(),               # 转为 Tensor，像素归一化到 [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 官方均值/标准差
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ─────────────────────────────────────────────
# 2. 数据集统计信息
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  数据集统计")
print(f"{'─'*50}")
print(f"  训练集大小 : {len(train_dataset):,} 张")
print(f"  测试集大小 : {len(test_dataset):,} 张")
print(f"  图像尺寸   : {train_dataset[0][0].shape}  (C, H, W)")
print(f"  类别数量   : {len(train_dataset.classes)} 类 → {train_dataset.classes}")

# 计算各类别样本数
label_counts = torch.zeros(10, dtype=torch.int)
for _, label in train_dataset:
    label_counts[label] += 1
print(f"\n  训练集各数字样本数:")
for i, count in enumerate(label_counts):
    bar = "█" * (count.item() // 300)
    print(f"    数字 {i}: {count.item():,}  {bar}")


# ─────────────────────────────────────────────
# 3. 可视化 1：5×10 样本总览（每类5张）
# ─────────────────────────────────────────────
def show_samples_per_class(dataset, samples_per_class=5):
    """每个数字类别各取 N 张，展示成网格"""
    # 收集每类图片（反归一化回原始像素）
    class_images = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(class_images[label]) < samples_per_class:
            # 反归一化: x * std + mean
            img_raw = img * 0.3081 + 0.1307
            class_images[label].append(img_raw.squeeze().numpy())
        if all(len(v) == samples_per_class for v in class_images.values()):
            break

    fig, axes = plt.subplots(
        10, samples_per_class,
        figsize=(samples_per_class * 1.4, 10 * 1.4)
    )
    fig.suptitle("MNIST 数据集 — 每类随机样本", fontsize=14, fontweight='bold',
                 y=1.01, fontproperties=zh_font)

    for digit in range(10):
        for j in range(samples_per_class):
            ax = axes[digit][j]
            ax.imshow(class_images[digit][j], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f"数字 {digit}", fontsize=10, rotation=0,
                              labelpad=30, va='center',
                              fontproperties=zh_font)

    plt.tight_layout()
    plt.savefig("mnist_samples.png", dpi=150, bbox_inches='tight')
    print("\n  [图1] 样本总览已保存: mnist_samples.png")
    plt.show()


# ─────────────────────────────────────────────
# 4. 可视化 2：单张图详细分析（像素热力图）
# ─────────────────────────────────────────────
def show_single_image_detail(dataset, index=0):
    """展示单张图的原图、像素值热力图、行/列像素强度曲线"""
    img_tensor, label = dataset[index]
    img_raw = (img_tensor * 0.3081 + 0.1307).squeeze().numpy()  # 反归一化
    img_norm = img_tensor.squeeze().numpy()                       # 归一化后

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(f"MNIST 单张详细分析 — 标签: 数字 {label}", fontsize=13,
                 fontweight='bold', fontproperties=zh_font)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    # 子图1: 原始图像
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_raw, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("原始图像 (28×28)", fontproperties=zh_font)
    ax1.axis('off')

    # 子图2: 像素值热力图（归一化值）
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(img_norm, cmap='hot', vmin=-1, vmax=2)
    ax2.set_title("像素强度热力图", fontproperties=zh_font)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # 子图3: 每行像素均值曲线（垂直方向分布）
    ax3 = fig.add_subplot(gs[2])
    row_means = img_raw.mean(axis=1)
    ax3.plot(row_means, range(28), color='steelblue', linewidth=2)
    ax3.invert_yaxis()
    ax3.set_title("行均值 (垂直)", fontproperties=zh_font)
    ax3.set_xlabel("像素均值", fontproperties=zh_font)
    ax3.set_ylabel("行号",     fontproperties=zh_font)
    ax3.grid(True, alpha=0.3)

    # 子图4: 每列像素均值曲线（水平方向分布）
    ax4 = fig.add_subplot(gs[3])
    col_means = img_raw.mean(axis=0)
    ax4.plot(range(28), col_means, color='tomato', linewidth=2)
    ax4.set_title("列均值 (水平)", fontproperties=zh_font)
    ax4.set_xlabel("列号",   fontproperties=zh_font)
    ax4.set_ylabel("像素均值", fontproperties=zh_font)
    ax4.grid(True, alpha=0.3)

    plt.savefig("mnist_detail.png", dpi=150, bbox_inches='tight')
    print("  [图2] 单张详细分析已保存: mnist_detail.png")
    plt.show()


# ─────────────────────────────────────────────
# 5. 可视化 3：像素值分布直方图
# ─────────────────────────────────────────────
def show_pixel_distribution(dataset, num_samples=1000):
    """展示数据集像素值的整体分布"""
    all_pixels = []
    for i, (img, _) in enumerate(dataset):
        if i >= num_samples:
            break
        all_pixels.extend(img.numpy().flatten().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MNIST 像素值分布统计", fontsize=13, fontweight='bold',
                 fontproperties=zh_font)

    # 归一化像素分布
    axes[0].hist(all_pixels, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].set_title(f"归一化像素分布 (前 {num_samples} 张)", fontproperties=zh_font)
    axes[0].set_xlabel("像素值 (归一化后)", fontproperties=zh_font)
    axes[0].set_ylabel("频次",             fontproperties=zh_font)
    axes[0].grid(True, alpha=0.3)

    # 各类别样本数柱状图
    label_counts_np = label_counts.numpy()
    bars = axes[1].bar(range(10), label_counts_np, color=plt.cm.tab10(np.linspace(0, 1, 10)))
    axes[1].set_title("训练集各类别样本数", fontproperties=zh_font)
    axes[1].set_xlabel("数字类别",         fontproperties=zh_font)
    axes[1].set_ylabel("样本数量",         fontproperties=zh_font)
    axes[1].set_xticks(range(10))
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, label_counts_np):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("mnist_distribution.png", dpi=150, bbox_inches='tight')
    print("  [图3] 像素分布统计已保存: mnist_distribution.png")
    plt.show()


# ─────────────────────────────────────────────
# 执行所有可视化
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*50}")
    print("  开始生成可视化图表 ...")
    print(f"{'─'*50}")

    show_samples_per_class(train_dataset, samples_per_class=5)
    show_single_image_detail(train_dataset, index=0)
    show_pixel_distribution(train_dataset, num_samples=2000)

    print(f"\n{'='*50}")
    print("  完成！数据已保存至 ./data/")
    print("  生成图片: mnist_samples.png")
    print("             mnist_detail.png")
    print("             mnist_distribution.png")
    print(f"{'='*50}")
