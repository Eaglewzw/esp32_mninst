# ESP-Net

基于 **ESP32 (WT32-SC01)** 的实时手写数字识别系统，集成手写 C++ CNN 推理引擎与 LovyanGFX 图形库。

<div align="center">

[![Bilibili](https://img.shields.io/badge/Bilibili-观看演示-ff69b4?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV12XdBBdEAA/)
[![Platform](https://img.shields.io/badge/Platform-ESP32-red?style=flat-square)](https://www.espressif.com/en/products/socs/esp32)
[![Framework](https://img.shields.io/badge/Framework-Arduino-blue?style=flat-square)](https://www.arduino.cc/)
[![Training](https://img.shields.io/badge/Training-PyTorch-ee4c2c?style=flat-square)](https://pytorch.org/)

</div>

## 系统流程

系统利用 ESP32 的双核架构与手写优化的 CNN 算子实现高效识别：

1. **预处理**：WT32-SC01 电容触摸屏实时采集手写轨迹，自动下采样并归一化为 28x28 灰度矩阵。
2. **推理**：调用C++ 推理引擎 (`nn_ops`)，执行 Int8 量化卷积、BN 融合、MaxPool 与全连接计算。
3. **渲染**：LovyanGFX 驱动 3.5 寸屏幕，实时绘制手写画布与前 10 类数字的概率分布条形图。

> ⚠️ **注意**：模型在标准 MNIST 数据集上训练，仅支持 **正向** 写入识别。

<div align="center">
  <img src="assets/result.gif" width="80%" />
</div>

## 项目结构

```
esp_net/
├── esp_mnist_arduino/   # ESP32 固件实现（推理算子、UI 交互、驱动配置）
├── train_mcu.py         # 模型训练脚本（PyTorch 实现，含数据增强）
├── export_weights.py    # 权重导出工具（模型量化 & C 头文件生成）
├── download_mnist.py    # MNIST 数据集下载与预处理
└── mnist_gui.py         # PC 端模型验证 GUI
```

## 依赖

- **硬件**：WT32-SC01 (ESP32-WROVER-B)、3.5 寸 320x480 电容触摸屏
- **软件**：PlatformIO (Arduino)、LovyanGFX、PyTorch 2.x

## 快速开始

用于训练与部署模型：

```bash
# 1. 训练模型并导出 Int8 权重
python download_mnist.py
python train_mcu.py
python export_weights.py

# 2. 部署固件
# 将生成的 model_weights.h 放入 esp_mnist_arduino/src/
# 使用 PlatformIO 编译并烧录 esp_mnist_arduino 项目
```
