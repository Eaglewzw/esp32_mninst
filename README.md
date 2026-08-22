# ESP-Net

<p align="center">English | <a href="README.zh-CN.md">中文</a></p>

A real-time handwritten digit recognition system for the **ESP32 (WT32-SC01)**, featuring a custom C++ CNN inference engine and the LovyanGFX graphics library.

<div align="center">

[![Bilibili](https://img.shields.io/badge/Bilibili-Watch_Demo-ff69b4?style=flat-square&logo=bilibili)](https://www.bilibili.com/video/BV12XdBBdEAA/)
[![Platform](https://img.shields.io/badge/Platform-ESP32-red?style=flat-square)](https://www.espressif.com/en/products/socs/esp32)
[![Framework](https://img.shields.io/badge/Framework-Arduino-blue?style=flat-square)](https://www.arduino.cc/)
[![Training](https://img.shields.io/badge/Training-PyTorch-ee4c2c?style=flat-square)](https://pytorch.org/)

</div>

## How It Works

The system uses the ESP32's dual-core architecture and hand-optimized CNN operators for efficient recognition:

1. **Preprocessing:** The WT32-SC01 capacitive touchscreen captures handwriting strokes in real time, then downsamples and normalizes them into a 28x28 grayscale matrix.
2. **Inference:** The C++ inference engine (`nn_ops`) performs Int8 quantized convolution, batch-normalization fusion, MaxPool, and fully connected operations.
3. **Rendering:** LovyanGFX drives the 3.5-inch display, rendering the drawing canvas and a live bar chart of probabilities for all 10 digit classes.

> ⚠️ **Note:** The model is trained on the standard MNIST dataset and recognizes upright handwritten digits only.

<div align="center">
  <img src="assets/result.gif" width="80%" />
</div>

## Project Structure

```
esp_net/
├── assets/                 # Demo assets and training/evaluation charts
│   └── analysis/          # Model analysis results
├── data/                   # MNIST dataset
├── esp_mnist_arduino/      # ESP32 firmware project
│   ├── assets/            # Hardware and UI documentation images
│   ├── src/               # Firmware source and exported weights
│   └── tools/             # Firmware utility scripts
├── train_mcu.py            # Model training script
├── compare_models.py       # Float32/Int8 model comparison
├── export_weights.py       # Quantized weight export tool
├── download_mnist.py       # MNIST download and preprocessing
├── mnist_gui.py            # Desktop model validation GUI
├── mnist_model.pth         # Trained model
└── model_weights.h         # Generated MCU weight header
```

## Requirements

- **Hardware:** WT32-SC01 (ESP32-WROVER-B) with a 3.5-inch 320x480 capacitive touchscreen
- **Software:** PlatformIO (Arduino), LovyanGFX, and PyTorch 2.x

## Quick Start

Train and deploy the model:

```bash
# 1. Train the model and export Int8 weights
python download_mnist.py
python train_mcu.py
python export_weights.py

# 2. Deploy the firmware
# Copy the generated model_weights.h into esp_mnist_arduino/src/
# Build and flash the esp_mnist_arduino project with PlatformIO
```
