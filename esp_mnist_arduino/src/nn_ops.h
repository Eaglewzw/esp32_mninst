/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║  nn_ops.h  —  MNIST CNN 推理算子库（手写实现，无第三方依赖）║
 * ║                                                              ║
 * ║  支持模型结构：                                              ║
 * ║    Conv2D(8,3,same,no_bias) → BN → ReLU → MaxPool2D(2×2)   ║
 * ║    Conv2D(16,3,same,no_bias) → BN → ReLU → MaxPool2D(2×2)  ║
 * ║    Flatten → Dense(64,relu) → Dense(10,softmax)             ║
 * ║                                                              ║
 * ║  权重格式：int8 量化 + per-tensor scale/zero_point          ║
 * ║  激活格式：float32                                           ║
 * ║                                                              ║
 * ║  内存占用（激活缓冲区，静态分配在 .bss）：                  ║
 * ║    act_a : 28×28×8  = 6272 floats ≈ 24.5 KB               ║
 * ║    act_b : 14×14×8  = 1568 floats ≈  6.1 KB               ║
 * ║    dense : 64 floats ≈ 256 B                                ║
 * ║    共计  ≈ 31 KB (ESP32 有 ~320 KB DRAM，充裕)             ║
 * ╚══════════════════════════════════════════════════════════════╝
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

// ═══════════════════════════════════════════════════════════════
//  基础算子（可单独测试）
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Conv2D — SAME 填充，3×3 卷积核，无 bias，int8 权重
 *
 * 权重内存布局（C 数组顺序，与 Keras 导出脚本保持一致）：
 *   weights[ C_out ][ 3 ][ 3 ][ C_in ]
 *   访问：weights[(oc*3 + kh)*3*C_in + kw*C_in + ic]
 *
 * @param input   输入张量  [H × W × C_in]  (channels-last)
 * @param output  输出张量  [H × W × C_out] (channels-last)
 * @param H, W    输入/输出的空间尺寸（SAME 填充，尺寸不变）
 * @param C_in    输入通道数
 * @param C_out   输出通道数
 * @param weights int8 权重，量化格式：real = (q - w_zp) * w_scale
 * @param w_scale 权重反量化比例
 * @param w_zp    权重零点
 */
void nn_conv2d_same(
    const float*   input,
    float*         output,
    int H, int W, int C_in, int C_out,
    const int8_t*  weights,
    float          w_scale,
    int8_t         w_zp
);

/**
 * @brief BatchNorm — 推理时已折叠为逐通道 scale+bias
 *
 * 原始 BN：y = gamma*(x-mean)/sqrt(var+eps) + beta
 * 折叠后：  y = bn_scale[c]*x + bn_bias[c]
 *   其中 bn_scale[c] = gamma[c] / sqrt(var[c] + eps)
 *        bn_bias[c]  = beta[c] - mean[c] * bn_scale[c]
 * （折叠由 Python 导出脚本完成，见 export_weights.py）
 *
 * @param inout   原地修改，[H × W × C]（channels-last）
 * @param total   总元素数 = H * W * C
 * @param C       通道数（每 C 个元素对应一组参数）
 * @param bn_scale 折叠后的 scale，长度 C
 * @param bn_bias  折叠后的 bias，长度 C
 */
void nn_batchnorm(
    float*         inout,
    int            total,
    int            C,
    const float*   bn_scale,
    const float*   bn_bias
);

/**
 * @brief ReLU — 原地，max(0, x)
 *
 * @param inout 原地修改
 * @param total 总元素数
 */
void nn_relu(float* inout, int total);

/**
 * @brief MaxPool2D — 2×2 窗口，stride=2，不填充
 *
 * 输入尺寸 [H][W][C] → 输出 [H/2][W/2][C]
 * H、W 必须为偶数
 *
 * @param input   输入张量  [H × W × C]
 * @param output  输出张量  [(H/2) × (W/2) × C]
 * @param H, W, C 输入的高、宽、通道
 */
void nn_maxpool2d(
    const float*   input,
    float*         output,
    int H, int W, int C
);

/**
 * @brief Dense — 全连接层，int8 权重
 *
 * 权重内存布局：weights[ out_features ][ in_features ]
 * 计算：output[o] = bias[o] + sum_i{ (weights[o][i] - w_zp) * w_scale * input[i] }
 * 激活函数由调用方决定（如后接 nn_relu 或 nn_softmax）
 *
 * @param input       输入向量 [in_features]
 * @param output      输出向量 [out_features]
 * @param in_features 输入维度
 * @param out_features 输出维度
 * @param weights     int8 权重
 * @param w_scale     权重反量化比例
 * @param w_zp        权重零点
 * @param bias        float32 偏置（可为 nullptr 表示无偏置）
 */
void nn_dense(
    const float*   input,
    float*         output,
    int            in_features,
    int            out_features,
    const int8_t*  weights,
    float          w_scale,
    int8_t         w_zp,
    const float*   bias
);

/**
 * @brief Softmax — 原地，数值稳定版（减去 max 防溢出）
 *
 * @param inout 原地修改
 * @param N     元素数
 */
void nn_softmax(float* inout, int N);

// ═══════════════════════════════════════════════════════════════
//  顶层推理接口
// ═══════════════════════════════════════════════════════════════

/**
 * @brief 执行 MNIST CNN 完整推理
 *
 * @param input  28×28 灰度图，float，归一化到 [0.0, 1.0]
 *               像素值约定：0.0 = 黑色背景，1.0 = 白色笔迹
 *               （与 MNIST 数据集一致，注意部分实现是反的）
 * @param output 10 类 Softmax 概率，output[i] = 数字 i 的置信度
 *               sum(output) ≈ 1.0
 *
 * @note  推理期间使用内部静态缓冲区（非线程安全，单核 ESP32 无碍）
 */
void mnist_infer(const float input[28][28], float output[10]);

/**
 * @brief 从推理结果中取最大置信度的类别
 *
 * @param probs  nn_infer 输出的 10 元素概率数组
 * @return int   预测数字 0~9
 */
int  mnist_argmax(const float probs[10]);
