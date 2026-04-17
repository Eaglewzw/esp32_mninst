/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║  nn_ops.cpp  —  MNIST CNN 算子实现                          ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 *  ★ 激活格式：channels-first（与 PyTorch 一致）
 *    tensor[c][h][w]  ←→  flat[c * H * W + h * W + w]
 *
 *  ★ Conv 权重格式：[C_out][kH][kW][C_in]
 *    （导出脚本已将 PyTorch [C_out,C_in,kH,kW] 转置为此布局）
 *    flat idx = ((oc * K + kh) * K + kw) * C_in + ic
 *
 *  ★ Dense 权重格式：[out][in]  (PyTorch Linear.weight 原生)
 *
 *  ★ Dense1 输入维度说明：
 *    act_b 在 channels-first 下为 [C=16][H=7][W=7]，展平顺序是
 *    c*49 + h*7 + w，与 PyTorch Flatten 后再转置 dense 权重一致
 *    （详见 model_weights.h 导出脚本）
 *
 *  ★ 量化格式（对称 per-tensor，zero_point = 0）：
 *    real = int8_value * scale
 *
 *  ─────────────────────────────────────────────────────────────
 *  Bug fix log（对比初始版）：
 *   [BUG-1] 激活格式：channels-last [H][W][C] → channels-first [C][H][W]
 *           BN 通道索引 i%C → 块状 c*HW，MaxPool 访问模式同步修正
 *   [BUG-2] Dense bias：(bias+acc)*scale → acc*scale + bias
 *           bias 存储真实 float，不再被 scale 缩放
 *  ─────────────────────────────────────────────────────────────
 */

#include "nn_ops.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════
//  静态激活缓冲区（channels-first，ESP32 .bss 段，避免栈溢出）
//
//  复用计划（→ 表示写入目标）：
//    in_flat(784)  → act_a: Conv1 [8][28][28] = 6272 f  → BN/ReLU 原地
//    act_a         → act_b: Pool1 [8][14][14] = 1568 f
//    act_b         → act_a: Conv2 [16][14][14]= 3136 f  → BN/ReLU 原地
//    act_a         → act_b: Pool2 [16][7][7]  =  784 f  (act_b 容量 1568 ≥ 784 ✓)
//    act_b(784)    → dbuf:  Dense1[64]
//    dbuf          → out:   Dense2[10] → Softmax
// ═══════════════════════════════════════════════════════════════
static float act_a[8 * 28 * 28];   // 6272 floats ≈ 24.5 KB
static float act_b[8 * 14 * 14];   // 1568 floats ≈  6.1 KB  (后续复用存 784 f)
static float dense_buf[64];

// ═══════════════════════════════════════════════════════════════
//  nn_conv2d_same
//  激活：channels-first [C][H][W]
//  权重：[C_out][kH][kW][C_in]
//    flat idx = ((oc * K + kh) * K + kw) * C_in + ic
//    （导出脚本做了 np.transpose(w, (0,2,3,1))，即 [Co,Ci,kH,kW]→[Co,kH,kW,Ci]）
// ═══════════════════════════════════════════════════════════════
void nn_conv2d_same(
    const float*  input,      // [C_in * H * W]  channels-first
    float*        output,     // [C_out * H * W] channels-first
    int H, int W, int C_in, int C_out,
    const int8_t* weights,    // [C_out][kH=3][kW=3][C_in]
    float         w_scale,
    int8_t        w_zp)
{
    const int K = 3, PAD = 1;
    const int HW = H * W;

    for (int oc = 0; oc < C_out; oc++) {
        float* out_plane = output + oc * HW;
        // weights[oc][0][0][0] 基址
        const int8_t* w_oc = weights + oc * K * K * C_in;

        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {

                float acc = 0.0f;

                for (int kh = 0; kh < K; kh++) {
                    int ih = oh + kh - PAD;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < K; kw++) {
                        int iw = ow + kw - PAD;
                        if (iw < 0 || iw >= W) continue;

                        // 权重：weights[oc][kh][kw][ic]
                        const int8_t* w_ptr = w_oc + (kh * K + kw) * C_in;

                        for (int ic = 0; ic < C_in; ic++) {
                            // 激活 channels-first: input[ic][ih][iw]
                            float in_val = input[ic * HW + ih * W + iw];
                            acc += in_val * (float)(w_ptr[ic] - w_zp);
                        }
                    }
                }

                out_plane[oh * W + ow] = acc * w_scale;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  nn_batchnorm
//  激活：channels-first [C][H][W]
//  通道 c 对应 inout[c * (total/C) .. (c+1)*(total/C) - 1]
// ═══════════════════════════════════════════════════════════════
void nn_batchnorm(
    float*        inout,
    int           total,
    int           C,
    const float*  bn_scale,
    const float*  bn_bias)
{
    // [BUG-1 fix] channels-first：连续 HW 个元素属于同一通道
    const int HW = total / C;
    for (int c = 0; c < C; c++) {
        float s = bn_scale[c];
        float b = bn_bias[c];
        float* p = inout + c * HW;
        for (int i = 0; i < HW; i++) {
            p[i] = s * p[i] + b;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  nn_relu  （与数据格式无关，原地即可）
// ═══════════════════════════════════════════════════════════════
void nn_relu(float* inout, int total)
{
    for (int i = 0; i < total; i++) {
        if (inout[i] < 0.0f) inout[i] = 0.0f;
    }
}

// ═══════════════════════════════════════════════════════════════
//  nn_maxpool2d
//  激活：channels-first [C][H][W] → [C][H/2][W/2]
//  2×2 窗口，stride=2，无填充
// ═══════════════════════════════════════════════════════════════
void nn_maxpool2d(
    const float* input,
    float*       output,
    int H, int W, int C)
{
    const int OH = H / 2, OW = W / 2;

    for (int c = 0; c < C; c++) {
        // [BUG-1 fix] 逐通道处理，每个通道是一块连续的 H×W 平面
        const float* in_c  = input  + c * H  * W;
        float*       out_c = output + c * OH * OW;

        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int ih0 = oh * 2, iw0 = ow * 2;

                float v0 = in_c[ ih0      * W + iw0    ];
                float v1 = in_c[ ih0      * W + iw0 + 1];
                float v2 = in_c[(ih0 + 1) * W + iw0    ];
                float v3 = in_c[(ih0 + 1) * W + iw0 + 1];

                float mx = v0;
                if (v1 > mx) mx = v1;
                if (v2 > mx) mx = v2;
                if (v3 > mx) mx = v3;

                out_c[oh * OW + ow] = mx;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  nn_dense
//  weights[out_features][in_features]  ← PyTorch Linear.weight 原生布局
//
//  [BUG-3 fix] 正确公式：
//    output[o] = Σ_i (w_q[o][i] - w_zp) * input[i] * w_scale  +  bias[o]
//                └──────────────── acc ───────────────────────┘
//  旧版错误地把 bias 和 acc 一起乘以 w_scale，导致 bias 被缩小 ~scale 倍
// ═══════════════════════════════════════════════════════════════
void nn_dense(
    const float*  input,
    float*        output,
    int           in_features,
    int           out_features,
    const int8_t* weights,
    float         w_scale,
    int8_t        w_zp,
    const float*  bias)
{
    for (int o = 0; o < out_features; o++) {
        const int8_t* w_row = weights + o * in_features;
        float acc = 0.0f;

        for (int i = 0; i < in_features; i++) {
            acc += input[i] * (float)(w_row[i] - w_zp);
        }

        // [BUG-3 fix] bias 在 scale 外部相加（bias 存储为真实 float 值）
        output[o] = acc * w_scale + (bias ? bias[o] : 0.0f);
    }
}

// ═══════════════════════════════════════════════════════════════
//  nn_softmax  （数值稳定版，减去 max 防上溢）
// ═══════════════════════════════════════════════════════════════
void nn_softmax(float* inout, int N)
{
    float mx = inout[0];
    for (int i = 1; i < N; i++) {
        if (inout[i] > mx) mx = inout[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        inout[i] = expf(inout[i] - mx);
        sum += inout[i];
    }

    float inv = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        inout[i] *= inv;
    }
}

// ═══════════════════════════════════════════════════════════════
//  mnist_infer  —  完整推理流水线
//
//  输入约定：
//    input[28][28]  float，已归一化到 [0.0, 1.0]
//    0.0 = 黑色背景，1.0 = 白色笔迹（与 MNIST 数据集一致）
//
//  数据格式：全程 channels-first，与 PyTorch 一致
//    C_in = 1 时，float[1][28][28] 与 float[28][28] 内存完全相同 ✓
// ═══════════════════════════════════════════════════════════════
void mnist_infer(const float input[28][28], float output[10])
{
    const float* in_flat = &input[0][0];   // float[1*28*28]，C_in=1，CHW≡HW

    // ──────────────────────────────────────────────────────────
    //  Block 1: Conv2d(1→8) → BN → ReLU → MaxPool(2×2)
    //  输入: [1][28][28]   中间: [8][28][28]   输出: [8][14][14]
    // ──────────────────────────────────────────────────────────
    nn_conv2d_same(
        in_flat, act_a,
        28, 28, /*C_in=*/1, /*C_out=*/8,
        (const int8_t*)CONV1_W,
        CONV1_W_SCALE, CONV1_W_ZP
    );
    // act_a: [8][28][28] = 6272 元素
    nn_batchnorm(act_a, 8 * 28 * 28, /*C=*/8, CONV1_BN_SCALE, CONV1_BN_BIAS);
    nn_relu(act_a, 8 * 28 * 28);
    // Dropout2d: 推理时忽略
    nn_maxpool2d(act_a, act_b, /*H=*/28, /*W=*/28, /*C=*/8);
    // act_b: [8][14][14] = 1568 元素

    // ──────────────────────────────────────────────────────────
    //  Block 2: Conv2d(8→16) → BN → ReLU → MaxPool(2×2)
    //  输入: [8][14][14]   中间: [16][14][14]   输出: [16][7][7]
    // ──────────────────────────────────────────────────────────
    nn_conv2d_same(
        act_b, act_a,                       // act_a: [16][14][14] = 3136 ≤ 6272 ✓
        14, 14, /*C_in=*/8, /*C_out=*/16,
        (const int8_t*)CONV2_W,
        CONV2_W_SCALE, CONV2_W_ZP
    );
    nn_batchnorm(act_a, 16 * 14 * 14, /*C=*/16, CONV2_BN_SCALE, CONV2_BN_BIAS);
    nn_relu(act_a, 16 * 14 * 14);
    // Dropout2d: 推理时忽略
    nn_maxpool2d(act_a, act_b, /*H=*/14, /*W=*/14, /*C=*/16);
    // act_b: [16][7][7] = 784 元素 ≤ act_b 容量 1568 ✓

    // ──────────────────────────────────────────────────────────
    //  Flatten → Dense1(784→64, ReLU) → Dense2(64→10, Softmax)
    //
    //  Flatten 在 channels-first 下：[16][7][7] → [16*7*7=784]
    //  内存已经是连续的，act_b 直接作为 Dense1 输入 ✓
    //  Dense1 权重 [64][784] 中的 784 维对应 [C=16, H=7, W=7]，与 act_b 一致 ✓
    // ──────────────────────────────────────────────────────────
    nn_dense(
        act_b, dense_buf,
        /*in=*/784, /*out=*/64,
        (const int8_t*)DENSE1_W,
        DENSE1_W_SCALE, DENSE1_W_ZP,
        DENSE1_BIAS
    );
    nn_relu(dense_buf, 64);
    // Dropout: 推理时忽略

    nn_dense(
        dense_buf, output,
        /*in=*/64, /*out=*/10,
        (const int8_t*)DENSE2_W,
        DENSE2_W_SCALE, DENSE2_W_ZP,
        DENSE2_BIAS
    );
    nn_softmax(output, 10);
}

// ═══════════════════════════════════════════════════════════════
//  mnist_argmax
// ═══════════════════════════════════════════════════════════════
int mnist_argmax(const float probs[10])
{
    int   best = 0;
    float best_val = probs[0];
    for (int i = 1; i < 10; i++) {
        if (probs[i] > best_val) { best_val = probs[i]; best = i; }
    }
    return best;
}
