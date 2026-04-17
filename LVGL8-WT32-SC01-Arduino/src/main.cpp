/**
 * ╔══════════════════════════════════════════════════════════╗
 * ║      MNIST Digit Recognition Demo  —  UI Layer          ║
 * ║  Platform : WT32-SC01  (ESP32 + 480×320 cap-touch)      ║
 * ║  Library  : LovyanGFX v1                                ║
 * ║                                                          ║
 * ║  Layout                                                  ║
 * ║  ┌─ Title bar (h=36) ────────────────────────────────┐  ║
 * ║  │ Left card 242×242 (28×28 grid, 8px/cell)          │  ║
 * ║  │ Right panel 220×272  (big digit + bar chart)       │  ║
 * ║  │ CLEAR button 242×26 below canvas                   │  ║
 * ║  └────────────────────────────────────────────────────┘  ║
 * ╚══════════════════════════════════════════════════════════╝
 */

#define LGFX_AUTODETECT
#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include "nn_ops.h"      // MNIST CNN 推理算子库

static LGFX lcd;

// ═══════════════════════════════════════════════════════════
//  颜色工具  (RGB888 → RGB565)
// ═══════════════════════════════════════════════════════════
static inline uint16_t C(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint16_t)(r >> 3) << 11) | ((uint16_t)(g >> 2) << 5) | (b >> 3);
}

// ── 调色板 ──────────────────────────────────────────────────
const uint16_t COL_BG        = C(  8,  10,  22);  // 深夜背景
const uint16_t COL_CARD      = C( 14,  15,  38);  // 卡片底色
const uint16_t COL_BORDER    = C( 45,  70, 170);  // 卡片边框蓝
const uint16_t COL_BORDER2   = C( 30,  45, 120);  // 内层边框
const uint16_t COL_GRID      = C( 28,  30,  70);  // 网格线
const uint16_t COL_CELL_BG   = C(  8,   9,  22);  // 格子底色
const uint16_t COL_ACCENT    = C( 78, 195, 252);  // 科技蓝高亮
const uint16_t COL_ACCENT2   = C(100, 220, 200);  // 青绿辅色
const uint16_t COL_DIM       = C( 70,  80, 150);  // 暗蓝次要文字
const uint16_t COL_SHADOW    = C(  2,   3,  10);  // 阴影
const uint16_t COL_CLR_BG    = C(140,  20,  30);  // 清除按钮底色
const uint16_t COL_CLR_BDR   = C(210,  55,  65);  // 清除按钮边框
const uint16_t COL_CLR_ACT   = C(200,  40,  50);  // 清除按钮按下色
const uint16_t COL_BAR_FILL  = C( 35,  65, 160);  // 普通柱状图填充

// ═══════════════════════════════════════════════════════════
//  布局常量  (所有尺寸均为像素，屏幕 480×320 横屏)
// ═══════════════════════════════════════════════════════════
#define SCREEN_W   480
#define SCREEN_H   320

// ── 标题栏 ──
#define TITLE_H     36

// ── 左侧画布卡片 ──
#define MARGIN       6
#define CELL_SZ      8      // 每个 MNIST 像素 = 8×8 屏幕像素
#define MNIST_W     28
#define MNIST_H     28
#define CANVAS_PX  (MNIST_W * CELL_SZ)      // 224 px
#define CARD_PAD     9      // 卡片内边距
#define CARD_L_X     MARGIN
#define CARD_L_Y    (TITLE_H + MARGIN)      // 42
#define CARD_L_W    (CANVAS_PX + CARD_PAD * 2)   // 242
#define CARD_L_H    (CANVAS_PX + CARD_PAD * 2)   // 242
#define CVS_X       (CARD_L_X + CARD_PAD)        // 15
#define CVS_Y       (CARD_L_Y + CARD_PAD)        // 51

// ── 清除按钮 ──
#define BTN_H        26
#define BTN_Y       (CARD_L_Y + CARD_L_H + 4)   // 288
#define BTN_X        CARD_L_X
#define BTN_W        CARD_L_W

// ── 右侧结果面板 ──
#define PANEL_X     (CARD_L_X + CARD_L_W + MARGIN)  // 254
#define PANEL_Y      CARD_L_Y                         // 42
#define PANEL_W     (SCREEN_W - PANEL_X - MARGIN)    // 220
#define PANEL_H     (SCREEN_H - PANEL_Y - MARGIN)    // 272

// ── 面板子区域 ──
#define RES_TITLE_H  20
#define RES_BIG_Y   (PANEL_Y + RES_TITLE_H + 2)     // 64
#define RES_BIG_H    70
#define RES_CONF_Y  (RES_BIG_Y + RES_BIG_H + 4)     // 138
#define RES_CONF_H   14
#define RES_BAR_Y   (RES_CONF_Y + RES_CONF_H + 2)   // 154
// 柱状图区域高度 = 面板底 - 柱起点 - 底边距
#define RES_BAR_AREA (PANEL_Y + PANEL_H - RES_BAR_Y - 4)  // 156
// 每条高度：10 条 + 9 条间距(2px)
#define BAR_H       ((RES_BAR_AREA - 9 * 2) / 10)   // 13

// ═══════════════════════════════════════════════════════════
//  全局状态
// ═══════════════════════════════════════════════════════════
uint8_t mnist_buf[MNIST_H][MNIST_W];  // 0~255 灰度值
float   infer_res[10] = {0};          // 推理置信度（后续填入）
int     predicted     = -1;           // 预测数字，-1 = 未推理

// ═══════════════════════════════════════════════════════════
//  绘制函数
// ═══════════════════════════════════════════════════════════

// ── 标题栏（渐变 + 图标 + 标签）──────────────────────────
void drawTitleBar() {
    // 逐行渐变：深蓝 → 稍亮海蓝
    for (int y = 0; y < TITLE_H; y++) {
        float t = (float)y / (TITLE_H - 1);
        uint16_t c = C(
            (uint8_t)( 5 + t * 18),
            (uint8_t)( 8 + t * 55),
            (uint8_t)(25 + t * 75)
        );
        lcd.drawFastHLine(0, y, SCREEN_W, c);
    }
    // 底部高亮线
    lcd.drawFastHLine(0, TITLE_H - 1, SCREEN_W, COL_ACCENT);

    // 左侧同心圆图标
    lcd.fillCircle(20, TITLE_H / 2, 10, COL_ACCENT);
    lcd.fillCircle(20, TITLE_H / 2,  6, C(8, 15, 45));
    lcd.fillCircle(20, TITLE_H / 2,  3, COL_ACCENT);

    // 标题文字
    lcd.setTextSize(2);
    lcd.setTextColor(TFT_WHITE);
    lcd.setCursor(38, 9);
    lcd.print("MNIST Digit Recognition");

    // 右上角芯片标签
    const int tagX = SCREEN_W - 82, tagY = 7;
    lcd.fillRoundRect(tagX,     tagY,     74, 22, 4, C(18, 38, 100));
    lcd.drawRoundRect(tagX,     tagY,     74, 22, 4, COL_ACCENT);
    lcd.setTextSize(1);
    lcd.setTextColor(COL_ACCENT);
    lcd.setCursor(tagX + 9, tagY + 7);
    lcd.print("WT32-SC01");
}

// ── 左侧画布卡片（含 28×28 网格）──────────────────────────
void drawCanvasCard() {
    // 阴影
    lcd.fillRoundRect(CARD_L_X + 3, CARD_L_Y + 3, CARD_L_W, CARD_L_H, 8, COL_SHADOW);
    // 卡片本体
    lcd.fillRoundRect(CARD_L_X, CARD_L_Y, CARD_L_W, CARD_L_H, 8, COL_CARD);
    // 双层边框
    lcd.drawRoundRect(CARD_L_X,     CARD_L_Y,     CARD_L_W,     CARD_L_H,     8, COL_BORDER);
    lcd.drawRoundRect(CARD_L_X + 1, CARD_L_Y + 1, CARD_L_W - 2, CARD_L_H - 2, 7, COL_BORDER2);

    // 网格背景
    lcd.fillRect(CVS_X, CVS_Y, CANVAS_PX, CANVAS_PX, COL_CELL_BG);

    // 垂直网格线
    for (int i = 0; i <= MNIST_W; i++) {
        lcd.drawFastVLine(CVS_X + i * CELL_SZ, CVS_Y, CANVAS_PX, COL_GRID);
    }
    // 水平网格线
    for (int i = 0; i <= MNIST_H; i++) {
        lcd.drawFastHLine(CVS_X, CVS_Y + i * CELL_SZ, CANVAS_PX, COL_GRID);
    }

    // 卡片左上"DRAW"标签
    lcd.setTextSize(1);
    lcd.setTextColor(COL_DIM);
    lcd.setCursor(CARD_L_X + 6, CARD_L_Y + 4);
    lcd.print("DRAW");

    // 右下尺寸标注
    lcd.setCursor(CARD_L_X + CARD_L_W - 37, CARD_L_Y + CARD_L_H - 10);
    lcd.print("28x28");
}

// ── 清除按钮 ────────────────────────────────────────────────
void drawClearBtn(bool pressed = false) {
    uint16_t bg  = pressed ? COL_CLR_ACT : COL_CLR_BG;
    uint16_t txt = pressed ? TFT_WHITE   : C(255, 160, 165);

    lcd.fillRoundRect(BTN_X + 2, BTN_Y + 2, BTN_W, BTN_H, 6, COL_SHADOW);
    lcd.fillRoundRect(BTN_X,     BTN_Y,     BTN_W, BTN_H, 6, bg);
    lcd.drawRoundRect(BTN_X,     BTN_Y,     BTN_W, BTN_H, 6, COL_CLR_BDR);

    // 按钮文字："CLEAR" = 5 chars × 12px = 60px，居中
    lcd.setTextSize(2);
    lcd.setTextColor(txt);
    lcd.setCursor(BTN_X + (BTN_W - 60) / 2, BTN_Y + 5);
    lcd.print("CLEAR");
}

// ── 右侧结果面板 ────────────────────────────────────────────
void drawResultPanel() {
    // 阴影
    lcd.fillRoundRect(PANEL_X + 3, PANEL_Y + 3, PANEL_W, PANEL_H, 8, COL_SHADOW);
    // 面板本体
    lcd.fillRoundRect(PANEL_X, PANEL_Y, PANEL_W, PANEL_H, 8, COL_CARD);
    // 双层边框
    lcd.drawRoundRect(PANEL_X,     PANEL_Y,     PANEL_W,     PANEL_H,     8, COL_BORDER);
    lcd.drawRoundRect(PANEL_X + 1, PANEL_Y + 1, PANEL_W - 2, PANEL_H - 2, 7, COL_BORDER2);

    // ── "RESULT" 标题 ──
    // "RESULT" = 6 chars × 6px = 36px，居中
    lcd.setTextSize(1);
    lcd.setTextColor(COL_ACCENT);
    lcd.setCursor(PANEL_X + (PANEL_W - 36) / 2, PANEL_Y + 6);
    lcd.print("RESULT");
    lcd.drawFastHLine(PANEL_X + 12, PANEL_Y + RES_TITLE_H - 2,
                      PANEL_W - 24, C(30, 50, 130));

    // ── 大数字显示框 ──
    lcd.fillRoundRect(PANEL_X + 10, RES_BIG_Y, PANEL_W - 20, RES_BIG_H,
                      6, COL_CELL_BG);
    lcd.drawRoundRect(PANEL_X + 10, RES_BIG_Y, PANEL_W - 20, RES_BIG_H,
                      6, C(40, 60, 150));

    if (predicted >= 0) {
        // 预测到数字：size 6 → 36×48px，垂直居中 (70-48)/2=11
        lcd.setTextSize(6);
        lcd.setTextColor(COL_ACCENT);
        lcd.setCursor(PANEL_X + (PANEL_W - 36) / 2, RES_BIG_Y + 11);
        lcd.printf("%d", predicted);
    } else {
        // 未推理："?"，size 5 → 30×40px，垂直居中 (70-40)/2=15
        lcd.setTextSize(5);
        lcd.setTextColor(COL_DIM);
        lcd.setCursor(PANEL_X + (PANEL_W - 30) / 2, RES_BIG_Y + 15);
        lcd.print("?");
    }

    // ── 置信度文字行 ──
    lcd.setTextSize(1);
    lcd.setTextColor(COL_DIM);
    lcd.setCursor(PANEL_X + 12, RES_CONF_Y);
    lcd.print("Confidence");
    if (predicted >= 0) {
        lcd.setTextColor(COL_ACCENT2);
        lcd.setCursor(PANEL_X + PANEL_W - 54, RES_CONF_Y);
        lcd.printf("%5.1f%%", infer_res[predicted] * 100.0f);
    }

    // ── 柱状图 0~9 ──
    const int bx = PANEL_X + 22;       // 条形起始 X（留出标签空间）
    const int bw = PANEL_W - 34;       // 条形最大宽度

    for (int i = 0; i < 10; i++) {
        const int   by    = RES_BAR_Y + i * (BAR_H + 2);
        const bool  isTop = (i == predicted);

        // 数字标签
        lcd.setTextSize(1);
        lcd.setTextColor(isTop ? COL_ACCENT : COL_DIM);
        lcd.setCursor(PANEL_X + 12, by + (BAR_H > 8 ? (BAR_H - 8) / 2 : 0));
        lcd.printf("%d", i);

        // 背景槽
        lcd.fillRoundRect(bx, by, bw, BAR_H, 2, COL_CELL_BG);

        // 填充条
        const int fill = (int)(infer_res[i] * bw);
        if (fill > 0) {
            lcd.fillRoundRect(bx, by, fill, BAR_H, 2,
                              isTop ? COL_ACCENT : COL_BAR_FILL);
        }
    }
}

// ── 完整界面初始化 ───────────────────────────────────────────
void drawUI() {
    lcd.fillScreen(COL_BG);
    drawTitleBar();
    drawCanvasCard();
    drawClearBtn(false);
    drawResultPanel();
}

// ═══════════════════════════════════════════════════════════
//  画布操作
// ═══════════════════════════════════════════════════════════

// 更新单格显示（纯二值：0 或 255）
void paintCell(int col, int row) {
    if (col < 0 || col >= MNIST_W || row < 0 || row >= MNIST_H) return;
    if (mnist_buf[row][col] == 255) return;   // 已经是白色，跳过重绘

    mnist_buf[row][col] = 255;

    // 纯白笔迹（带淡蓝冷调，与深色背景对比鲜明）
    lcd.fillRect(
        CVS_X + col * CELL_SZ + 1,
        CVS_Y + row * CELL_SZ + 1,
        CELL_SZ - 1, CELL_SZ - 1,
        TFT_WHITE
    );
}

// 实心方形笔刷（3×3，全部置为 255）
// 手指触点大于单格，3×3 保证每次至少点亮 1 个像素
void drawBrush(int col, int row) {
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            paintCell(col + dc, row + dr);
        }
    }
}

// 清空画布数据并重绘格子区域
void clearCanvas() {
    memset(mnist_buf, 0, sizeof(mnist_buf));

    // 重填背景
    lcd.fillRect(CVS_X + 1, CVS_Y + 1,
                 CANVAS_PX - 1, CANVAS_PX - 1, COL_CELL_BG);

    // 重绘内部网格线（外边框由卡片保持，只需内部线）
    for (int i = 1; i < MNIST_W; i++) {
        lcd.drawFastVLine(CVS_X + i * CELL_SZ, CVS_Y + 1,
                          CANVAS_PX - 1, COL_GRID);
    }
    for (int i = 1; i < MNIST_H; i++) {
        lcd.drawFastHLine(CVS_X + 1, CVS_Y + i * CELL_SZ,
                          CANVAS_PX - 1, COL_GRID);
    }
}

// 局部刷新结果面板（避免整屏重绘闪烁）
void refreshResultPanel() {
    // 刷新大数字框内容
    lcd.fillRoundRect(PANEL_X + 11, RES_BIG_Y + 1,
                      PANEL_W - 22, RES_BIG_H - 2, 5, COL_CELL_BG);
    if (predicted >= 0) {
        lcd.setTextSize(6);
        lcd.setTextColor(COL_ACCENT);
        lcd.setCursor(PANEL_X + (PANEL_W - 36) / 2, RES_BIG_Y + 11);
        lcd.printf("%d", predicted);
    } else {
        lcd.setTextSize(5);
        lcd.setTextColor(COL_DIM);
        lcd.setCursor(PANEL_X + (PANEL_W - 30) / 2, RES_BIG_Y + 15);
        lcd.print("?");
    }

    // 刷新置信度文字行
    lcd.fillRect(PANEL_X + 12, RES_CONF_Y, PANEL_W - 14, RES_CONF_H, COL_CARD);
    lcd.setTextSize(1);
    lcd.setTextColor(COL_DIM);
    lcd.setCursor(PANEL_X + 12, RES_CONF_Y);
    lcd.print("Confidence");
    if (predicted >= 0) {
        lcd.setTextColor(COL_ACCENT2);
        lcd.setCursor(PANEL_X + PANEL_W - 54, RES_CONF_Y);
        lcd.printf("%5.1f%%", infer_res[predicted] * 100.0f);
    }

    // 刷新柱状图
    const int bx = PANEL_X + 22;
    const int bw = PANEL_W - 34;
    for (int i = 0; i < 10; i++) {
        const int  by    = RES_BAR_Y + i * (BAR_H + 2);
        const bool isTop = (i == predicted);

        lcd.setTextSize(1);
        lcd.setTextColor(isTop ? COL_ACCENT : COL_DIM);
        lcd.setCursor(PANEL_X + 12, by + (BAR_H > 8 ? (BAR_H - 8) / 2 : 0));
        lcd.printf("%d", i);

        lcd.fillRoundRect(bx, by, bw, BAR_H, 2, COL_CELL_BG);
        const int fill = (int)(infer_res[i] * bw);
        if (fill > 0) {
            lcd.fillRoundRect(bx, by, fill, BAR_H, 2,
                              isTop ? COL_ACCENT : COL_BAR_FILL);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  触摸处理
// ═══════════════════════════════════════════════════════════
//  推理触发策略（与 Python mnist_gui.py 保持一致）：
//    _on_drag    → 拖动中每帧实时推理（节流：每 INFER_DRAG_MS 最多一次）
//    _on_release → 抬手时再推理一次（保证最终结果准确）
//    CLEAR 按钮  → 手动清空，重置面板（无自动清空）
// ═══════════════════════════════════════════════════════════
#define INFER_DRAG_MS  800u    // 拖动推理节流间隔（ms），书写中途只偶尔触发一次

// 前向声明（定义在下方，handleTouch 内需要调用）
void runInference();

static bool     was_touching         = false;
static bool     clear_pressed        = false;
static bool     canvas_touched       = false;  // 本次按下是否触碰了画布
static bool     canvas_has_ink       = false;  // 画布上是否有笔迹
static uint32_t last_drag_infer_ms   = 0;      // 上次拖动推理时间戳

void handleTouch(int32_t tx, int32_t ty) {
    // ─ 画布区域 ─
    if (tx >= CVS_X && tx < CVS_X + CANVAS_PX &&
        ty >= CVS_Y && ty < CVS_Y + CANVAS_PX) {

        int col = (tx - CVS_X) / CELL_SZ;
        int row = (ty - CVS_Y) / CELL_SZ;
        drawBrush(col, row);

        canvas_touched = true;
        canvas_has_ink = true;

        // 拖动中实时推理（节流，对应 Python _on_drag → _infer()）
        uint32_t now = millis();
        if (now - last_drag_infer_ms >= INFER_DRAG_MS) {
            last_drag_infer_ms = now;
            runInference();
        }
        return;
    }

    // ─ 清除按钮 ─
    if (tx >= BTN_X && tx < BTN_X + BTN_W &&
        ty >= BTN_Y && ty < BTN_Y + BTN_H) {
        if (!clear_pressed) {
            clear_pressed  = true;
            drawClearBtn(true);
        }
        return;
    }
}

// 将二值 mnist_buf[28][28] (0 或 255) 转为 float[28][28] (0.0 或 1.0)
static float infer_input[28][28];

void runInference() {
    // 1. 二值转浮点并应用 MNIST 归一化
    //    Python 参考: T.Normalize((0.1307,), (0.3081,))
    //    0   → (0.0 - 0.1307) / 0.3081 = -0.4242f  (黑色背景)
    //    255 → (1.0 - 0.1307) / 0.3081 =  2.8215f  (白色笔迹)
    const float VAL_WHITE = (1.0f - 0.1307f) / 0.3081f;   //  2.8215f
    const float VAL_BLACK = (0.0f - 0.1307f) / 0.3081f;   // -0.4242f
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            infer_input[r][c] = (mnist_buf[r][c] == 255) ? VAL_WHITE : VAL_BLACK;
        }
    }

    // 2. 推理（调用 nn_ops.cpp 中的完整流水线）
    mnist_infer(infer_input, infer_res);

    // 3. 取最大置信度类别
    predicted = mnist_argmax(infer_res);

    // 4. 局部刷新结果面板
    refreshResultPanel();

    // 5. 打印日志
    Serial.printf("[Infer] predicted=%d  conf=%.1f%%\n",
                  predicted, infer_res[predicted] * 100.0f);
}

void onTouchRelease() {
    if (clear_pressed) {
        // 手动 CLEAR：重置所有状态（对应 Python _clear()）
        clearCanvas();
        predicted      = -1;
        canvas_has_ink = false;
        memset(infer_res, 0, sizeof(infer_res));
        refreshResultPanel();
        drawClearBtn(false);
        clear_pressed = false;
        return;
    }

    // 抬手时推理（对应 Python _on_release → _infer()）
    if (canvas_touched && canvas_has_ink) {
        runInference();
    }
    canvas_touched = false;
}

// ═══════════════════════════════════════════════════════════
//  Arduino 入口
// ═══════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    lcd.init();

    // 确保横屏
    if (lcd.width() < lcd.height()) {
        lcd.setRotation(lcd.getRotation() ^ 1);
    }
    lcd.setBrightness(220);

    memset(mnist_buf, 0, sizeof(mnist_buf));
    memset(infer_res, 0, sizeof(infer_res));

    drawUI();

    Serial.printf("[MNIST] UI ready. Screen: %dx%d\n",
                  lcd.width(), lcd.height());
    Serial.printf("  Canvas origin : (%d, %d)  size: %dx%d\n",
                  CVS_X, CVS_Y, CANVAS_PX, CANVAS_PX);
    Serial.printf("  Cell size     : %d px / MNIST pixel\n", CELL_SZ);
    Serial.printf("  Panel         : x=%d y=%d w=%d h=%d\n",
                  PANEL_X, PANEL_Y, PANEL_W, PANEL_H);
    Serial.printf("  BAR_H         : %d px  (10 bars + 9 gaps in %d px)\n",
                  BAR_H, RES_BAR_AREA);
}

static int32_t touch_x, touch_y;

void loop() {
    bool touching = lcd.getTouch(&touch_x, &touch_y);

    if (touching) {
        handleTouch(touch_x, touch_y);
        was_touching = true;
    } else {
        if (was_touching) {
            onTouchRelease();   // CLEAR 处理 + 画布抬手推理
            was_touching = false;
        }
    }
}
