"""
mnist_gui.py — 手写数字识别 GUI
仿 WT32-SC01 480×320 触摸屏布局

修复：条形框延伸至面板右边界，百分比标签叠放在条形内部右侧

布局（坐标以窗口左上角为原点）：
  ┌── 标题栏 h=40 ──────────────────────────────────────────┐
  │ ●  MNIST Digit Recognition               [WT32-SC01]   │
  ├──────────────────────────┬─────────────────────────────┤
  │  DRAW 卡片               │  RESULT 面板                 │
  │  x=8  w=248  h=248       │  x=268  w=364  h=320         │
  │  画布 224×224 (28×8px)   │  大数字 + Confidence + 概率条│
  ├──────────────────────────┤                              │
  │  CLEAR 按钮 y=310 h=36   │                              │
  └──────────────────────────┴─────────────────────────────┘
"""

import os, math
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
import tkinter as tk

MODEL_PATH = "mnist_model.pth"

# ──────────────────────────────────────────────
# 布局常数
# ──────────────────────────────────────────────
WIN_W, WIN_H = 640, 360
TITLE_H      = 40

CELL   = 8
GRID_N = 28
CAN_PX = CELL * GRID_N  # 224

# 左卡片
LX, LY = 8, TITLE_H + 8   # (8, 48)
LW, LH = 248, 248

# 画布在卡片内的偏移
C_OFF_X = (LW - CAN_PX) // 2  # 12
C_OFF_Y = 20

# CLEAR 按钮
CLX, CLY = 8,   310
CLW, CLH = 248,  36

# 右面板
RX, RY = 268, TITLE_H
RW     = WIN_W - RX - 8   # 364
RH     = WIN_H - TITLE_H  # 320

BRUSH_INIT = 10
BLUR_R     = 0.8

# ──────────────────────────────────────────────
# 颜色
# ──────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
ACC     = "#58a6ff"
TEXT    = "#e6edf3"
SUB     = "#8b949e"
GRN     = "#3fb950"
YLW     = "#d29922"
RED_C   = "#f85149"
BAR_BG  = "#21262d"
GRID_LN = "#1c2030"   # 网格线颜色

# ──────────────────────────────────────────────
# 模型定义（与 train_mcu.py 一致）
# ──────────────────────────────────────────────
class MCU_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Dropout2d(0.10), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout2d(0.15), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*7*7, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到 {MODEL_PATH}，请先运行 train_mcu.py")

_model = MCU_CNN()
_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_model.eval()
print(f"✓ 模型已加载: {MODEL_PATH}")

_pre = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

def infer(pil28):
    """PIL 灰度 28×28 → (预测数字, softmax 概率 ndarray)"""
    with torch.no_grad():
        p = torch.softmax(_model(_pre(pil28).unsqueeze(0)), dim=1)
    probs = p.squeeze().numpy()
    return int(np.argmax(probs)), probs


# ──────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Recognition")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._pil   = Image.new("L", (CAN_PX, CAN_PX), 0)
        self._pdraw = ImageDraw.Draw(self._pil)
        self._last  = None
        self._brush = BRUSH_INIT

        self._build_title()
        self._build_left()
        self._build_right()
        self._reset()

    # ── 标题栏 ────────────────────────────────
    def _build_title(self):
        bar = tk.Frame(self, bg=CARD, height=TITLE_H)
        bar.place(x=0, y=0, width=WIN_W, height=TITLE_H)

        tk.Label(bar, text="●  MNIST Digit Recognition",
                 bg=CARD, fg=ACC,
                 font=("Mono", 200, "bold")).place(x=12, y=0, height=TITLE_H)
        tk.Label(bar, text="[WT32-SC01]",
                 bg=CARD, fg=SUB,
                 font=("Mono", 200)).place(x=WIN_W-130, y=0, height=TITLE_H)
        # 底部分隔线
        tk.Frame(self, bg=BORDER).place(x=0, y=TITLE_H-1, width=WIN_W, height=1)

    # ── 左侧：DRAW 卡片 + CLEAR 按钮 ─────────
    def _build_left(self):
        card = tk.Frame(self, bg=CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        card.place(x=LX, y=LY, width=LW, height=LH)

        tk.Label(card, text="DRAW", bg=CARD, fg=SUB,
                 font=("Mono", 200)).place(x=0, y=2, width=LW, height=16)

        # 画布（黑底，供鼠标绘图）
        self.cv = tk.Canvas(card, width=CAN_PX, height=CAN_PX,
                            bg="black", cursor="crosshair",
                            highlightthickness=0)
        self.cv.place(x=C_OFF_X, y=C_OFF_Y)
        self._draw_grid()

        self.cv.bind("<ButtonPress-1>",   self._on_press)
        self.cv.bind("<B1-Motion>",       self._on_drag)
        self.cv.bind("<ButtonRelease-1>", self._on_release)

        # CLEAR 按钮（绝对坐标，在卡片下方）
        tk.Button(self, text="CLEAR", command=self._clear,
                  bg=RED_C, fg="white", relief=tk.FLAT,
                  font=("Mono", 200, "bold"),
                  activebackground="#cc2a2a",
                  cursor="hand2").place(x=CLX, y=CLY, width=CLW, height=CLH)

    def _draw_grid(self):
        """28×28 网格参考线"""
        for i in range(1, GRID_N):
            v = i * CELL
            self.cv.create_line(v, 0, v, CAN_PX, fill=GRID_LN)
            self.cv.create_line(0, v, CAN_PX, v, fill=GRID_LN)

    # ── 右侧：RESULT 面板 ─────────────────────
    def _build_right(self):
        panel = tk.Frame(self, bg=CARD,
                         highlightthickness=1, highlightbackground=BORDER)
        panel.place(x=RX, y=RY, width=RW, height=RH)

        # "RESULT" 标签
        tk.Label(panel, text="RESULT", bg=CARD, fg=SUB,
                 font=("Mono", 200)).place(x=0, y=4, width=RW, height=18)

        # 大数字显示框
        DIG_Y, DIG_H = 24, 100
        dbox = tk.Frame(panel, bg=BG,
                        highlightthickness=1, highlightbackground=BORDER)
        dbox.place(x=6, y=DIG_Y, width=RW-12, height=DIG_H)

        self.lbl_digit = tk.Label(dbox, text="?", bg=BG, fg=GRN,
                                  font=("Mono", 800, "bold"))
        self.lbl_digit.place(relx=0.5, rely=0.5, anchor="center")

        # Confidence 行
        CONF_Y = DIG_Y + DIG_H + 8
        self.lbl_conf = tk.Label(panel,
                                 text="Confidence        —",
                                 bg=CARD, fg=TEXT,
                                 font=("Mono", 200), anchor="w")
        self.lbl_conf.place(x=6, y=CONF_Y, width=RW-12, height=24)

        # ── 概率条（0~9）────────────────────────────────────────────────
        # 修复：BAR_W 撑满至右边界，不再为百分比标签预留右侧空间
        # 百分比文字改为叠放在条形背景框内部右侧
        BAR_Y0    = CONF_Y + 28
        BAR_ROW_H = (RH - BAR_Y0 - 6) // 10   # ≈ 18~20

        NUM_W = 24   # 左侧数字标签宽
        GAP   = 6    # 数字标签与条形之间的间距
        PAD_R = 6    # 条形距面板右边缘的留白

        # ★ 关键修复：BAR_W 延伸到右边界，不再减去 PCT_W
        BAR_W = RW - 6 - NUM_W - GAP - PAD_R   # = 364-6-24-6-6 = 322

        self._fills = []
        self._plbls = []
        self._bw    = BAR_W

        for d in range(10):
            y = BAR_Y0 + d * BAR_ROW_H

            # 数字编号（左侧）
            tk.Label(panel, text=str(d), bg=CARD, fg=TEXT,
                     font=("Mono", 150, "bold"),
                     anchor="center").place(x=6, y=y,
                                            width=NUM_W,
                                            height=BAR_ROW_H - 2)

            # 条形背景框（延伸至右边界）
            bg_f = tk.Frame(panel, bg=BAR_BG)
            bg_f.place(x=6 + NUM_W + GAP, y=y + 3,
                       width=BAR_W, height=BAR_ROW_H - 6)

            # 条形填充（宽度由推理结果动态设置）
            fill = tk.Frame(bg_f, bg=ACC)
            fill.place(x=0, y=0, relheight=1, width=0)
            self._fills.append((bg_f, fill))

            # ★ 百分比标签：叠放在条形背景框内部，贴右对齐
            pct = tk.Label(bg_f, text=" 0%", bg=BAR_BG, fg=SUB,
                           font=("Mono", 120), anchor="e")
            pct.place(relx=1.0, rely=0.5, anchor="e", x=-3)
            self._plbls.append(pct)

    # ── 绘图 ──────────────────────────────────
    def _dot(self, x, y):
        r = self._brush
        self.cv.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self._pdraw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def _line(self, x0, y0, x1, y1):
        dist = math.hypot(x1-x0, y1-y0)
        n    = max(1, int(dist / max(1, self._brush * 0.4)))
        for i in range(n+1):
            t = i / n
            self._dot(x0+(x1-x0)*t, y0+(y1-y0)*t)

    def _on_press(self, e):
        self._last = (e.x, e.y)
        self._dot(e.x, e.y)

    def _on_drag(self, e):
        if self._last:
            self._line(*self._last, e.x, e.y)
        self._last = (e.x, e.y)
        self._infer()

    def _on_release(self, e):
        self._last = None
        self._infer()

    def _clear(self):
        self.cv.delete("all")
        self._draw_grid()
        self._pil   = Image.new("L", (CAN_PX, CAN_PX), 0)
        self._pdraw = ImageDraw.Draw(self._pil)
        self._reset()

    # ── 推理 & 更新 ───────────────────────────
    def _infer(self):
        small = self._pil.resize((28, 28), Image.LANCZOS)
        small = small.filter(ImageFilter.GaussianBlur(BLUR_R))

        if np.array(small).max() == 0:
            self._reset()
            return

        pred, probs = infer(small)
        conf = float(probs[pred])

        # 置信度颜色
        color = GRN if conf >= 0.90 else (YLW if conf >= 0.60 else RED_C)

        self.lbl_digit.config(text=str(pred), fg=color)
        self.lbl_conf.config(
            text=f"Confidence    {conf*100:5.1f}%",
            fg=color
        )

        for d, ((bg_f, fill), pct) in enumerate(zip(self._fills, self._plbls)):
            p  = float(probs[d])
            w  = int(p * self._bw)
            c  = color if d == pred else ACC

            fill.place(x=0, y=0, relheight=1, width=max(w, 0))
            fill.config(bg=c)

            # 百分比标签：条形足够宽时将背景色改为与填充色一致，使文字浮于条形上
            pct_text = f"{p*100:3.0f}%"
            if d == pred:
                pct.config(text=pct_text, fg="white",
                           bg=c if w > 36 else BAR_BG)
            else:
                pct.config(text=pct_text, fg=SUB,
                           bg=c if w > 36 else BAR_BG)

    def _reset(self):
        self.lbl_digit.config(text="?", fg=GRN)
        self.lbl_conf.config(text="Confidence           —", fg=TEXT)
        for (bg_f, fill), pct in zip(self._fills, self._plbls):
            fill.place(x=0, y=0, relheight=1, width=0)
            fill.config(bg=ACC)
            pct.config(text=" 0%", fg=SUB, bg=BAR_BG)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()