import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})

FIG_W, FIG_H = 7.2, 4.1
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

INK="#1a1a1a"; RULE="#6b6b6b"
FILL_IN="#eef2f5"; FILL_CNN="#dfe8ef"; FILL_LSTM="#e8e3ef"; FILL_AUX="#f0ece2"; FILL_RF="#e2ece4"

def box(x, y, w, h, text, fill, fs=8, bold=False, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=0.9, edgecolor=RULE, facecolor=fill))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, color=INK, linespacing=1.35,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=8, linewidth=0.8, color=RULE,
        shrinkA=0, shrinkB=0))

def lane(y0, title, src, srcsub, cnn, lstm, aux, out):
    # lane label
    ax.text(1.5, y0+11.5, title, fontsize=8.5, fontweight="bold",
            color=INK, ha="left", va="center")
    box(1.5, y0, 19, 9.5, src+"\n"+srcsub, FILL_IN, fs=7.6)
    box(24, y0+5.4, 21, 6.6, cnn, FILL_CNN, fs=7.6)
    box(24, y0-2.4, 21, 6.6, lstm, FILL_LSTM, fs=7.6)
    box(49, y0+1.2, 14, 7.2, aux, FILL_AUX)
    # source -> both branches
    arrow(20.5, y0+4.75, 24, y0+8.7)
    arrow(20.5, y0+4.75, 24, y0+0.9)
    # branches -> concat
    arrow(45, y0+8.7, 49, y0+5.6)
    arrow(45, y0+0.9, 49, y0+4.0)
    return out

# --- lane A: social media ---
yA = 36
lane(yA, "Social media pipeline",
     "Ebola tweets", "8,395 padded index\nsequences (V = 5,000)",
     "Multi-kernel CNN\nk = 3, 4, 5 x 64\nGlobalMaxPool  ->  192",
     "Bidirectional LSTM\n2 x 32 units  ->  64",
     "Concatenate\n+ sentiment\nclass  (1)", None)

# --- lane B: clinical ---
yB = 11
lane(yB, "Clinical pipeline",
     "COVID-19 records", "5,644 scaled\nfeature vectors",
     "Conv1D 32, Conv1D 64\nGlobalMaxPool  ->  64",
     "Bidirectional LSTM\n2 x 32 units  ->  64",
     "Concatenate\n+ scaled raw\nfeatures", None)

# --- shared classifier column ---
box(68, yA+0.6, 15, 8.4, "StandardScaler\n\nRandom Forest\n50 trees", FILL_RF)
box(68, yB+0.6, 15, 8.4, "StandardScaler\n\nRandom Forest\n50 trees", FILL_RF)
arrow(63, yA+4.8, 68, yA+4.8)
arrow(63, yB+4.8, 68, yB+4.8)

box(87.5, yA+1.8, 11, 6, "Warning /\nnon-warning", "#ffffff")
box(87.5, yB+1.8, 11, 6, "Positive /\nnegative", "#ffffff")
arrow(83, yA+4.8, 87.5, yA+4.8)
arrow(83, yB+4.8, 87.5, yB+4.8)

# SHAP branch off the clinical RF
box(68, yB-9.5, 30.5, 6.6,
    "SHAP (TreeExplainer) over the fitted forest;\nattributions displayed on the raw clinical block",
    "#f4f4f0", fs=7.5)
arrow(75.5, yB+0.6, 75.5, yB-2.9)

# separation note
ax.plot([0.5, 99.5], [yA-4.5, yA-4.5], linewidth=0.6, color="#b8b8b8", linestyle=(0,(4,3)))
ax.text(50, yA-6.6,
        "Both branches are trained end-to-end with a sigmoid head, then used as fixed feature maps for the Random Forest.\n"
        "The two pipelines share this template but use disjoint data on different diseases and are never fused.",
        ha="center", va="center", fontsize=7.3, color="#4a4a4a", linespacing=1.5, style="italic")

plt.tight_layout(pad=0.2)
plt.savefig("results/figures/fig_architecture.png", dpi=400, bbox_inches="tight", facecolor="white")
print("written results/figures/fig_architecture.png")
