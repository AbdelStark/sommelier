#!/usr/bin/env python3
"""Generate fig_results: base vs adapter across five metrics, EN (v1 run) and FR (v2 run) panels.

Sources:
  docs/results/reference-run.md (run nemotron-8b-full-3, n=1000, EN slice)
  docs/results/french-run.md   (run nemotron-8b-fr-full-4, n=879, FR slice)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
})

BASE_COLOR = "#B0BEC5"   # neutral gray, recedes
V1_COLOR = "#0072B2"     # Okabe-Ito blue (v1 adapter, English-only training)
V2_COLOR = "#D55E00"     # Okabe-Ito vermillion (v2 adapter, mixed en+fr training)

METRICS = ["Valid JSON", "Func. name\naccuracy", "Arg. exact\nmatch",
           "Argument\nF1", "Full-call\nexact match"]

# EN slice of the v1 reference run nemotron-8b-full-3 (n=1000)
EN_BASE = [0.9160, 0.9110, 0.7070, 0.7569, 0.7050]
EN_ADPT = [1.0000, 0.9960, 0.8760, 0.9291, 0.8740]

# FR slice of the v2 French run nemotron-8b-fr-full-4 (n=879)
FR_BASE = [0.9044, 0.8976, 0.6655, 0.7091, 0.6633]
FR_ADPT = [0.9954, 0.9898, 0.8760, 0.9208, 0.8726]


def panel(ax, base, adpt, adpt_color, title):
    x = np.arange(len(METRICS))
    w = 0.36
    b1 = ax.bar(x - w / 2, base, w, color=BASE_COLOR,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, adpt, w, color=adpt_color,
                edgecolor="white", linewidth=0.5)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.010,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=6.3, color="#444", rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=7.5)
    ax.set_ylim(0.60, 1.049)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
panel(axes[0], EN_BASE, EN_ADPT, V1_COLOR, "(a) English, n=1,000")
panel(axes[1], FR_BASE, FR_ADPT, V2_COLOR, "(b) French, n=879")
axes[0].set_ylabel("Score")

handles = [
    Patch(color=BASE_COLOR, label="Base model"),
    Patch(color=V1_COLOR, label="v1 adapter (en-only training)"),
    Patch(color=V2_COLOR, label="v2 adapter (en+fr training)"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.04), fontsize=8, handlelength=1.4)
fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.92))
fig.savefig("fig_results.pdf")
fig.savefig("fig_results.png", dpi=300)
print("wrote fig_results.pdf/.png")
