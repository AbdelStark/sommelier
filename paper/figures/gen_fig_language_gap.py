#!/usr/bin/env python3
"""Generate fig_language_gap: full-call exact match, English vs French, across base/v1/v2.

Source: docs/results/french-run.md (language gap table; en slice digest identical to v1 reference).
"""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
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
})

# model: (en, fr, color, left label dy, right label dy, gap text)
SERIES = [
    ("Base model", 0.7050, 0.6633, "#8C8C8C", 0, 0, "gap $-$4.2 pts"),
    ("v1 adapter (en-only training)", 0.8740, 0.8510, "#0072B2", 8, -4, "gap $-$2.3 pts"),
    ("v2 adapter (en+fr training)", 0.8700, 0.8726, "#D55E00", -8, 8, "gap $+$0.3 pts"),
]

fig, ax = plt.subplots(figsize=(4.4, 2.8))
xs = [0, 1]
for name, en, fr, color, dy_l, dy_r, gap in SERIES:
    ax.plot(xs, [en, fr], color=color, marker="o", markersize=5,
            linewidth=1.8, label=name, zorder=3)
    ax.annotate(f"{en:.4f}", (0, en), textcoords="offset points",
                xytext=(-8, dy_l), ha="right", va="center", fontsize=7.5, color=color)
    ax.annotate(f"{fr:.4f}  ({gap})", (1, fr), textcoords="offset points",
                xytext=(8, dy_r), ha="left", va="center", fontsize=7.5, color=color)

ax.set_xticks(xs)
ax.set_xticklabels(["English slice", "French slice"])
ax.set_xlim(-0.35, 1.95)
ax.set_ylim(0.62, 0.92)
ax.set_ylabel("Full-call exact match")
ax.legend(loc="lower left", handlelength=1.4)
fig.tight_layout()
fig.savefig("fig_language_gap.pdf")
fig.savefig("fig_language_gap.png", dpi=300)
print("wrote fig_language_gap.pdf/.png")
