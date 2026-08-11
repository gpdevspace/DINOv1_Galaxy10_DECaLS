"""Static architecture diagram: full DINO training-time pipeline,
annotated with where centering, sharpening, stop-gradient, and the EMA
teacher update each happen.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

BG = "#0c0c0c"
WHITE = "#f2f2f2"
GRAY = "#9a9a9a"
STUDENT = "#2166c4"
STUDENT_EDGE = "#7fb0ff"
TEACHER = "#c4681f"
TEACHER_EDGE = "#ffb877"
NEUTRAL = "#2a2a2a"
NEUTRAL_EDGE = "#cfcfcf"
CENTERING = "#7d3fae"
CENTERING_EDGE = "#d4a8ff"
SHARPENING = "#12897a"
SHARPENING_EDGE = "#7fe0d0"
LOSS = "#2f7d3a"
LOSS_EDGE = "#8fe89f"
STOP = "#a5281f"
STOP_EDGE = "#ff9188"

STUDENT_X = 5.2
TEACHER_X = 15.2
MID_X = 10.2
SIDE_W = 7.6
MID_W = 13.5


def box(ax, cx, cy, w, h, text, fc, ec, fontsize=9.3, fontweight="normal", tc=WHITE, lw=1.6):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=tc, fontweight=fontweight, linespacing=1.5, zorder=4)


def arrow(ax, xy_start, xy_end, color=WHITE, lw=1.7, ls="-", rad=0.0, z=2):
    ax.annotate(
        "", xy=xy_end, xytext=xy_start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                         connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3),
        zorder=z,
    )


def line(ax, xy_start, xy_end, color=WHITE, lw=1.7, ls="-", z=2):
    ax.annotate(
        "", xy=xy_end, xytext=xy_start,
        arrowprops=dict(arrowstyle="-", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=z,
    )


def label(ax, x, y, text, color=GRAY, fontsize=8.2, ha="center", style="normal", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
             linespacing=1.4, fontstyle=style, fontweight=weight, zorder=4)


def badge(ax, x, y, n):
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9.5,
             color="black", fontweight="bold", zorder=6,
             bbox=dict(boxstyle="circle,pad=0.28", facecolor=WHITE, edgecolor="black", linewidth=1.0))


def build():
    fig, ax = plt.subplots(figsize=(15, 21))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0, 27.2)
    ax.axis("off")

    ax.text(10.2, 26.6, "DINO — training-time architecture", ha="center",
             fontsize=17, color=WHITE, fontweight="bold")
    ax.text(10.2, 26.05, "Emerging Properties in Self-Supervised Vision Transformers (Caron et al., 2021)",
             ha="center", fontsize=9.5, color=GRAY, fontstyle="italic")
    ax.text(10.2, 25.62, "numbers = order of operations  ·  read top to bottom",
             ha="center", fontsize=8, color=GRAY, fontstyle="italic")

    # 1. input
    y = 24.55
    box(ax, MID_X, y, 4.2, 1.05, "Input image", NEUTRAL, NEUTRAL_EDGE, fontsize=10)
    badge(ax, MID_X - 2.1 + 0.28, y + 0.525 - 0.05, 1)

    # 2. multi-crop
    y2 = 22.65
    arrow(ax, (MID_X, y - 0.55), (MID_X, y2 + 0.65))
    box(ax, MID_X, y2, MID_W, 1.4,
        "Multi-Crop Augmentation\n2 global crops (224², 50–100% of image)  +  6 local crops (96², 5–50% of image)",
        NEUTRAL, NEUTRAL_EDGE, fontsize=9.5)
    badge(ax, MID_X - MID_W / 2 + 0.32, y2 + 0.7 - 0.08, 2)

    # branch labels
    y3 = 21.0
    arrow(ax, (MID_X - 2.2, y2 - 0.7), (STUDENT_X, y3 + 0.35), color=STUDENT_EDGE, rad=0.05)
    arrow(ax, (MID_X + 2.2, y2 - 0.7), (TEACHER_X, y3 + 0.35), color=TEACHER_EDGE, rad=-0.05)
    label(ax, STUDENT_X, y3 + 0.35, "sees ALL 8 crops\n(global + local)", color=STUDENT_EDGE, fontsize=8.4, weight="bold")
    label(ax, TEACHER_X, y3 + 0.35, "sees ONLY the\n2 global crops", color=TEACHER_EDGE, fontsize=8.4, weight="bold")

    # column headers
    yh = 20.15
    label(ax, STUDENT_X, yh, "STUDENT  gθs   (trained by backprop)", color=STUDENT_EDGE, fontsize=10.5, weight="bold")
    label(ax, TEACHER_X, yh, "TEACHER  gθt   (EMA of student, no gradient)", color=TEACHER_EDGE, fontsize=10.5, weight="bold")

    # 3. backbone
    y4 = 18.7
    box(ax, STUDENT_X, y4, SIDE_W, 1.5, "ViT-S/8 backbone\n(weights θs)", STUDENT, STUDENT_EDGE)
    box(ax, TEACHER_X, y4, SIDE_W, 1.5, "ViT-S/8 backbone\n(weights θt)", TEACHER, TEACHER_EDGE)
    arrow(ax, (STUDENT_X, y3 - 0.05), (STUDENT_X, y4 + 0.75), color=STUDENT_EDGE)
    arrow(ax, (TEACHER_X, y3 - 0.05), (TEACHER_X, y4 + 0.75), color=TEACHER_EDGE)
    label(ax, MID_X, y4 + 0.05, "identical\narchitecture,\nseparate\nweights", color=GRAY, fontsize=7.6)
    badge(ax, MID_X, y4 + 0.98, 3)

    # EMA feedback arrow (student weights -> teacher weights)
    yema = 17.35
    line(ax, (STUDENT_X + SIDE_W / 2 - 0.3, y4 - 0.75), (STUDENT_X + SIDE_W / 2 - 0.3, yema),
         color=GRAY, ls=(0, (5, 3)), lw=1.4)
    line(ax, (STUDENT_X + SIDE_W / 2 - 0.3, yema), (TEACHER_X - SIDE_W / 2 + 0.3, yema),
         color=GRAY, ls=(0, (5, 3)), lw=1.4)
    arrow(ax, (TEACHER_X - SIDE_W / 2 + 0.3, yema), (TEACHER_X - SIDE_W / 2 + 0.3, y4 - 0.75),
          color=GRAY, ls=(0, (5, 3)), rad=0.0, lw=1.4)
    label(ax, MID_X, yema + 0.32,
          "EMA teacher update (no gradient):   θt ← λ·θt + (1−λ)·θs      λ: 0.996 → 1 (cosine)",
          color=GRAY, fontsize=8.2)

    # 4. CLS token
    y5 = 16.5
    box(ax, STUDENT_X, y5, SIDE_W, 1.1, "[CLS] token → 384-d embedding", STUDENT, STUDENT_EDGE, fontsize=9)
    box(ax, TEACHER_X, y5, SIDE_W, 1.1, "[CLS] token → 384-d embedding", TEACHER, TEACHER_EDGE, fontsize=9)
    arrow(ax, (STUDENT_X, y4 - 0.75), (STUDENT_X, y5 + 0.55), color=STUDENT_EDGE)
    arrow(ax, (TEACHER_X, y4 - 0.75), (TEACHER_X, y5 + 0.55), color=TEACHER_EDGE)
    badge(ax, MID_X, y5, 4)

    label(ax, MID_X, y5 - 1.05,
          "Note: at inference (no training), only this 384-d [CLS] output is kept.\nEverything below (the projection head + loss machinery) is discarded.",
          color=GRAY, fontsize=7.8, style="italic")

    # 5. projection head
    y6 = 13.75
    head_txt = "Projection head\n3-layer MLP (hidden 2048, GELU)\n↓ L2-normalize\n↓ weight-normalized linear\n→ K = 65536 logits"
    box(ax, STUDENT_X, y6, SIDE_W, 2.5, head_txt, STUDENT, STUDENT_EDGE, fontsize=8.8)
    box(ax, TEACHER_X, y6, SIDE_W, 2.5, head_txt, TEACHER, TEACHER_EDGE, fontsize=8.8)
    arrow(ax, (STUDENT_X, y5 - 0.55), (STUDENT_X, y6 + 1.25), color=STUDENT_EDGE)
    arrow(ax, (TEACHER_X, y5 - 0.55), (TEACHER_X, y6 + 1.25), color=TEACHER_EDGE)
    badge(ax, MID_X, y6, 5)

    # 6. centering (teacher only)
    y7 = 10.9
    arrow(ax, (STUDENT_X, y6 - 1.25), (STUDENT_X, y7 + 0.75), color=STUDENT_EDGE)
    box(ax, STUDENT_X, y7, SIDE_W, 1.5, "(no centering\non student side)", "#151515", "#3a3a3a", fontsize=8.6, tc=GRAY)
    arrow(ax, (TEACHER_X, y6 - 1.25), (TEACHER_X, y7 + 0.75), color=TEACHER_EDGE)
    box(ax, TEACHER_X, y7, SIDE_W, 1.5,
        "CENTERING\nsubtract running mean:  logits − c\nc ← 0.9·c + 0.1·mean_batch(teacher logits)",
        CENTERING, CENTERING_EDGE, fontsize=8.7, fontweight="bold")
    badge(ax, TEACHER_X - SIDE_W / 2 + 0.32, y7 + 0.75 - 0.08, 6)
    label(ax, MID_X, y7, "anti-collapse\nhalf 1 of 2:\n→ pushes toward\nUNIFORM", color=CENTERING_EDGE, fontsize=7.4)

    # 7. sharpening / softmax
    y8 = 8.4
    arrow(ax, (STUDENT_X, y7 - 0.75), (STUDENT_X, y8 + 0.75), color=STUDENT_EDGE)
    box(ax, STUDENT_X, y8, SIDE_W, 1.5, "Softmax\ntemperature τs = 0.1", STUDENT, STUDENT_EDGE, fontsize=9.3)
    arrow(ax, (TEACHER_X, y7 - 0.75), (TEACHER_X, y8 + 0.75), color=TEACHER_EDGE)
    box(ax, TEACHER_X, y8, SIDE_W, 1.5,
        "SHARPENING\nSoftmax, temperature τt = 0.04 → 0.07\nlow temp → peaky, confident distribution",
        SHARPENING, SHARPENING_EDGE, fontsize=8.7, fontweight="bold")
    badge(ax, TEACHER_X - SIDE_W / 2 + 0.32, y8 + 0.75 - 0.08, 7)
    label(ax, MID_X, y8, "anti-collapse\nhalf 2 of 2:\n→ pushes away\nfrom uniform", color=SHARPENING_EDGE, fontsize=7.4)

    # 8. outputs P_s / P_t
    y9 = 6.6
    arrow(ax, (STUDENT_X, y8 - 0.75), (STUDENT_X, y9 + 0.45), color=STUDENT_EDGE)
    box(ax, STUDENT_X, y9, SIDE_W, 0.9, "P_s   (student distribution, K=65536)", STUDENT, STUDENT_EDGE, fontsize=8.8)
    arrow(ax, (TEACHER_X, y8 - 0.75), (TEACHER_X, y9 + 0.45), color=TEACHER_EDGE)
    box(ax, TEACHER_X, y9, SIDE_W, 0.9, "P_t   (teacher distribution, K=65536)", TEACHER, TEACHER_EDGE, fontsize=8.8)

    # stop-gradient marker on teacher path
    y10 = 5.3
    box(ax, TEACHER_X, y10, 4.6, 0.85, "STOP-GRADIENT — teacher treated as a constant here", STOP, STOP_EDGE, fontsize=8.6, fontweight="bold")
    arrow(ax, (TEACHER_X, y9 - 0.45), (TEACHER_X, y10 + 0.42), color=STOP_EDGE)
    badge(ax, TEACHER_X - 4.6 / 2, y10 + 0.85 / 2 + 0.18, 8)

    # 9. loss
    yL = 3.45
    arrow(ax, (STUDENT_X, y9 - 0.45), (MID_X - 1.8, yL + 0.9), color=STUDENT_EDGE, rad=0.15)
    arrow(ax, (TEACHER_X, y10 - 0.42), (MID_X + 1.8, yL + 0.9), color=STOP_EDGE, rad=-0.15)
    box(ax, MID_X, yL, MID_W, 1.6,
        "Cross-Entropy Loss\nL = − Σₖ P_t(k) · log P_s(k)      (summed over student-crop × teacher-crop pairs)",
        LOSS, LOSS_EDGE, fontsize=10, fontweight="bold")
    badge(ax, MID_X - MID_W / 2 + 0.32, yL + 0.8 - 0.08, 9)

    # backprop lane back to student stack
    lane_x = 1.05
    line(ax, (MID_X - MID_W / 2 + 0.15, yL), (lane_x, yL), color=STUDENT_EDGE, lw=1.6)
    line(ax, (lane_x, yL), (lane_x, y4), color=STUDENT_EDGE, lw=1.9)
    arrow(ax, (lane_x, y4), (STUDENT_X - SIDE_W / 2 + 0.2, y4), color=STUDENT_EDGE, lw=1.6)
    label(ax, lane_x + 0.05, 2.15, "∇ backprop\nupdates θs\n(student\npath only)",
          color=STUDENT_EDGE, fontsize=7.6, weight="bold", ha="center")

    # legend
    yleg = 1.1
    items = [
        (STUDENT, STUDENT_EDGE, "Student path (gradient-trained)"),
        (TEACHER, TEACHER_EDGE, "Teacher path (EMA-only)"),
        (CENTERING, CENTERING_EDGE, "Centering"),
        (SHARPENING, SHARPENING_EDGE, "Sharpening"),
        (STOP, STOP_EDGE, "Stop-gradient"),
        (LOSS, LOSS_EDGE, "Loss"),
    ]
    lx = 1.0
    for fc, ec, txt in items:
        ax.add_patch(FancyBboxPatch((lx, yleg - 0.18), 0.46, 0.36, boxstyle="round,pad=0.02,rounding_size=0.06",
                                     linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=4))
        ax.text(lx + 0.64, yleg, txt, ha="left", va="center", fontsize=8.3, color=WHITE, zorder=4)
        lx += 0.9 + len(txt) * 0.105

    fig.savefig(OUT_ROOT / "dino_architecture.png", dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"wrote {OUT_ROOT / 'dino_architecture.png'}")


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build()
