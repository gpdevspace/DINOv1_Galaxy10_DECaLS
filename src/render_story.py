"""Render the full story as one short video.

Three acts, one argument:

  1. reveal   — where a frozen DINO looks on galaxies it never trained on
  2. control  — one galaxy, four panels, two labelled places held fixed across
                all of them, so brightness and attention can actually be compared
  3. probe    — the k-NN numbers, and the untrained-architecture control

Frames are piped straight into ffmpeg.
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

import focus
from attention import attention_map
from data import load_labels
from model import build_transform, get_device, load_dino, load_image

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

# Measured by knn_probe.py on 1,500 held-out galaxies.
BARS = [
    ("DINO ViT-S/8", 50.4, (255, 138, 61)),
    ("raw pixels", 41.9, (120, 132, 150)),
    ("same ViT, untrained", 31.1, (86, 90, 104)),
]
BAR_SCALE = 60.0


def load_font(size, bold=False):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except OSError:
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
    return ImageFont.load_default()


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def pick_n_per_class(rows, classes, per_class):
    picked = []
    for cls in classes:
        found = 0
        for path, label in rows:
            if label == cls:
                picked.append((path, label))
                found += 1
                if found == per_class:
                    break
    return picked


def tile(cells, grid, cell, gap, width, height, reserve):
    out = np.zeros((height, width, 3), dtype=np.uint8)
    span_w = grid * cell + (grid - 1) * gap
    rows_n = int(np.ceil(len(cells) / grid))
    span_h = rows_n * cell + (rows_n - 1) * gap
    off_x = (width - span_w) // 2
    off_y = (height - reserve - span_h) // 2
    for i, c in enumerate(cells):
        r, col = divmod(i, grid)
        y = off_y + r * (cell + gap)
        x = off_x + col * (cell + gap)
        out[y : y + cell, x : x + cell] = c
    return out


def build_reveal(model, patch_size, examples, cell, device, boost):
    transform = build_transform(256, patch_size)
    cmap = colormaps["inferno"]
    raws, overlays = [], []
    for path, _ in examples:
        x = load_image(path, transform).to(device)
        attn = attention_map(model, x, patch_size, size=(cell, cell))
        raw = np.clip(
            np.asarray(Image.open(path).convert("RGB").resize((cell, cell)), dtype=np.float32) * boost,
            0, 255,
        )
        heat = cmap(attn)[:, :, :3] * 255.0
        raws.append(raw.astype(np.uint8))
        overlays.append(np.clip(raw * 0.45 + heat * 0.55, 0, 255).astype(np.uint8))
    return raws, overlays


def place(panel, px, py, width, height):
    """Drop one panel onto an otherwise black canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[py : py + panel.shape[0], px : px + panel.shape[1]] = panel
    return canvas


def draw_markers(img, markers, px, py, font, alpha=1.0):
    """Ring + label at fixed image locations, repeated on every panel."""
    if alpha <= 0:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    a = int(235 * alpha)
    for (mx, my), label, side in markers:
        cx, cy = px + mx, py + my
        r = 54
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, a), width=3)
        ty = cy - r - 26 if side == "above" else cy + r + 26
        draw.text((cx + 2, ty + 2), label, font=font, fill=(0, 0, 0, int(200 * alpha)), anchor="mm")
        draw.text((cx, ty), label, font=font, fill=(255, 255, 255, int(245 * alpha)), anchor="mm")


def draw_legend(img, cx, y, w, h, font, alpha=1.0):
    """Gradient key for the difference map, so the colours mean something."""
    if alpha <= 0:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    ramp = np.linspace(-1, 1, w, dtype=np.float32)
    colors = focus.diverging(ramp[None, :])[0]
    for i in range(w):
        draw.line([cx - w // 2 + i, y, cx - w // 2 + i, y + h],
                  fill=tuple(int(v) for v in colors[i]))
    draw.rectangle([cx - w // 2, y, cx + w // 2, y + h], outline=(70, 72, 82, int(255 * alpha)))
    grey = (168, 170, 182, int(255 * alpha))
    draw.text((cx - w // 2 - 16, y + h // 2), "below brightness", font=font, fill=grey, anchor="rm")
    draw.text((cx + w // 2 + 16, y + h // 2), "above brightness", font=font, fill=grey, anchor="lm")


def text_block(img, lines, width, height):
    """Draw the caption stack in the reserved band at the bottom."""
    draw = ImageDraw.Draw(img, "RGBA")
    y = height - 128
    for content, font, color in lines:
        if content:
            draw.text((width // 2, y), content, font=font, fill=color, anchor="mm")
        y += 52


def frame_with_caption(arr, lines, width, height):
    img = Image.fromarray(arr)
    text_block(img, lines, width, height)
    return np.asarray(img)


def draw_bars(width, height, progress, fonts):
    """Act 3: horizontal bars growing to their measured values."""
    f_title, f_label, f_value, f_sub = fonts
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.text((width // 2, 210), "20-NN on frozen features", font=f_title,
              fill=(255, 255, 255), anchor="mm")
    draw.text((width // 2, 268), "1,500 held-out galaxies · no training",
              font=f_sub, fill=(150, 152, 162), anchor="mm")

    x0, x1 = 70, width - 70
    span = x1 - x0
    top = 430
    row_h = 230
    bar_h = 60
    chance_x = x0 + span * (10.0 / BAR_SCALE)

    for i, (name, value, color) in enumerate(BARS):
        # Bars arrive one after another rather than all at once.
        local = smoothstep(np.clip(progress * len(BARS) - i, 0, 1))
        y = top + i * row_h
        draw.text((x0, y), name, font=f_label, fill=(226, 228, 236), anchor="lm")

        bar_y = y + 48
        full = span * (value / BAR_SCALE)
        draw.rectangle([x0, bar_y, x0 + span, bar_y + bar_h], fill=(26, 27, 33))
        if local > 0:
            draw.rectangle([x0, bar_y, x0 + full * local, bar_y + bar_h], fill=color)
            draw.text((x0 + full * local + 18, bar_y + bar_h // 2), f"{value * local:.1f}%",
                      font=f_value, fill=color, anchor="lm")
        # Chance tick, drawn inside the track so it never crosses a label.
        draw.line([chance_x, bar_y, chance_x, bar_y + bar_h], fill=(126, 128, 140), width=2)

    last_bottom = top + (len(BARS) - 1) * row_h + 48 + bar_h
    draw.text((chance_x, last_bottom + 26), "chance", font=f_sub,
              fill=(112, 114, 126), anchor="mt")
    return np.asarray(img)


def fade(arr, factor):
    return np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vits8")
    parser.add_argument("--split", default="test")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1350)
    parser.add_argument("--reserve", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--boost", type=float, default=1.35)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    width, height, fps = args.width, args.height, args.fps
    device = get_device()
    model, patch_size = load_dino(args.arch, device)
    rows = load_labels(args.split)

    gap = 10
    cell_a = (width - 3 * gap - 40) // 4

    print("building act 1 …")
    ex_a = pick_n_per_class(rows, [7, 5, 8, 2, 6, 9, 1, 3], 2)
    raws, overlays = build_reveal(model, patch_size, ex_a, cell_a, device, args.boost)
    raw_grid = tile(raws, 4, cell_a, gap, width, height, args.reserve)
    attn_grid = tile(overlays, 4, cell_a, gap, width, height, args.reserve)

    print("building act 2 …")
    # Keep the upscale modest — these cutouts are 256px and noisy, and blowing
    # them up much further makes the sensor noise louder than the galaxy.
    panel = 820
    px = (width - panel) // 2
    py = 150
    focus_path = focus.choose(model, patch_size, rows, panel, device)
    f = focus.build_panels(model, patch_size, focus_path, panel, device, 1.15)

    focus_raw = place(f["raw"], px, py, width, height)
    focus_lum = place(f["lum"], px, py, width, height)
    focus_attn = place(f["attn"], px, py, width, height)
    focus_diff = place(f["diff"], px, py, width, height)

    sx, sy = f["star"]
    markers = [
        (f["galaxy"], "the galaxy", "below"),
        ((sx, sy), "a foreground star", "above" if sy > panel * 0.3 else "below"),
    ]

    f_big = load_font(46)
    f_sm = load_font(27)
    f_mark = load_font(28)
    f_leg = load_font(23)
    fonts = (load_font(44), load_font(34), load_font(34), load_font(26))

    white = (255, 255, 255, 255)
    grey = (176, 178, 190, 255)
    legend_y = py + panel + 34

    # Act 2 introduces each map on its own before blinking between them. Going
    # straight to the blink gives the viewer nothing to compare against and
    # reads as flicker rather than as a comparison.
    segments = [
        ("hold_raw", 0.8),
        ("wipe", 1.5),
        ("hold_attn", 0.9),
        ("gap1", 0.3),
        ("focus_raw", 1.6),
        ("show_lum", 1.9),
        ("show_attn", 1.9),
        ("compare", 3.3),
        ("diff_reveal", 1.7),
        ("diff_hold", 1.9),
        ("gap2", 0.3),
        ("bars", 2.3),
        ("bars_hold", 1.6),
    ]

    out_path = Path(args.out) if args.out else OUT_ROOT / "story.mp4"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total = 0
    for name, seconds in segments:
        n = max(int(seconds * fps), 1)
        for i in range(n):
            t = i / max(n - 1, 1)

            if name == "hold_raw":
                frame = frame_with_caption(raw_grid, [
                    ("16 galaxies", f_big, white),
                    ("no labels, no astronomy training", f_sm, grey),
                ], width, height)

            elif name == "wipe":
                y = int(smoothstep(t) * height)
                arr = raw_grid.copy()
                arr[:y] = attn_grid[:y]
                if 0 < y < height:
                    lo, hi = max(0, y - 3), min(height, y + 3)
                    arr[lo:hi] = np.clip(arr[lo:hi].astype(np.float32) + 90, 0, 255).astype(np.uint8)
                frame = frame_with_caption(arr, [
                    ("where the model looks", f_big, white),
                    ("DINO ViT-S/8, frozen", f_sm, grey),
                ], width, height)

            elif name == "hold_attn":
                frame = frame_with_caption(attn_grid, [
                    ("where the model looks", f_big, white),
                    ("it finds objects — including foreground stars", f_sm, grey),
                ], width, height)

            elif name == "focus_raw":
                # Establish the two places the viewer will track from here on.
                img = Image.fromarray(focus_raw)
                draw_markers(img, markers, px, py, f_mark, smoothstep(min(t * 3, 1.0)))
                text_block(img, [
                    ("one galaxy, one star", f_big, white),
                    ("watch these two spots", f_sm, grey),
                ], width, height)
                frame = np.asarray(img)

            elif name == "show_lum":
                img = Image.fromarray(focus_lum)
                draw_markers(img, markers, px, py, f_mark)
                text_block(img, [
                    ("brightness", f_big, white),
                    ("what the pixels say", f_sm, grey),
                ], width, height)
                frame = np.asarray(img)

            elif name == "show_attn":
                # Ease over from brightness so the change is a transition, not a cut.
                a = smoothstep(min(t / 0.22, 1.0))
                arr = (focus_lum * (1 - a) + focus_attn * a).astype(np.uint8)
                img = Image.fromarray(arr)
                draw_markers(img, markers, px, py, f_mark)
                text_block(img, [
                    ("attention", f_big, white),
                    ("where DINO looks", f_sm, grey),
                ], width, height)
                frame = np.asarray(img)

            elif name == "compare":
                # Soft square wave between the two maps; the markers never move.
                # Slow enough to read each state, having already seen both held.
                phase = (t * 1.8) % 1.0
                alpha = float(
                    smoothstep(np.clip(phase / 0.20, 0, 1))
                    - smoothstep(np.clip((phase - 0.5) / 0.20, 0, 1))
                )
                arr = (focus_lum * (1 - alpha) + focus_attn * alpha).astype(np.uint8)
                img = Image.fromarray(arr)
                draw_markers(img, markers, px, py, f_mark)
                label = "attention" if alpha > 0.5 else "brightness"
                text_block(img, [
                    ("is it just brightness?", f_big, white),
                    (label, f_sm, grey),
                ], width, height)
                frame = np.asarray(img)

            elif name in ("diff_reveal", "diff_hold"):
                reveal = smoothstep(min(t * 2.2, 1.0)) if name == "diff_reveal" else 1.0
                arr = (focus_diff * reveal).astype(np.uint8)
                img = Image.fromarray(arr)
                draw_markers(img, markers, px, py, f_mark)
                draw_legend(img, width // 2, legend_y, 560, 26, f_leg, reveal)
                lines = ([("attention minus brightness", f_big, white),
                          ("ρ = 0.56 — related, not the same", f_sm, grey)]
                         if name == "diff_reveal" else
                         [("it prefers compact objects", f_big, white),
                          ("and under-weights the diffuse galaxy", f_sm, grey)])
                text_block(img, lines, width, height)
                frame = np.asarray(img)

            elif name == "bars":
                frame = frame_with_caption(
                    draw_bars(width, height, t, fonts),
                    [("", f_big, white), ("", f_sm, grey)], width, height
                )

            elif name == "bars_hold":
                frame = frame_with_caption(draw_bars(width, height, 1.0, fonts), [
                    ("the training, not the architecture", f_big, white),
                    ("same net untrained scores below raw pixels", f_sm, grey),
                ], width, height)

            else:  # short dip through black between acts
                prev = attn_grid if name == "gap1" else focus_diff
                nxt = focus_raw if name == "gap1" else draw_bars(width, height, 0.0, fonts)
                frame = fade(prev, 1 - 2 * t) if t < 0.5 else fade(nxt, 2 * t - 1)

            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            total += 1

    proc.stdin.close()
    proc.wait()
    print(f"wrote {out_path} ({total} frames, {total / fps:.1f}s)")


if __name__ == "__main__":
    main()
