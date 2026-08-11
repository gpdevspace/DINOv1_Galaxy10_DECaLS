"""Render raw / attention / overlay grids for a sample of galaxies."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from attention import attention_map
from data import CLASS_NAMES, load_labels
from model import build_transform, get_device, load_dino, load_image

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def pick_examples(rows, classes):
    """One example per requested class, in the order given."""
    picked = []
    for cls in classes:
        for path, label in rows:
            if label == cls and path not in [p for p, _ in picked]:
                picked.append((path, label))
                break
    return picked


def render_grid(model, patch_size, examples, size, out_path, device):
    transform = build_transform(size, patch_size)
    n = len(examples)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.4))
    fig.patch.set_facecolor("black")

    for col, (path, label) in enumerate(examples):
        x = load_image(path, transform).to(device)
        attn = attention_map(model, x, patch_size)
        raw = np.asarray(Image.open(path).convert("RGB").resize((size, size)))

        axes[0, col].imshow(raw)
        axes[0, col].set_title(CLASS_NAMES[label], color="white", fontsize=7, pad=4)
        axes[1, col].imshow(attn, cmap="inferno")
        axes[2, col].imshow(raw)
        axes[2, col].imshow(attn, cmap="inferno", alpha=0.55)

        for row in range(3):
            axes[row, col].axis("off")

    for row, name in enumerate(["image", "DINO attention", "overlay"]):
        axes[row, 0].text(
            -0.08, 0.5, name, color="white", fontsize=9, rotation=90,
            va="center", ha="center", transform=axes[row, 0].transAxes,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vits16")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--split", default="train")
    parser.add_argument("--classes", type=int, nargs="+", default=[7, 5, 8, 2, 1, 9, 6, 0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = get_device()
    model, patch_size = load_dino(args.arch, device)
    rows = load_labels(args.split)
    examples = pick_examples(rows, args.classes)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT_ROOT / f"stills_{args.arch}_{args.size}.png"
    render_grid(model, patch_size, examples, args.size, out_path, device)


if __name__ == "__main__":
    main()
