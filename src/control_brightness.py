"""Control: is DINO's attention just tracking brightness?

A galaxy cutout is a bright blob on a dark field, so an attention map that
merely followed pixel brightness would look impressive and mean nothing. This
compares attention against two null models:

  brightness — image luminance, downsampled to the attention grid so the
               comparison is resolution-fair, then upsampled back
  center     — a fixed radial Gaussian, since the galaxy is always centered

Reported per image and averaged: Spearman correlation and IoU of the top-20%
masks. High agreement with either null model would mean the attention map is
not showing us anything the nulls don't already explain.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from attention import attention_map
from data import CLASS_NAMES, load_labels
from model import build_transform, get_device, load_dino, load_image

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def luminance_at_grid(path, size, grid):
    """Image luminance, band-limited to the attention grid then upsampled back."""
    img = Image.open(path).convert("L").resize((size, size))
    lum = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None, None]
    small = F.interpolate(lum, size=(grid, grid), mode="area")
    back = F.interpolate(small, size=(size, size), mode="bilinear", align_corners=False)
    return back[0, 0].numpy()


def center_prior(size, sigma_frac=0.2):
    """Radial Gaussian centred on the frame."""
    coords = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(coords, coords)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma_frac**2))


def rank(a):
    flat = a.ravel()
    order = flat.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(flat))
    return ranks


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / (denom + 1e-12))


def top_iou(a, b, frac=0.2):
    """IoU of the top-`frac` masks of two maps."""
    k = int(a.size * frac)
    ma = np.zeros(a.size, dtype=bool)
    mb = np.zeros(b.size, dtype=bool)
    ma[a.ravel().argsort()[-k:]] = True
    mb[b.ravel().argsort()[-k:]] = True
    union = (ma | mb).sum()
    return float((ma & mb).sum() / (union + 1e-12))


def run(model, patch_size, rows, size, device, limit):
    transform = build_transform(size, patch_size)
    grid = size // patch_size
    prior = center_prior(size)
    stats = {"lum_rho": [], "lum_iou": [], "ctr_rho": [], "ctr_iou": []}

    for i, (path, _) in enumerate(rows[:limit]):
        x = load_image(path, transform).to(device)
        attn = attention_map(model, x, patch_size)
        lum = luminance_at_grid(path, size, grid)

        stats["lum_rho"].append(spearman(attn, lum))
        stats["lum_iou"].append(top_iou(attn, lum))
        stats["ctr_rho"].append(spearman(attn, prior))
        stats["ctr_iou"].append(top_iou(attn, prior))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{min(limit, len(rows))}")

    return {k: np.array(v) for k, v in stats.items()}


def figure(model, patch_size, examples, size, device, out_path):
    transform = build_transform(size, patch_size)
    grid = size // patch_size
    n = len(examples)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.4))
    fig.patch.set_facecolor("black")

    for col, (path, label) in enumerate(examples):
        x = load_image(path, transform).to(device)
        attn = attention_map(model, x, patch_size)
        lum = luminance_at_grid(path, size, grid)
        raw = np.asarray(Image.open(path).convert("RGB").resize((size, size)))
        diff = attn - (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)

        axes[0, col].imshow(raw)
        axes[0, col].set_title(
            f"{CLASS_NAMES[label]}\nρ={spearman(attn, lum):.2f}", color="white", fontsize=7, pad=4
        )
        axes[1, col].imshow(lum, cmap="inferno")
        axes[2, col].imshow(attn, cmap="inferno")
        axes[3, col].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
        for row in range(4):
            axes[row, col].axis("off")

    for row, name in enumerate(["image", "brightness", "DINO attention", "attn − brightness"]):
        axes[row, 0].text(
            -0.08, 0.5, name, color="white", fontsize=8, rotation=90,
            va="center", ha="center", transform=axes[row, 0].transAxes,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vits8")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    device = get_device()
    model, patch_size = load_dino(args.arch, device)
    rows = load_labels(args.split)

    stats = run(model, patch_size, rows, args.size, device, args.limit)

    print(f"\n=== control over {len(stats['lum_rho'])} images ({args.arch}) ===")
    print("attention vs brightness:")
    print(f"  spearman rho  {stats['lum_rho'].mean():+.3f} ± {stats['lum_rho'].std():.3f}")
    print(f"  top-20% IoU    {stats['lum_iou'].mean():.3f} ± {stats['lum_iou'].std():.3f}")
    print("attention vs center prior:")
    print(f"  spearman rho  {stats['ctr_rho'].mean():+.3f} ± {stats['ctr_rho'].std():.3f}")
    print(f"  top-20% IoU    {stats['ctr_iou'].mean():.3f} ± {stats['ctr_iou'].std():.3f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    from render_stills import pick_examples

    examples = pick_examples(rows, [7, 5, 8, 2, 1, 9])
    figure(model, patch_size, examples, args.size, device, OUT_ROOT / f"control_{args.arch}.png")


if __name__ == "__main__":
    main()
