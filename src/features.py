"""Extract frozen [CLS] features for a cached split.

Also supports a randomly initialized model of the same architecture. That is
the control that separates two very different explanations for any result:
"the self-supervised training learned something" versus "a vision transformer
plus this image statistic would have done it anyway".
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import load_labels
from model import ARCHS, HUB_REPO, build_transform, get_device, load_dino

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "features"


def load_random_init(arch: str, device):
    """Same architecture, untrained weights."""
    entrypoint, patch_size = ARCHS[arch]
    model = torch.hub.load(HUB_REPO, entrypoint, pretrained=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, patch_size


@torch.no_grad()
def extract(model, rows, transform, device, batch_size=16):
    feats, labels = [], []
    batch = []
    for i, (path, label) in enumerate(rows):
        batch.append(transform(Image.open(path).convert("RGB")))
        labels.append(label)
        if len(batch) == batch_size or i == len(rows) - 1:
            x = torch.stack(batch).to(device)
            feats.append(model(x).float().cpu().numpy())
            batch = []
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(rows)}")
    return np.concatenate(feats), np.array(labels)


def raw_pixels(rows, size=32):
    """Downsampled raw-pixel baseline."""
    feats, labels = [], []
    for path, label in rows:
        img = Image.open(path).convert("RGB").resize((size, size))
        feats.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
        labels.append(label)
    return np.stack(feats), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vits8")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    if args.random_init:
        model, patch_size = load_random_init(args.arch, device)
        tag = f"{args.arch}_random"
    else:
        model, patch_size = load_dino(args.arch, device)
        tag = args.arch

    transform = build_transform(args.size, patch_size)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        rows = load_labels(split)
        print(f"{tag} / {split}: {len(rows)} images")
        feats, labels = extract(model, rows, transform, device, args.batch_size)
        np.save(OUT_ROOT / f"{tag}_{split}_X.npy", feats)
        np.save(OUT_ROOT / f"{tag}_{split}_y.npy", labels)
        print(f"  saved {feats.shape}")

        if not args.random_init and args.arch == "vits8":
            px, py = raw_pixels(rows)
            np.save(OUT_ROOT / f"pixels_{split}_X.npy", px)
            np.save(OUT_ROOT / f"pixels_{split}_y.npy", py)
            print(f"  saved pixels {px.shape}")


if __name__ == "__main__":
    main()
