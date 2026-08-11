"""Cache a subset of Galaxy10 DECaLS locally.

The full dataset is ~2.5 GB. We stream it and keep only as many images as the
experiments need, written out as plain JPEGs plus a labels CSV so every later
script can read them without touching the `datasets` library again.
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset

DATASET = "matthieulel/galaxy10_decals"

# Label order as published with the dataset.
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def cache_split(split: str, n: int, seed: int = 0, root: Path = DATA_ROOT) -> Path:
    """Stream `n` shuffled examples from `split` and write them to disk."""
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    rows = []
    for i, example in enumerate(ds.take(n)):
        name = f"img_{i:05d}.jpg"
        example["image"].convert("RGB").save(out_dir / name, quality=95)
        rows.append({"file": name, "label": example["label"]})
        if (i + 1) % 250 == 0:
            print(f"  {split}: {i + 1}/{n}")

    with open(out_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split}: wrote {len(rows)} images to {out_dir}")
    return out_dir


def load_labels(split: str, root: Path = DATA_ROOT):
    """Read back a cached split as a list of (image_path, label)."""
    split_dir = root / split
    with open(split_dir / "labels.csv") as f:
        return [(split_dir / r["file"], int(r["label"])) for r in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--test", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache_split("train", args.train, args.seed)
    cache_split("test", args.test, args.seed)


if __name__ == "__main__":
    main()
