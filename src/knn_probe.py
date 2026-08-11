"""Weighted k-NN probe over frozen features.

Follows the evaluation protocol from the DINO paper: L2-normalize, rank the
train set by cosine similarity, and let the top-k neighbours vote with weights
exp(sim / T).

No training happens at any point. The labels are used only to look up what the
retrieved neighbours were and to score the answer, which is what makes this a
measurement of the features rather than a model fit to the task.
"""

import argparse
from pathlib import Path

import numpy as np

from data import CLASS_NAMES

FEAT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "features"


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def knn_predict(train_X, train_y, test_X, k=20, temperature=0.07, n_classes=10):
    train_X = l2_normalize(train_X.astype(np.float32))
    test_X = l2_normalize(test_X.astype(np.float32))

    preds = np.empty(len(test_X), dtype=np.int64)
    for start in range(0, len(test_X), 256):
        sims = test_X[start : start + 256] @ train_X.T
        idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, idx, axis=1)
        weights = np.exp(top_sims / temperature)

        votes = np.zeros((len(idx), n_classes), dtype=np.float32)
        neighbour_labels = train_y[idx]
        for c in range(n_classes):
            votes[:, c] = (weights * (neighbour_labels == c)).sum(axis=1)
        preds[start : start + len(idx)] = votes.argmax(axis=1)
    return preds


def evaluate(tag, k, temperature):
    try:
        train_X = np.load(FEAT_ROOT / f"{tag}_train_X.npy")
        train_y = np.load(FEAT_ROOT / f"{tag}_train_y.npy")
        test_X = np.load(FEAT_ROOT / f"{tag}_test_X.npy")
        test_y = np.load(FEAT_ROOT / f"{tag}_test_y.npy")
    except FileNotFoundError:
        return None

    preds = knn_predict(train_X, train_y, test_X, k=k, temperature=temperature)
    acc = float((preds == test_y).mean())

    per_class = {}
    for c in range(10):
        mask = test_y == c
        if mask.sum():
            per_class[c] = float((preds[mask] == c).mean())
    return acc, per_class, test_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--tags", nargs="+", default=["vits8", "vits8_random", "pixels"]
    )
    args = parser.parse_args()

    results = {}
    for tag in args.tags:
        out = evaluate(tag, args.k, args.temperature)
        if out is None:
            print(f"(skipping {tag}: features not found)")
            continue
        results[tag] = out

    if not results:
        return

    any_y = next(iter(results.values()))[2]
    majority = float((any_y == np.bincount(any_y).argmax()).mean())

    print(f"\n=== {args.k}-NN on frozen features, {len(any_y)} held-out galaxies ===")
    print(f"{'features':<16} {'accuracy':>9}")
    print("-" * 27)
    for tag, (acc, _, _) in results.items():
        print(f"{tag:<16} {acc:>8.1%}")
    print(f"{'majority class':<16} {majority:>8.1%}")
    print(f"{'chance (1/10)':<16} {0.1:>8.1%}")

    if "vits8" in results:
        print("\nper-class accuracy (DINO ViT-S/8):")
        for c, v in sorted(results["vits8"][1].items()):
            print(f"  {CLASS_NAMES[c]:<26} {v:>6.1%}")


if __name__ == "__main__":
    main()
