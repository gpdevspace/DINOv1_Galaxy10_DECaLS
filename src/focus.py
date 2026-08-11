"""Pick and prepare the single galaxy used for the brightness comparison.

The comparison only reads clearly if the viewer can follow two specific places
across four panels: the diffuse galaxy at the centre, and a compact foreground
source somewhere off-centre. This searches for a cutout where those two places
disagree most strongly with brightness *and* both are actually visible, so the
effect can be seen rather than argued for.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from attention import attention_map
from control_brightness import luminance_at_grid
from model import build_transform, load_image

CENTER_FRAC = 0.15  # radius of the disc treated as "the galaxy"
EXCLUDE_FRAC = 0.26  # keep the point source clear of the galaxy
EDGE_FRAC = 0.15  # and clear of the panel edge, so its ring and label fit
STAR_MIN_LUM = 0.32  # a marked source has to be bright enough to look like one
GALAXY_MIN_LUM = 0.14  # and the galaxy has to be visibly extended


def diverging(diff):
    """Signed map: orange where attention exceeds brightness, blue where it trails.

    Black at zero, so it reads on a dark canvas without a white slab in the middle.
    """
    pos = np.clip(diff, 0, 1)[..., None] * np.array([255, 145, 60], dtype=np.float32)
    neg = np.clip(-diff, 0, 1)[..., None] * np.array([80, 150, 255], dtype=np.float32)
    return np.clip(pos + neg, 0, 255).astype(np.uint8)


def _geometry(size):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    centre = r < size * CENTER_FRAC
    outside = (
        (r > size * EXCLUDE_FRAC)
        & (xx > size * EDGE_FRAC)
        & (xx < size * (1 - EDGE_FRAC))
        & (yy > size * EDGE_FRAC)
        & (yy < size * (1 - EDGE_FRAC))
    )
    return centre, outside, xx, yy


def _local_maxima(lum, window):
    """True where a pixel is the brightest thing in its neighbourhood."""
    t = torch.from_numpy(lum.astype(np.float32))[None, None]
    pooled = F.max_pool2d(t, kernel_size=window, stride=1, padding=window // 2)
    return (t >= pooled)[0, 0].numpy()


def analyse(lum, diff, size):
    """Score this cutout and locate the two places to mark.

    Returns (score, star_xy, galaxy_xy). Score is -inf if the cutout doesn't
    have both a bright compact source and a visible central galaxy.
    """
    centre, outside, xx, yy = _geometry(size)

    if lum[centre].mean() < GALAXY_MIN_LUM:
        return -np.inf, None, None

    peaks = _local_maxima(lum, window=max(size // 24 | 1, 3))
    candidates = peaks & outside & (lum > STAR_MIN_LUM)
    if not candidates.any():
        return -np.inf, None, None

    # Among genuinely bright compact sources, take the one DINO over-attends most.
    scored = np.where(candidates, diff, -np.inf)
    idx = np.unravel_index(np.argmax(scored), scored.shape)
    star_strength = float(scored[idx])
    centre_strength = float(diff[centre].mean())

    # Brightness-weighted centroid of the central region, so the ring sits on
    # the galaxy rather than on the geometric centre of the frame.
    w = np.where(centre, lum, 0.0)
    total = w.sum() + 1e-8
    gx = int((w * xx).sum() / total)
    gy = int((w * yy).sum() / total)

    return star_strength - centre_strength, (int(idx[1]), int(idx[0])), (gx, gy)


def _maps(model, patch_size, path, size, device):
    transform = build_transform(256, patch_size)
    x = load_image(path, transform).to(device)
    attn = attention_map(model, x, patch_size, size=(size, size))
    lum = luminance_at_grid(path, size, 256 // patch_size)
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)
    return lum, attn


def build_panels(model, patch_size, path, size, device, boost):
    """Raw / brightness / attention / difference for one cutout, plus markers."""
    lum, attn = _maps(model, patch_size, path, size, device)
    diff = attn - lum
    _, star, galaxy = analyse(lum, diff, size)

    raw_img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    raw = np.clip(np.asarray(raw_img, dtype=np.float32) * boost, 0, 255).astype(np.uint8)

    cmap = colormaps["inferno"]
    return {
        "raw": raw,
        "lum": (cmap(lum)[:, :, :3] * 255).astype(np.uint8),
        "attn": (cmap(attn)[:, :, :3] * 255).astype(np.uint8),
        "diff": diverging(diff),
        "star": star,
        "galaxy": galaxy or (size // 2, size // 2),
    }


def choose(model, patch_size, rows, size, device, n_candidates=120):
    """Scan candidates and return the path that shows the effect most clearly."""
    best, best_path = -np.inf, None
    for path, _ in rows[:n_candidates]:
        lum, attn = _maps(model, patch_size, path, size, device)
        score, star, _ = analyse(lum, attn - lum, size)
        if star is not None and score > best:
            best, best_path = score, path

    if best_path is None:
        raise RuntimeError("no cutout had both a bright compact source and a visible galaxy")
    print(f"  focus: {Path(best_path).name} (score {best:.3f})")
    return best_path
