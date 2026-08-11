"""Pull the [CLS] attention maps out of a frozen DINO ViT.

The last block's attention tells us, for each head, how much the classification
token drew on each image patch. Reshaped back onto the image grid that map is
the thing DINO is known for: object structure with no labels anywhere.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def cls_attention(model, x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[CLS]-to-patch attention from the last block.

    Args:
        x: image batch of shape [1, 3, H, W].

    Returns:
        Tensor [heads, H//patch_size, W//patch_size], each head normalized to [0, 1].
    """
    if x.shape[0] != 1:
        raise ValueError(f"expected a single image, got batch of {x.shape[0]}")

    h_feat = x.shape[-2] // patch_size
    w_feat = x.shape[-1] // patch_size

    # [1, heads, tokens, tokens]; token 0 is [CLS], the rest are patches.
    attn = model.get_last_selfattention(x)
    nh = attn.shape[1]
    attn = attn[0, :, 0, 1:].reshape(nh, h_feat, w_feat)

    # Per-head min-max so heads with different sharpness stay comparable.
    flat = attn.reshape(nh, -1)
    lo = flat.min(dim=1).values.view(nh, 1, 1)
    hi = flat.max(dim=1).values.view(nh, 1, 1)
    return (attn - lo) / (hi - lo + 1e-8)


def upsample(attn: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Resize an attention map up to image resolution."""
    out = F.interpolate(attn.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    return out.squeeze(0).float().cpu().numpy()


@torch.no_grad()
def attention_map(model, x: torch.Tensor, patch_size: int, size=None) -> np.ndarray:
    """Head-averaged [CLS] attention, upsampled to `size` (defaults to input size)."""
    attn = cls_attention(model, x, patch_size).mean(dim=0, keepdim=True)
    size = size or (x.shape[-2], x.shape[-1])
    return upsample(attn, size)[0]
