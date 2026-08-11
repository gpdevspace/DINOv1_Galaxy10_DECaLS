"""Load frozen DINO ViT weights and prepare images for them.

Nothing here trains. The weights are the ones released with the DINO paper,
pulled through torch.hub, and used exactly as published.
"""

import torch
from PIL import Image
from torchvision import transforms

HUB_REPO = "facebookresearch/dino:main"

# Patch size is baked into the architecture name and we need it to reshape
# attention back onto the image grid.
ARCHS = {
    "vits16": ("dino_vits16", 16),
    "vits8": ("dino_vits8", 8),
    "vitb16": ("dino_vitb16", 16),
    "vitb8": ("dino_vitb8", 8),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dino(arch: str = "vits16", device: torch.device | None = None):
    """Return (frozen model, patch_size)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {list(ARCHS)}")
    entrypoint, patch_size = ARCHS[arch]
    device = device or get_device()

    model = torch.hub.load(HUB_REPO, entrypoint)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, patch_size


def build_transform(size: int, patch_size: int):
    """Resize so the image divides evenly into patches, then ImageNet-normalize."""
    size = (size // patch_size) * patch_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(path, transform) -> torch.Tensor:
    """Load one image as a batched tensor of shape [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)
