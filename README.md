# Frozen self-supervised features, out of domain

What does a self-supervised vision model know about images it was never trained on?

This applies **DINO** (Caron et al., 2021) — a ViT trained on ImageNet with no labels —
to **galaxy morphology**, a domain absent from its training data. The published weights
are used frozen. Nothing here is trained or fine-tuned.

Two questions:

1. Where does the model look, and is that anything more than pixel brightness?
2. Are the frozen features good enough to tell galaxy shapes apart?

## Results

**k-NN probe** — 20-NN on frozen features, 1,500 held-out galaxies, 10 morphology
classes. No training; labels used only for scoring.

| features | accuracy |
|---|---|
| DINO ViT-S/8 | **50.4%** |
| raw pixels (32×32 RGB) | 41.9% |
| same ViT-S/8, randomly initialized | 31.1% |
| majority class | 16.1% |
| chance | 10.0% |

The random-init row is the point of the table. Identical architecture, identical input
pipeline, untrained weights — and it lands *below* raw pixels. The transformer is not
what carries the result; the self-supervised pretraining is, by 19.3 points.

Per class, the pattern is interpretable: strong on global geometry (Round Smooth 78%,
Edge-on 77%/75%), weak where astrophysical context is required (Merging 26%, Disturbed
4%). Frozen ImageNet-SSL features encode **shape**, and galaxy morphology is only
partly shape.

**Attention control** — a galaxy cutout is a bright blob on a dark field, so attention
that merely tracked brightness would look impressive and mean nothing. Compared against
two null models at matched resolution, over 300 held-out images:

| comparison | Spearman ρ | top-20% IoU |
|---|---|---|
| attention vs brightness | +0.564 ± 0.100 | 0.624 ± 0.095 |
| attention vs centre prior | +0.077 ± 0.178 | 0.327 ± 0.111 |

Not a centre prior. Correlated with brightness but not reducible to it — and the
residual is structured rather than noise. Relative to brightness, attention
**under-weights the diffuse body of the galaxy and over-weights compact point sources**;
in `outputs/control_vits8.png` the galaxy reads negative and the foreground stars read
positive. If attention were a brightness proxy that residual would be unstructured.

Worth stating plainly: **it fires on foreground stars too.** This is an object-ness
prior transferring, not a galaxy detector.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and ffmpeg for video output.
Runs on Apple Silicon via MPS, or CPU.

```bash
uv sync
```

## Usage

```bash
# cache a subset of Galaxy10 DECaLS (streamed, ~180 MB)
uv run python src/data.py --train 3000 --test 1500

# attention overlay grids
uv run python src/render_stills.py --arch vits8

# is attention just brightness?
uv run python src/control_brightness.py --arch vits8 --limit 300

# frozen features, plus the random-init control
uv run python src/features.py --arch vits8
uv run python src/features.py --arch vits8 --random-init

# k-NN probe
uv run python src/knn_probe.py

# the whole story as one 14s video
uv run python src/render_story.py --arch vits8
```

Artifacts land in `outputs/`, cached images in `data/`; both are gitignored.

## Layout

| file | purpose |
|---|---|
| `src/data.py` | stream and cache Galaxy10 DECaLS |
| `src/model.py` | load frozen DINO weights via torch.hub |
| `src/attention.py` | [CLS]→patch attention maps |
| `src/render_stills.py` | raw / attention / overlay grids |
| `src/control_brightness.py` | brightness and centre-prior null models |
| `src/features.py` | frozen [CLS] features, incl. random-init control |
| `src/knn_probe.py` | weighted k-NN evaluation |
| `src/focus.py` | select and prepare the single cutout used for the comparison |
| `src/render_story.py` | reveal → brightness control → probe results, as one video |

## Notes

- Images are 256×256 and both patch sizes divide evenly, so no resizing is needed.
  ViT-S/8 gives a 32×32 attention grid; ViT-S/16 gives 16×16 and iterates faster.
- Inputs use ImageNet normalization, which is not obviously correct for astronomical
  imagery. A percentile or asinh stretch is an untested knob in `build_transform`.
- Class 4 (Cigar Shaped Smooth) has only 24 test examples; its accuracy is noisy.

## Credits

- DINO — [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294),
  Caron et al., 2021 · [facebookresearch/dino](https://github.com/facebookresearch/dino)
- Data — [Galaxy10 DECaLS](https://huggingface.co/datasets/matthieulel/galaxy10_decals),
  built from DESI Legacy Imaging Surveys and Galaxy Zoo
