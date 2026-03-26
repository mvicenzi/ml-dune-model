# ml-dune-model

## Installation

WarpConvNet requires Python >= 3.9 and a working CUDA toolchain.
See: [WarpConvNet/installation](https://nvlabs.github.io/WarpConvNet/getting_started/installation/).

1. Create a Python/`uv` environment to install packages.

2. Install dependencies for WarpConvNet. Note: other packages might need to be installed to resolve build errors. 
```
export CUDA=cu126  # For CUDA 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}$
uv pip install build ninja
uv pip install cupy-cuda12x
uv pip install git+https://github.com/rusty1s/pytorch_scatter.git
MAX_JOBS=2 uv pip install flash-attn --no-build-isolation     # reduce parallelism to help building
```

3. Clone and install WarpConvNet.
```
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
uv pip install .
```

4. Install additional packages as required for running.

## Training

Two training modes are available.
Both scripts use [python-fire](https://github.com/google/python-fire) for CLI argument parsing, so any parameter in `main()` can be overridden by passing `--param=value` on the command line.

### Legacy (supervised classification)

[legacy/training.py](legacy/training.py) trains a classifier on DUNE CVN images with cross-entropy loss.

```bash
python legacy/training.py
python legacy/training.py --model_name=attn_default --epochs=20 --lr=1e-3 --batch_size=64
```
### DINO (self-supervised)

[dino/train_dino.py](dino/train_dino.py) trains a sparse UNet backbone with a DINO-style student/teacher self-distillation objective.

```bash
python dino/train_dino.py
python dino/train_dino.py --backbone_name=attn_default --epochs=100 --batch_size=16 --mask_ratio=0.5
python dino/train_dino.py --epochs=2 --batch_size=4 --test_mode=True --debug=True
```
## Data loaders

All loaders live in [loader/](loader/) and cache their file-index to disk on first scan to speed up subsequent runs.

* `DUNEImageDataset` — [loader/dataset.py](loader/dataset.py): Dense DUNE CVN images with neutrino-flavour labels. It returns a single-channel `(1, 500, 500)` float tensor together with an integer label. 

* `APAImageDataset` — [loader/apa_dataset.py](loader/apa_dataset.py): Dense DUNE APA wire-plane images without labels. Returns a `(1, channels, ticks)` float tensor for the requested view (`U`, `V`, or `W`). 

* `APASparseDataset` — [loader/apa_sparse_dataset.py](loader/apa_sparse_dataset.py): Sparse DUNE APA wire-plane images without labels. Reads the same HDF5 structure as `APAImageDataset` but expects a sparse format in the files. Returns a WarpConvNet `Voxels` object (integer coordinates + scalar features).

## Model architectures

Model architectures are available in `./models` and must be declared in the `MODEL_REGISTRY` or `BACKBONE REGISTRY` (defined in `./models/__init__.py`) to be picked up for training.
For example:
```
from .minkunet_attention import MinkUNetSparseAttention

BACKBONE_REGISTRY = {
    # Backbone with sparse attention
    "attn_default":     MinkUNetSparseAttention,
    # add more here
    ...
}
```