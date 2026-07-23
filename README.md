# ml-dune-model

## Installation

WarpConvNet requires Python >= 3.9 and a working CUDA toolchain.
See: [WarpConvNet/installation](https://nvlabs.github.io/WarpConvNet/getting_started/installation/).

1. Create a Python/`uv` environment to install packages.

2. Install dependencies and WarpConvNet. Note: other packages might need to be installed to resolve build errors.

```
export CUDA=cu128
export TORCH_REL="2.10"
export WARPCONV_REL="1.7.3"

uv pip install torch==${TORCH_REL} torchvision --index-url https://download.pytorch.org/whl/${CUDA}
uv pip install build ninja
uv pip install torch-scatter --find-links https://data.pyg.org/whl/torch-${TORCH_REL}+${CUDA}.html
uv pip install "warpconvnet==${WARPCONV_REL}+torch${TORCH_REL}${CUDA}" /
    --find-links https://github.com/NVlabs/WarpConvNet/releases/latest/download/
uv pip install wheel setuptools psutil
MAX_JOBS=2 uv pip install flash-attn --no-build-isolation     # reduce parallelism to help building
```

3. Install additional packages as required for running. E.g.:

```
uv pip install fire h5py
```

## Training

[dino/train_dino.py](dino/train_dino.py) trains a sparse UNet backbone with a DINO-style student/teacher self-distillation objective.
The script uses [python-fire](https://github.com/google/python-fire) for CLI argument parsing, so any parameter in `main()` can be overridden by passing `--param=value` on the command line.

```bash
python dino/train_dino.py
python dino/train_dino.py --backbone_name=attn_default --epochs=100 --batch_size=16 --mask_ratio=0.5
python dino/train_dino.py --epochs=2 --batch_size=4 --test_mode=True --debug=True
```

## Data loaders

Dataset classes and preprocessing scripts live in [loader/](loader/).
See [loader/README.md](loader/README.md) for the available dataset classes and the locations of the dataset productions.

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
