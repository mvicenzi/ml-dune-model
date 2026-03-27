"""
Extract student and teacher features from a trained DINO checkpoint.

Features are collected at *valid* pixels — pixels that are active (non-zero)
in the original image. The output .npz contains per-pixel feature vectors
together with image-level class labels, suitable for PCA analysis along the
active directions of the covariance matrix.

Usage:
    python dino/extract_features.py path/to/checkpoint.pt
    python dino/extract_features.py path/to/checkpoint.pt --max_images=5000
    python dino/extract_features.py path/to/checkpoint.pt \
        --output=./my_features.npz --batch_size=16

Output (.npz):
    teacher_features  [N_valid, D]   float32   teacher features at valid pixels
    student_features  [N_valid, D]   float32   student features at valid pixels
    labels            [N_images]     int64     class label per image
    positions         [N_valid, 2]   int32     (row, col) pixel coordinates
    offsets           [N_images+1]   int64     CSR-style: image i occupies rows offsets[i]:offsets[i+1]
"""

import fire
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from loader.dataset import DUNEImageDataset
from loader.splits import Subset
from models import BACKBONE_REGISTRY
from dino.config import DINOConfig
from warpconvnet.geometry.types.voxels import Voxels


def _load_backbone(ckpt: dict, key: str, device: torch.device):
    cfg = ckpt["cfg"]
    model = BACKBONE_REGISTRY[cfg.backbone_name]().to(device)
    model.load_state_dict(ckpt[key])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _run_loader(student, teacher, loader, device):
    """
    Run both student and teacher over the loader.

    Returns flat arrays: student_features, teacher_features, labels,
    positions — one row per valid (non-zero) pixel across the whole dataset.
    """
    s_feats_all, t_feats_all, labels_all, pos_all, offsets = [], [], [], [], [0]

    for images, labels in loader:
        images = images.to(device)          # [B, 1, H, W]

        # Convert dense → sparse; coords are (row, col) for active pixels
        xs = Voxels.from_dense(images)      # Voxels: N_active voxels across batch

        s_out = student(xs)                 # Voxels [N_active, D]
        t_out = teacher(xs)                 # Voxels [N_active, D]

        coords   = xs.coordinate_tensor.cpu()              # [N_active, 2]
        s_feats  = s_out.feature_tensor.float().cpu()      # [N_active, D]
        t_feats  = t_out.feature_tensor.float().cpu()      # [N_active, D]
        img_offs = xs.offsets.cpu()                        # [B+1]

        for b in range(images.shape[0]):
            start = int(img_offs[b])
            end   = int(img_offs[b + 1])
            n     = end - start

            s_feats_all.append(s_feats[start:end].numpy())   # [n, D]
            t_feats_all.append(t_feats[start:end].numpy())
            pos_all.append(coords[start:end].numpy())         # [n, 2]
            labels_all.append(labels[b].item())
            offsets.append(offsets[-1] + n)

    return (
        np.concatenate(s_feats_all, axis=0).astype(np.float32),
        np.concatenate(t_feats_all, axis=0).astype(np.float32),
        np.array(labels_all, dtype=np.int64),
        np.concatenate(pos_all,     axis=0).astype(np.int32),
        np.array(offsets,           dtype=np.int64),
    )


def main(
    checkpoint: str,
    output: str = "",
    max_images: int = 5000,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
):
    """
    Extract DINO features from a trained checkpoint for PCA / probing.

    Args:
        checkpoint:  Path to a .pt checkpoint saved by train_dino.py
        output:      Output .npz path. Defaults to <checkpoint_dir>/features_ep<N>.npz
        max_images:  Max number of images to process (-1 = full dataset)
        batch_size:  Inference batch size
        num_workers: DataLoader workers
        device:      "cuda" or "cpu"
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    with torch.serialization.safe_globals([DINOConfig]):
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}  backbone={cfg.backbone_name}  feature_dim={cfg.feature_dim}")

    if not output:
        output = str(ckpt_path.parent / f"features_ep{epoch}.npz")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset
    print(f"\nLoading dataset from {cfg.rootdir} ...")
    dataset = DUNEImageDataset(
        rootdir=cfg.rootdir,
        class_names=["numu", "nue", "nutau", "NC"],
        view_index=cfg.view_index,
        use_cache=True,
    )
    if 0 < max_images < len(dataset):
        indices = torch.randperm(len(dataset))[:max_images]
        dataset = Subset(dataset, indices)
    print(f"  Images to process: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load both models
    print("\nLoading student and teacher backbones ...")
    student = _load_backbone(ckpt, "student", device)
    teacher = _load_backbone(ckpt, "teacher", device)

    # Extract
    print("Extracting features ...")
    s_feats, t_feats, labels, positions, offsets = _run_loader(student, teacher, loader, device)

    print(f"  Images:      {len(labels)}")
    print(f"  Valid pixels: {s_feats.shape[0]}")
    print(f"  Feature dim:  {s_feats.shape[1]}")

    np.savez_compressed(
        out_path,
        student_features=s_feats,
        teacher_features=t_feats,
        labels=labels,
        positions=positions,
        offsets=offsets,
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
