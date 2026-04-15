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
    teacher_features       [N_valid, D_bb]   float32   teacher backbone features at valid pixels
    student_features       [N_valid, D_bb]   float32   student backbone features at valid pixels
    teacher_head_features  [N_valid, D_hd]   float32   teacher head features (only if head present)
    student_head_features  [N_valid, D_hd]   float32   student head features (only if head present)
    labels            [N_images]     int64     class label per image
    positions         [N_valid, 2]   int32     (row, col) pixel coordinates
    charges           [N_valid, 1]   float32   raw pixel charge (ADC value) at each active pixel
    offsets           [N_images+1]   int64     CSR-style: image i occupies rows offsets[i]:offsets[i+1]
"""

import fire
import inspect
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from loader.dataset import DUNEImageDataset
from loader.splits import Subset
from models import BACKBONE_REGISTRY
from dino.config import DINOConfig
from dino.projhead import DINOProjectionHead
from warpconvnet.geometry.types.voxels import Voxels


def _load_backbone(ckpt: dict, key: str, device: torch.device):
    cfg = ckpt["cfg"]
    backbone_cls = BACKBONE_REGISTRY[cfg.backbone_name]
    backbone_kwargs = {}
    if "encoding_range" in inspect.signature(backbone_cls.__init__).parameters:
        backbone_kwargs["encoding_range"] = cfg.encoding_range
    model = backbone_cls(**backbone_kwargs).to(device)
    model.load_state_dict(ckpt[key])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_head(ckpt: dict, key: str, device: torch.device):
    """Load a projection head from checkpoint; returns None if key absent."""
    if key not in ckpt:
        return None
    cfg = ckpt["cfg"]
    head = DINOProjectionHead(
        in_dim=cfg.feature_dim,
        hidden_dim=cfg.proj_head_hidden_dim,
        out_dim=cfg.proj_head_output_dim,
        n_layers=cfg.proj_head_n_layers,
    ).to(device)
    head.load_state_dict(ckpt[key])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


@torch.no_grad()
def _run_loader(student, teacher, loader, device, student_head=None, teacher_head=None):
    """
    Run both student and teacher (+ optional heads) over the loader.

    Returns flat arrays: student_features, teacher_features,
    student_head_features (or None), teacher_head_features (or None),
    labels, positions, charges — one row per valid (non-zero) pixel.
    """
    s_feats_all, t_feats_all, labels_all, pos_all, charges_all, offsets = [], [], [], [], [], [0]
    s_head_all, t_head_all = [], []
    have_head = student_head is not None

    for images, labels in loader:
        images = images.to(device)          # [B, 1, H, W]

        # Convert dense → sparse; coords are (row, col) for active pixels
        xs = Voxels.from_dense(images)                  # Voxels: N_active voxels across batch

        # Grab raw pixel charges before the model transforms the features
        input_charges = xs.feature_tensor.float()       # [N_active, 1]

        s_out = student(xs)                             # Voxels [N_active, D_bb]
        t_out = teacher(xs)                             # Voxels [N_active, D_bb]

        coords   = xs.coordinate_tensor.cpu()              # [N_active, 2]
        charges  = input_charges.cpu()                     # [N_active, 1]
        s_feats  = s_out.feature_tensor.float().cpu()      # [N_active, D_bb]
        t_feats  = t_out.feature_tensor.float().cpu()      # [N_active, D_bb]
        img_offs = xs.offsets.cpu()                        # [B+1]

        if have_head:
            s_hd = student_head(s_out).feature_tensor.float().cpu()  # [N_active, D_hd]
            t_hd = teacher_head(t_out).feature_tensor.float().cpu()

        for b in range(images.shape[0]):
            start = int(img_offs[b])
            end   = int(img_offs[b + 1])
            n     = end - start

            s_feats_all.append(s_feats[start:end].numpy())
            t_feats_all.append(t_feats[start:end].numpy())
            pos_all.append(coords[start:end].numpy())
            charges_all.append(charges[start:end].numpy())
            labels_all.append(labels[b].item())
            offsets.append(offsets[-1] + n)

            if have_head:
                s_head_all.append(s_hd[start:end].numpy())
                t_head_all.append(t_hd[start:end].numpy())

    return (
        np.concatenate(s_feats_all,  axis=0).astype(np.float32),
        np.concatenate(t_feats_all,  axis=0).astype(np.float32),
        np.concatenate(s_head_all,   axis=0).astype(np.float32) if have_head else None,
        np.concatenate(t_head_all,   axis=0).astype(np.float32) if have_head else None,
        np.array(labels_all, dtype=np.int64),
        np.concatenate(pos_all,      axis=0).astype(np.int32),
        np.concatenate(charges_all,  axis=0).astype(np.float32),
        np.array(offsets,            dtype=np.int64),
    )


def main(
    checkpoint: str,
    output: str = "",
    max_images: int = 10000,
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

    student_head = _load_head(ckpt, "student_head", device)
    teacher_head = _load_head(ckpt, "teacher_head", device)
    if student_head is not None:
        print(f"  Projection head found: {cfg.feature_dim}→{cfg.proj_head_output_dim}D")

    # Extract
    print("Extracting features ...")
    s_feats, t_feats, s_head_feats, t_head_feats, labels, positions, charges, offsets = _run_loader(
        student, teacher, loader, device, student_head, teacher_head
    )

    print(f"  Images:        {len(labels)}")
    print(f"  Valid pixels:  {s_feats.shape[0]}")
    print(f"  Backbone dim:  {s_feats.shape[1]}")
    if s_head_feats is not None:
        print(f"  Head dim:      {s_head_feats.shape[1]}")

    arrays = dict(
        student_features=s_feats,
        teacher_features=t_feats,
        labels=labels,
        positions=positions,
        charges=charges,
        offsets=offsets,
    )
    if s_head_feats is not None:
        arrays["student_head_features"] = s_head_feats
        arrays["teacher_head_features"] = t_head_feats

    np.savez_compressed(out_path, **arrays)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
