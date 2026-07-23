"""
Extract student and teacher features from a trained DINO checkpoint.

Features are collected at *valid* pixels — pixels that are active (non-zero)
in the original image. The output .npz contains per-pixel feature vectors
together with image-level truth (class, vertex, neutrino kinematics),
suitable for PCA analysis and downstream k-NN / probing diagnostics.

Usage:
    python dino/extract_features.py path/to/checkpoint.pt
    python dino/extract_features.py path/to/checkpoint.pt --max_images=5000
    python dino/extract_features.py path/to/checkpoint.pt \
        --output=./my_features.npz --batch_size=16
    python dino/extract_features.py path/to/checkpoint.pt \
        --truth_shards_dir=/path/to/truth_shards

Output (.npz):
    teacher_features       [N_valid, D_bb]   float16   teacher backbone features at valid pixels
    student_features       [N_valid, D_bb]   float16   student backbone features at valid pixels
    teacher_head_features  [N_valid, D_hd]   float16   teacher head features (only if head present)
    student_head_features  [N_valid, D_hd]   float16   student head features (only if head present)
    labels            [N_images]     int64     class label per image (0=numuCC, 1=nueCC, 2=NC, -1=unknown)
    nu_pdg            [N_images]     int64     neutrino pdg code per image
    nu_ccnc           [N_images]     int64     0=CC, 1=NC, -1=unknown
    nu_intType        [N_images]     int64     GENIE interaction type code
    nu_energy         [N_images]     float32   true neutrino energy
    vertex_xyz        [N_images, 3]  float32   true neutrino vertex in detector coords
    event_key         [N_images]     <U...     per-image traceability "{file}:{group}"
    positions         [N_valid, 2]   int32     (channel, tick) pixel coordinates in view-local frame
    charges           [N_valid, 1]   float32   raw pixel charge (ADC value) at each active pixel
    offsets           [N_images+1]   int64     CSR-style: image i occupies rows offsets[i]:offsets[i+1]
    pixel_labels      [N_valid]      int8      per-pixel class label (0=Background/no-truth,
                                               1=Track 2=Shower 3=Michel 4=DeltaRay 5=Blip 6=Other)
                                               only present when extracted with --pixel_truth
"""

import fire
import itertools
import inspect
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_meta_collate_fn
from loader.splits import Subset
from models import BACKBONE_REGISTRY
from dino.config import DINOConfig
from dino.projhead import DINOProjectionHead
from dino.transforms import FeatureLogTransform


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
def _run_loader(student, teacher, loader, device, normalizer=None, student_head=None, teacher_head=None):
    """
    Run both student and teacher (+ optional heads) over the loader.

    Returns a dict of flat arrays — one row per valid (non-zero) pixel for
    pixel-level fields, one row per image for event-level fields.
    """
    s_feats_all, t_feats_all = [], []
    s_head_all,  t_head_all  = [], []
    pos_all, charges_all = [], []
    offsets = [0]

    labels_all, pdg_all, ccnc_all, intType_all, energy_all = [], [], [], [], []
    vertex_all, event_keys_all = [], []
    pixel_labels_all = []

    have_head = student_head is not None

    for xs, meta in loader:
        xs = xs.to(device)

        # Raw pixel charges before normalization
        input_charges = xs.feature_tensor.float().clone()  # [N_active, 1]

        if normalizer is not None:
            xs = normalizer(xs)

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

        B = img_offs.shape[0] - 1
        for b in range(B):
            start = int(img_offs[b])
            end   = int(img_offs[b + 1])
            n     = end - start

            s_feats_all.append(s_feats[start:end].numpy())
            t_feats_all.append(t_feats[start:end].numpy())
            pos_all.append(coords[start:end].numpy())
            charges_all.append(charges[start:end].numpy())
            offsets.append(offsets[-1] + n)

            if have_head:
                s_head_all.append(s_hd[start:end].numpy())
                t_head_all.append(t_hd[start:end].numpy())

        # Event-level metadata (one row per image)
        labels_all.extend(meta["label"].tolist())
        pdg_all.extend(meta["nu_pdg"].tolist())
        ccnc_all.extend(meta["nu_ccnc"].tolist())
        intType_all.extend(meta["nu_intType"].tolist())
        energy_all.extend(meta["nu_energy"].tolist())
        vertex_all.append(meta["vertex_xyz"].numpy())   # [B, 3]
        event_keys_all.extend(meta["event_key"])

        # Optional pixel-level PID truth (list of B arrays, one per image)
        if "pixel_labels" in meta:
            pixel_labels_all.extend(meta["pixel_labels"])

    return {
        "student_features":      np.concatenate(s_feats_all, axis=0).astype(np.float16),
        "teacher_features":      np.concatenate(t_feats_all, axis=0).astype(np.float16),
        "student_head_features": (np.concatenate(s_head_all, axis=0).astype(np.float16) if have_head else None),
        "teacher_head_features": (np.concatenate(t_head_all, axis=0).astype(np.float16) if have_head else None),
        "labels":     np.array(labels_all, dtype=np.int64),
        "nu_pdg":     np.array(pdg_all,    dtype=np.int64),
        "nu_ccnc":    np.array(ccnc_all,   dtype=np.int64),
        "nu_intType": np.array(intType_all, dtype=np.int64),
        "nu_energy":  np.array(energy_all, dtype=np.float32),
        "vertex_xyz": np.concatenate(vertex_all, axis=0).astype(np.float32),
        "event_key":  np.array(event_keys_all),
        "positions":  np.concatenate(pos_all,     axis=0).astype(np.int32),
        "charges":    np.concatenate(charges_all, axis=0).astype(np.float32),
        "offsets":    np.array(offsets, dtype=np.int64),
        "pixel_labels": (np.concatenate(pixel_labels_all, axis=0).astype(np.int8)
                       if pixel_labels_all else None),
    }


def main(
    checkpoint: str,
    output: str = "",
    max_images: int = 2000,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    pixel_truth: bool = False,
    cache_dir: str = "",
    truth_shards_dir: str = "",
):
    """
    Extract DINO features from a trained checkpoint for PCA / probing.

    Args:
        checkpoint:       Path to a .pt checkpoint saved by train_dino.py
        output:           Output .npz path. Defaults to <checkpoint_dir>/features_ep<N>.npz
        max_images:       Max number of images to process (-1 = full dataset)
        batch_size:       Inference batch size
        num_workers:      DataLoader workers (ignored when truth_shards_dir is set)
        device:           "cuda" or "cpu"
        pixel_truth:      If True, also save per-pixel class labels (pixel_labels) from
                          frame_label_1st, enabling pixel-level PID k-NN analysis.
                          Ignored when truth_shards_dir is set (pixel_labels present when
                          the shards were created with --with_pixel_truth).
        cache_dir:        Directory for the dataset index cache. Defaults to ./data.
                          Point this at the same persistent cache used during training
                          to avoid re-scanning the full dataset on every run.
        truth_shards_dir: Path to a truth shard set created by loader/create_shards.py
                          (--n_shards N --with_pixel_truth).
                          When provided, data is loaded from the pre-built shards instead of
                          the original dataset, giving fast sequential I/O regardless of
                          the underlying filesystem. The shard apa/view are asserted against
                          the checkpoint config to catch mismatches early.
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

    if truth_shards_dir:
        from loader.apa_sparse_sharded_dataset import APASparseShardedDataset
        print(f"\nLoading truth shards from {truth_shards_dir} ...")
        # shuffle=False -> deterministic event order, identical across checkpoints.
        dataset = APASparseShardedDataset(
            truth_shards_dir, batch_size=batch_size, shuffle=False,
        )
        if dataset.apa is not None and dataset.apa != cfg.apa:
            raise ValueError(f"Shard apa={dataset.apa} != checkpoint apa={cfg.apa}")
        if dataset.view is not None and dataset.view != cfg.view:
            raise ValueError(f"Shard view={dataset.view!r} != checkpoint view={cfg.view!r}")
        n_images = len(dataset) * batch_size
        print(f"  apa={dataset.apa}  view={dataset.view}  images={n_images}")
    else:
        # Dataset (sparse, with full event metadata)
        print(f"\nLoading dataset from {cfg.datadir} ...")
        dataset_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        dataset = APASparseMetaDataset(
            datadir=cfg.datadir,
            apa=cfg.apa,
            view=cfg.view,
            use_cache=True,
            return_pixel_truth=pixel_truth,
            **dataset_kwargs,
        )
        if 0 < max_images < len(dataset):
            indices = torch.randperm(len(dataset))[:max_images]
            dataset = Subset(dataset, indices)
        print(f"  Images to process: {len(dataset)}")

    if truth_shards_dir:
        # Reader yields pre-batched (voxels, meta); cap batches for max_images.
        loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
        if 0 < max_images < len(dataset) * batch_size:
            max_batches = max_images // batch_size
            print(f"  Limiting to first {max_batches * batch_size} images ({max_batches} batches)")
            loader = itertools.islice(iter(loader), max_batches)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=voxels_meta_collate_fn,
        )

    # Load both models
    print("\nLoading student and teacher backbones ...")
    student = _load_backbone(ckpt, "student", device)
    teacher = _load_backbone(ckpt, "teacher", device)

    student_head = _load_head(ckpt, "student_head", device)
    teacher_head = _load_head(ckpt, "teacher_head", device)
    if student_head is not None:
        print(f"  Projection head found: {cfg.feature_dim}→{cfg.proj_head_output_dim}D")

    # Data normalizer (must match training config)
    normalizer = FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val) if cfg.use_log_transform else None
    if normalizer is not None:
        print(f"  Log-transform: min={cfg.feat_min_val}, max={cfg.feat_max_val}")
    else:
        print("  No normalization (use_log_transform=False in cfg)")

    # Extract
    print("Extracting features ...")
    results = _run_loader(student, teacher, loader, device, normalizer, student_head, teacher_head)

    print(f"  Images:        {len(results['labels'])}")
    print(f"  Valid pixels:  {results['student_features'].shape[0]}")
    print(f"  Backbone dim:  {results['student_features'].shape[1]}")
    if results["student_head_features"] is not None:
        print(f"  Head dim:      {results['student_head_features'].shape[1]}")

    arrays = dict(
        student_features=results["student_features"],
        teacher_features=results["teacher_features"],
        labels=results["labels"],
        nu_pdg=results["nu_pdg"],
        nu_ccnc=results["nu_ccnc"],
        nu_intType=results["nu_intType"],
        nu_energy=results["nu_energy"],
        vertex_xyz=results["vertex_xyz"],
        event_key=results["event_key"],
        positions=results["positions"],
        charges=results["charges"],
        offsets=results["offsets"],
    )
    if results["student_head_features"] is not None:
        arrays["student_head_features"] = results["student_head_features"]
        arrays["teacher_head_features"] = results["teacher_head_features"]
    if results["pixel_labels"] is not None:
        arrays["pixel_labels"] = results["pixel_labels"]
        print(f"  Pixels with truth label: {(results['pixel_labels'] != 0).sum()}")

    np.savez_compressed(out_path, **arrays)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
