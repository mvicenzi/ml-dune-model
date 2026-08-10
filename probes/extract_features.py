"""
Extract features from one branch of a trained DINO checkpoint.

Features are collected at *valid* pixels — pixels that are active (non-zero)
in the original image. The output .npz contains per-pixel feature vectors
together with image-level truth (class, vertex, neutrino kinematics),
suitable for PCA analysis and downstream k-NN / probing diagnostics.

One branch per file: `--source=student` (default) or `--source=teacher`, never
both. Each branch costs ~1.4 GB of backbone features per 2000 events (plus ~2.8 GB
of head features when a projection head is present) and no probe reads both, so
writing both doubled every file for nothing. Run the tool twice, into two output
paths, if a student/teacher comparison is actually wanted.

Usage:
    python -m probes.extract_features path/to/checkpoint.pt
    python -m probes.extract_features path/to/checkpoint.pt --max_images=5000
    python -m probes.extract_features path/to/checkpoint.pt --source=teacher \
        --output=./teacher_features.npz --batch_size=16
    python -m probes.extract_features path/to/checkpoint.pt \
        --truth_shards_dir=/path/to/truth_shards

Output (.npz), where <src> is the extracted branch:
    <src>_features         [N_valid, D_bb]   float16   backbone features at valid pixels
                                                       (backbone output, NOT passed
                                                       through the projection head)
    <src>_head_features    [N_valid, D_hd]   float16   projection-head features, written
                                                       ONLY with --head_features; the
                                                       head being in the checkpoint is
                                                       not enough
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
    pixel_energyfrac  [N_valid]      float32   leading contributor's energy share (→ overlap score)
    pixel_trackid     [N_valid]      int32     truth track id, signed (→ instance id via abs())
    pixel_truth_q     [N_valid]      float32   truth deposited charge (electrons)
                                               the three above only with --extra_truth

Provenance scalars (always written; consumed by probes/features.py so a metric
can never be mis-scored against the wrong charge transform):
    epoch, backbone_name, encoding_range, feature_dim, apa, view,
    use_log_transform, feat_min_val, feat_max_val, backbone_kwargs_applied,
    extraction_source ("raw" | "shards" | "packed")
"""

import fire
import itertools
import inspect
import json
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

# Per-pixel truth tiers and their stored dtypes. pixel_labels comes with
# --pixel_truth; the rest need --extra_truth (all three readers support both).
PIXEL_TRUTH_KEYS = {
    "pixel_labels":     np.int8,
    "pixel_energyfrac": np.float32,
    "pixel_trackid":    np.int32,
    "pixel_truth_q":    np.float32,
}


def _load_backbone(ckpt: dict, key: str, device: torch.device):
    """Rebuild a backbone from a checkpoint, exactly as training built it.

    The kwarg filter below is deliberately the same single-level
    `inspect.signature` check `dino/model.py` uses, so a rebuilt backbone is the
    architecture that was actually trained rather than the one the config
    describes. Those differ today: MAE backbones declare `__init__(self, **kw)`,
    which `inspect.signature` cannot see through, so `encoding_range` is dropped
    on both sides and training silently used the hardcoded default. Reproducing
    that faithfully is the point — "fixing" it here alone would score a network
    against a positional encoding it was never trained with.

    For the encoding specifically this is belt-and-braces: warpconvnet's
    SinusoidalEncoding consumes data_range only to build its `freqs` buffer, and
    that buffer is persistent, so load_state_dict restores the trained table
    regardless. The strict load below is what actually guarantees the rebuild
    matches: any architectural mismatch raises instead of leaving random weights.

    Returns (model, applied_kwargs) — the kwargs are recorded in the output .npz.
    """
    cfg = ckpt["cfg"]
    backbone_cls = BACKBONE_REGISTRY[cfg.backbone_name]
    backbone_kwargs = {}
    if "encoding_range" in inspect.signature(backbone_cls.__init__).parameters:
        backbone_kwargs["encoding_range"] = cfg.encoding_range
    model = backbone_cls(**backbone_kwargs).to(device)
    model.load_state_dict(ckpt[key])          # strict: mismatches must be loud
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, backbone_kwargs


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
def _run_loader(backbone, source, loader, device, normalizer=None, head=None):
    """
    Run ONE branch (student or teacher) + its optional head over the loader.

    Only one branch is extracted: the two sets of features are ~1.4 GB each per
    2000 events and no probe reads both, so writing both doubled every feature
    file for nothing. `source` names the branch and picks the output keys.

    Returns a dict of flat arrays — one row per valid (non-zero) pixel for
    pixel-level fields, one row per image for event-level fields.
    """
    feats_all, head_all = [], []
    pos_all, charges_all = [], []
    offsets = [0]

    labels_all, pdg_all, ccnc_all, intType_all, energy_all = [], [], [], [], []
    vertex_all, event_keys_all = [], []
    pixel_truth_all = {k: [] for k in PIXEL_TRUTH_KEYS}

    have_head = head is not None

    for xs, meta in loader:
        xs = xs.to(device)

        # Raw pixel charges before normalization
        input_charges = xs.feature_tensor.float().clone()  # [N_active, 1]

        if normalizer is not None:
            xs = normalizer(xs)

        out_v = backbone(xs)                            # Voxels [N_active, D_bb]

        # Per-pixel truth is CSR-aligned to the INPUT voxels, and features are
        # sliced by the input offsets below, so output row i must still describe
        # input pixel i. That holds by construction for these backbones — the
        # decoder's transposed convs are geometry-guided (ConvTrBlock2D upsamples
        # onto the skip tensor's coordinates) and the full-resolution skip is the
        # stride-1 stem, so the active set and its order are the input's. The
        # check costs an int compare per batch and converts a future generative or
        # pruning layer from silently mislabelled features into a loud failure.
        for name, out in ((source, out_v),):
            if not torch.equal(out.offsets, xs.offsets):
                raise RuntimeError(
                    f"{name} backbone changed the per-event voxel counts "
                    f"({out.offsets.tolist()[:5]}... vs input "
                    f"{xs.offsets.tolist()[:5]}...). Per-pixel truth alignment is "
                    f"no longer positional; extraction must join on coordinates."
                )
            if not torch.equal(out.coordinate_tensor, xs.coordinate_tensor):
                n_diff = int((out.coordinate_tensor != xs.coordinate_tensor).any(1).sum())
                raise RuntimeError(
                    f"{name} backbone reordered or moved {n_diff} voxel coordinates. "
                    f"Per-pixel truth alignment is no longer positional; extraction "
                    f"must join features to truth on (channel, tick)."
                )

        coords   = xs.coordinate_tensor.cpu()              # [N_active, 2]
        charges  = input_charges.cpu()                     # [N_active, 1]
        feats    = out_v.feature_tensor.float().cpu()       # [N_active, D_bb]
        img_offs = xs.offsets.cpu()                         # [B+1]

        if have_head:
            hd = head(out_v).feature_tensor.float().cpu()   # [N_active, D_hd]

        B = img_offs.shape[0] - 1
        for b in range(B):
            start = int(img_offs[b])
            end   = int(img_offs[b + 1])
            n     = end - start

            # Cast to the on-disk dtype here rather than after the concatenate.
            # The rounding is elementwise, so the result is bit-identical either
            # way, but accumulating float32 costs 2x and then peaks at 5x the
            # output while the list, the concatenated copy and the cast copy are
            # all live: 35 GB of features alone at 10000 events, against a 32 GB
            # condor request. Casting per slice keeps the peak at ~14 GB.
            feats_all.append(feats[start:end].numpy().astype(np.float16))
            pos_all.append(coords[start:end].numpy())
            charges_all.append(charges[start:end].numpy())
            offsets.append(offsets[-1] + n)

            if have_head:
                head_all.append(hd[start:end].numpy().astype(np.float16))

        # Event-level metadata (one row per image)
        labels_all.extend(meta["label"].tolist())
        pdg_all.extend(meta["nu_pdg"].tolist())
        ccnc_all.extend(meta["nu_ccnc"].tolist())
        intType_all.extend(meta["nu_intType"].tolist())
        energy_all.extend(meta["nu_energy"].tolist())
        vertex_all.append(meta["vertex_xyz"].numpy())   # [B, 3]
        event_keys_all.extend(meta["event_key"])

        # Optional per-pixel truth tiers (each a list of B arrays, one per image)
        for key in PIXEL_TRUTH_KEYS:
            if key in meta:
                pixel_truth_all[key].extend(meta[key])

    out = {
        # Already float16 per slice (see the cast in the loop above).
        f"{source}_features":      np.concatenate(feats_all, axis=0),
        f"{source}_head_features": (np.concatenate(head_all, axis=0)
                                    if have_head else None),
        "source": source,
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
    }
    for key, dtype in PIXEL_TRUTH_KEYS.items():
        parts = pixel_truth_all[key]
        out[key] = np.concatenate(parts, axis=0).astype(dtype) if parts else None
    return out


def main(
    checkpoint: str,
    output: str = "",
    max_images: int = 10000,
    source: str = "student",
    head_features: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    pixel_truth: bool = False,
    extra_truth: bool = False,
    cache_dir: str = "",
    truth_shards_dir: str = "",
    packed_path: str = "",
    output_prefix: str = "features_",
):
    """
    Extract DINO features from a trained checkpoint for PCA / probing.

    Args:
        checkpoint:       Path to a .pt checkpoint saved by train_dino.py
        output:           Output .npz path. Defaults to <checkpoint_dir>/features_ep<N>.npz
        max_images:       Max number of images to process (-1 = full dataset).
                          10000 by default: at 2000 the pools behind the rare
                          motifs came from too few events (Michel from 113 of
                          397 validation events), which set the noise floor of
                          every per-class average. Note the sharded reader
                          rounds down to a whole number of batches, so 10000
                          with batch_size=32 extracts 9984.
        source:           Which branch to extract, "student" or "teacher" — one
                          per file, never both. They are ~1.4 GB each per 2000
                          events and no probe reads both; run the tool twice if
                          a comparison is genuinely needed.
        head_features:    Also save the projection-head features. Off by default:
                          they are 2x the size of the backbone features (~2.8 GB
                          per 2000 events) and no probe reads them — only the
                          legacy dino/diagnostics/plot_features.py does. At
                          max_images=10000 they add ~14 GB per checkpoint and the
                          extraction needs ~64 GB of RAM rather than ~32 GB.
        batch_size:       Inference batch size
        num_workers:      DataLoader workers (ignored when truth_shards_dir is set)
        device:           "cuda" or "cpu"
        pixel_truth:      If True, also save per-pixel class labels (pixel_labels) from
                          frame_label_1st, enabling pixel-level PID k-NN analysis.
                          With a shard set or pack, the container must have been
                          built with the matching --with_pixel_truth flag.
        extra_truth:      If True, additionally save pixel_energyfrac (→ overlap
                          score), pixel_trackid (→ instance id) and pixel_truth_q
                          (→ truth charge). Implies pixel_truth. These are what the
                          instance / charge / overlap-strata probes need; the
                          container must carry them (--with_extra_truth).
        cache_dir:        Directory for the dataset index cache. Defaults to ./data.
                          Point this at the same persistent cache used during training
                          to avoid re-scanning the full dataset on every run.
        truth_shards_dir: Path to a truth shard set created by loader/create_shards.py
                          (--with_pixel_truth; the full training shard set —
                          shuffle=False + max_images gives a deterministic
                          subset, identical across checkpoints).
                          When provided, data is loaded from the pre-built shards instead of
                          the original dataset, giving fast sequential I/O regardless of
                          the underlying filesystem. The shard apa/view are asserted against
                          the checkpoint config to catch mismatches early.
        packed_path:      Path to a packed .npz built by loader/pack_dataset.py. Mutually
                          exclusive with truth_shards_dir; same deterministic seeded
                          subsetting as the raw path, so the event set matches across
                          checkpoints.
        output_prefix:    Basename prefix for the default output name
                          (<prefix>ep<N>.npz). Use it to extract a second family
                          of feature files alongside existing ones — e.g.
                          --output_prefix=features_probe_ when adding the extra
                          truth tiers without overwriting earlier extractions.
                          Ignored when `output` is given explicitly.
    """
    if source not in ("student", "teacher"):
        raise ValueError(f'source must be "student" or "teacher", got {source!r}')
    if extra_truth:
        pixel_truth = True
    if truth_shards_dir and packed_path:
        raise ValueError("truth_shards_dir and packed_path are mutually exclusive")
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
        output = str(ckpt_path.parent / f"{output_prefix}ep{epoch}.npz")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if truth_shards_dir:
        from loader.apa_sparse_sharded_dataset import APASparseShardedDataset
        print(f"\nLoading truth shards from {truth_shards_dir} ...")
        # shuffle=False -> deterministic event order, identical across checkpoints.
        dataset = APASparseShardedDataset(
            truth_shards_dir, batch_size=batch_size, shuffle=False,
            return_pixel_truth=True, return_extra_truth=extra_truth,
        )
        if dataset.apa is not None and dataset.apa != cfg.apa:
            raise ValueError(f"Shard apa={dataset.apa} != checkpoint apa={cfg.apa}")
        if dataset.view is not None and dataset.view != cfg.view:
            raise ValueError(f"Shard view={dataset.view!r} != checkpoint view={cfg.view!r}")
        n_images = len(dataset) * batch_size
        print(f"  apa={dataset.apa}  view={dataset.view}  images={n_images}")
    elif packed_path:
        from loader.apa_packed_dataset import APAPackedDataset
        print(f"\nLoading packed dataset from {packed_path} ...")
        dataset = APAPackedDataset(
            packed_path, return_pixel_truth=pixel_truth, return_extra_truth=extra_truth,
        )
        if dataset.apa is not None and dataset.apa != cfg.apa:
            raise ValueError(f"Pack apa={dataset.apa} != checkpoint apa={cfg.apa}")
        if dataset.view is not None and dataset.view != cfg.view:
            raise ValueError(f"Pack view={dataset.view!r} != checkpoint view={cfg.view!r}")
        print(f"  apa={dataset.apa}  view={dataset.view}  events={len(dataset)}")
        if 0 < max_images < len(dataset):
            # Same seed as the raw path: identical event set across checkpoints.
            rng = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(dataset), generator=rng)[:max_images]
            dataset = Subset(dataset, indices)
        print(f"  Images to process: {len(dataset)}")
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
            return_extra_truth=extra_truth,
            **dataset_kwargs,
        )
        if 0 < max_images < len(dataset):
            # Seeded: every extraction samples the same events, so features
            # are comparable across checkpoints.
            rng = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(dataset), generator=rng)[:max_images]
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

    print(f"\nLoading {source} backbone ...")
    backbone, applied_kwargs = _load_backbone(ckpt, source, device)
    print(f"  Backbone kwargs applied: {applied_kwargs}")
    if "encoding_range" not in applied_kwargs:
        print(f"  Note: cfg.encoding_range={cfg.encoding_range} is not a ctor kwarg for "
              f"{cfg.backbone_name}; the positional encoding comes from the "
              f"checkpoint's freqs buffer (see _load_backbone).")

    # The projection head is extracted ONLY on explicit request. Presence in the
    # checkpoint is not consent: every DINO (and our MAE) checkpoint carries one,
    # and its features are 2x the size of the backbone's with no probe reading
    # them. Asking for a head that is not in the checkpoint is an error, not a
    # silent no-op — otherwise `--head_features` could appear to work and produce
    # a file without them.
    head = None
    if head_features:
        head_key = f"{source}_head"
        head = _load_head(ckpt, head_key, device)
        if head is None:
            raise SystemExit(
                f"--head_features was requested but {ckpt_path.name} has no "
                f"{head_key!r} (checkpoint keys: {sorted(ckpt.keys())}). Extract "
                f"without --head_features, or point at a checkpoint trained with "
                f"a projection head."
            )
        print(f"  Projection head: extracting, "
              f"{cfg.feature_dim}→{cfg.proj_head_output_dim}D "
              f"(--head_features set)")
    else:
        in_ckpt = "present in checkpoint" if f"{source}_head" in ckpt else "absent"
        print(f"  Projection head: NOT extracted ({in_ckpt}; pass --head_features "
              f"to include it)")

    # Data normalizer (must match training config)
    normalizer = FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val) if cfg.use_log_transform else None
    if normalizer is not None:
        print(f"  Log-transform: min={cfg.feat_min_val}, max={cfg.feat_max_val}")
    else:
        print("  No normalization (use_log_transform=False in cfg)")

    # Extract
    print("Extracting features ...")
    results = _run_loader(backbone, source, loader, device, normalizer, head)

    print(f"  Images:        {len(results['labels'])}")
    print(f"  Valid pixels:  {results[f'{source}_features'].shape[0]}")
    print(f"  Backbone dim:  {results[f'{source}_features'].shape[1]}")
    if results[f"{source}_head_features"] is not None:
        print(f"  Head dim:      {results[f'{source}_head_features'].shape[1]}")

    arrays = {
        f"{source}_features": results[f"{source}_features"],
        "labels": results["labels"],
        "nu_pdg": results["nu_pdg"],
        "nu_ccnc": results["nu_ccnc"],
        "nu_intType": results["nu_intType"],
        "nu_energy": results["nu_energy"],
        "vertex_xyz": results["vertex_xyz"],
        "event_key": results["event_key"],
        "positions": results["positions"],
        "charges": results["charges"],
        "offsets": results["offsets"],
    }
    if results[f"{source}_head_features"] is not None:
        arrays[f"{source}_head_features"] = results[f"{source}_head_features"]
    for key in PIXEL_TRUTH_KEYS:
        if results[key] is not None:
            arrays[key] = results[key]
    if results["pixel_labels"] is not None:
        print(f"  Pixels with truth label: {(results['pixel_labels'] != 0).sum()}")
    if results["pixel_truth_q"] is not None:
        print(f"  Pixels with truth charge: {(results['pixel_truth_q'] > 0).sum()}")

    # Provenance: what produced these features. The probes read this to score the
    # raw floor with the charge transform the backbone was actually fed, and to
    # keep results self-describing when JSONs from many runs are compared.
    arrays.update(
        epoch=np.array(epoch),
        backbone_name=np.array(cfg.backbone_name),
        encoding_range=np.array(cfg.encoding_range),
        feature_dim=np.array(cfg.feature_dim),
        apa=np.array(cfg.apa),
        view=np.array(cfg.view),
        use_log_transform=np.array(bool(cfg.use_log_transform)),
        feat_min_val=np.array(cfg.feat_min_val),
        feat_max_val=np.array(cfg.feat_max_val),
        backbone_kwargs_applied=np.array(json.dumps(
            {k: str(v) for k, v in applied_kwargs.items()})),
        extraction_source=np.array(
            "shards" if truth_shards_dir else ("packed" if packed_path else "raw")),
        checkpoint_path=np.array(str(ckpt_path)),
        source=np.array(source),
    )

    np.savez_compressed(out_path, **arrays)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
