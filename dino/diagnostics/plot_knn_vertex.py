"""
Vertex-focused k-NN analysis of DINO features.

For each image the neutrino interaction vertex is estimated in two steps:
  1. Restrict active pixels to a coarse **search region** around the known
     approximate vertex position (time ≈ 0, wire ≈ 250).
  2. Within that region, pick the pixel with the highest count of active
     neighbours (from the whole image) within radius `--density_r`.

An evaluation box of size `(2*box_h) × (2*box_w)` is centred on that refined
vertex.  Two analyses are run on those box pixels:

  Image-level (vertex pool):
    Mean-pool the box pixels into one vector per image, then run k-NN exactly
    as in the standard image-level analysis.  This is a focused alternative to
    the global mean-pool.

  Pixel-level (vertex pixels):
    Use each individual pixel inside the box as a sample (inheriting the parent
    image label), then run k-NN purity / confusion / scatter.

The k-NN similarity is computed in batches on GPU (or CPU) via PyTorch matmuls,
so the full N×N matrix is never materialised in memory.

Produces (in --out_dir):
  knn_vertex_image_purity.png    — image-level purity (vertex mean-pool)
  knn_vertex_image_confusion.png — image-level confusion matrix
  knn_vertex_image_scatter.png   — image-level 2-D scatter
  knn_vertex_pixel_purity.png    — pixel-level purity
  knn_vertex_pixel_confusion.png — pixel-level confusion matrix
  knn_vertex_pixel_scatter.png   — pixel-level 2-D scatter

Usage:
    python -m dino.diagnostics.plot_knn_vertex path/to/features_ep10.npz
    python -m dino.diagnostics.plot_knn_vertex path/to/features_ep10.npz \\
        --search_h=80 --search_w=100 --density_r=10 \\
        --box_h=30 --box_w=30 --n_images=2000 --device=cuda
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

# Re-use plot functions from the sibling module
from dino.diagnostics.plot_knn import (
    CLASS_NAMES,
    _reduce_2d,
    plot_purity,
    plot_confusion,
    plot_scatter,
)


# ---------------------------------------------------------------------------
# Vertex finding
# ---------------------------------------------------------------------------

def _find_vertex(
    pos: np.ndarray,
    search_row: int,
    search_col: int,
    search_h: int,
    search_w: int,
    density_r: float,
) -> tuple:
    """
    Find the local density maximum inside the search region.

    pos        : [N, 2] float32  (row, col) of all active pixels in one image
    search_row : top row of the search region
    search_col : centre column of the search region
    search_h   : height of the search region  (rows [search_row, search_row+search_h))
    search_w   : half-width of the search region (cols [search_col±search_w))
    density_r  : neighbour-counting radius (pixels, L2)

    Returns (vertex_row, vertex_col) as floats.
    Falls back to (search_row, search_col) if the search region is empty.
    """
    rows, cols = pos[:, 0], pos[:, 1]
    mask = (
        (rows >= search_row) & (rows < search_row + search_h) &
        (cols >= search_col - search_w) & (cols < search_col + search_w)
    )
    candidates = pos[mask]   # [C, 2]

    if len(candidates) == 0:
        return float(search_row), float(search_col)

    # Count neighbours from the full image within density_r
    diff   = candidates[:, None, :] - pos[None, :, :]                  # [C, N, 2]
    counts = (np.linalg.norm(diff, axis=-1) < density_r).sum(axis=1)  # [C]

    best = candidates[counts.argmax()]
    return float(best[0]), float(best[1])


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect(
    s_feats: np.ndarray,
    t_feats: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    labels: np.ndarray,
    n_images: int,
    search_row: int,
    search_col: int,
    search_h: int,
    search_w: int,
    density_r: float,
    box_h: int,
    box_w: int,
    seed: int = 42,
) -> tuple:
    """
    For each selected image, find the vertex, collect box pixels, and build:
      - per-image mean-pooled feature vectors  (for image-level k-NN)
      - flat per-pixel feature arrays          (for pixel-level k-NN)

    Images with no active pixel inside the search region are skipped entirely.

    Returns:
        s_img    : [N_img, D]  student mean-pool per image
        t_img    : [N_img, D]  teacher mean-pool per image
        img_lbls : [N_img]     class label per image
        s_pix    : [M, D]      student per-pixel features
        t_pix    : [M, D]      teacher per-pixel features
        pix_lbls : [M]         class label per pixel
        n_zero   : images skipped (empty search region)
    """
    rng = np.random.default_rng(seed)

    # Stratified image selection
    if n_images >= len(labels):
        img_indices = np.arange(len(labels))
    else:
        classes = np.unique(labels)
        per_class = max(1, n_images // len(classes))
        chosen = []
        for c in classes:
            cls_idx = np.where(labels == c)[0]
            chosen.append(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False))
        img_indices = np.concatenate(chosen)

    s_img_parts, t_img_parts, img_lbl_parts = [], [], []
    s_pix_parts, t_pix_parts, pix_lbl_parts = [], [], []
    n_zero = 0

    for img_idx in img_indices:
        sl  = slice(offsets[img_idx], offsets[img_idx + 1])
        pos = positions[sl].astype(np.float32)

        vr, vc = _find_vertex(
            pos, search_row, search_col, search_h, search_w, density_r
        )

        rows, cols = pos[:, 0], pos[:, 1]
        in_box = (
            (rows >= vr - box_h) & (rows < vr + box_h) &
            (cols >= vc - box_w) & (cols < vc + box_w)
        )

        if in_box.sum() == 0:
            n_zero += 1
            continue

        lbl   = labels[img_idx]
        n     = in_box.sum()
        s_box = s_feats[sl][in_box]   # [n, D]
        t_box = t_feats[sl][in_box]

        # Image-level: mean-pool over box pixels
        s_img_parts.append(s_box.mean(axis=0))
        t_img_parts.append(t_box.mean(axis=0))
        img_lbl_parts.append(lbl)

        # Pixel-level: keep individual pixels
        s_pix_parts.append(s_box)
        t_pix_parts.append(t_box)
        pix_lbl_parts.append(np.full(n, lbl, dtype=np.int64))

    s_img    = np.stack(s_img_parts, axis=0)
    t_img    = np.stack(t_img_parts, axis=0)
    img_lbls = np.array(img_lbl_parts, dtype=np.int64)

    s_pix    = np.concatenate(s_pix_parts,   axis=0)
    t_pix    = np.concatenate(t_pix_parts,   axis=0)
    pix_lbls = np.concatenate(pix_lbl_parts, axis=0)

    return s_img, t_img, img_lbls, s_pix, t_pix, pix_lbls, n_zero


# ---------------------------------------------------------------------------
# Batched GPU k-NN  (no full N×N matrix)
# ---------------------------------------------------------------------------

def _l2_normalise(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _knn_purity_batched(
    feats: np.ndarray,
    labels: np.ndarray,
    ks: list,
    device: torch.device,
    batch_size: int,
) -> dict:
    """
    Compute k-NN label purity for all values of k in one pass.

    Processes queries in chunks of `batch_size` rows, computing cosine
    similarities against the full set without ever materialising the N×N
    matrix.

    Returns {k: (overall_purity, per_class_array)}.
    """
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)

    max_k = min(max(ks), N - 1)

    # Accumulate top-max_k neighbour labels for every sample  [N, max_k]
    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        B    = end - start
        sim  = X[start:end] @ X.T                               # [B, N]
        # Mask self-similarity
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(max_k, dim=1)                         # [B, max_k]
        nn_labels[start:end] = lbls[idx]

    results = {}
    for k in ks:
        k_eff  = min(k, N - 1)
        same   = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)  # [N]
        overall = float(same.mean())
        per_class = np.full(len(CLASS_NAMES), np.nan)
        for c in np.unique(labels):
            mask = lbls == c
            per_class[c] = float(same[mask].mean())
        results[k] = (overall, per_class)

    return results


def _knn_predict_batched(
    feats: np.ndarray,
    labels: np.ndarray,
    k: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Majority-vote k-NN prediction, batched.  Returns [N] int predictions.
    """
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    n_cls = len(CLASS_NAMES)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T                                # [B, N]
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(k_eff, dim=1)                         # [B, k]
        nn_lbls = lbls[idx]                                     # [B, k]

        # Vectorised majority vote via batched bincount
        offsets_bc = torch.arange(B, device=device).unsqueeze(1) * n_cls  # [B, 1]
        flat = (nn_lbls + offsets_bc).reshape(-1)               # [B*k]
        counts = torch.bincount(flat, minlength=B * n_cls).reshape(B, n_cls)
        preds[start:end] = counts.argmax(dim=1)

    return preds.cpu().numpy()


# ---------------------------------------------------------------------------
# k-NN pipeline (shared by both levels)
# ---------------------------------------------------------------------------

def _run_knn(
    s_feats: np.ndarray,
    t_feats: np.ndarray,
    lbls: np.ndarray,
    ks: list,
    knn_k: int,
    reducer: str,
    out_dir: Path,
    tag: str,
    prefix: str,
    title: str,
    scatter_alpha: float,
    scatter_s: float,
    device: torch.device,
    batch_size: int,
):
    print(f"\n[{title}]  n={len(lbls)}")
    print(f"  Class distribution: "
          f"{ {CLASS_NAMES[c]: int((lbls==c).sum()) for c in np.unique(lbls)} }")

    print(f"  Computing k-NN purity (device={device}, batch={batch_size}) ...")
    s_purity_k = _knn_purity_batched(s_feats, lbls, ks, device, batch_size)
    t_purity_k = _knn_purity_batched(t_feats, lbls, ks, device, batch_size)
    for k in ks:
        print(f"  k={k:2d}  student purity={s_purity_k[k][0]:.3f}  "
              f"teacher purity={t_purity_k[k][0]:.3f}")
    plot_purity(s_purity_k, t_purity_k, out_dir, tag, f"{prefix}_purity.png", title)

    k_eff = min(knn_k, len(lbls) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    s_preds = _knn_predict_batched(s_feats, lbls, k_eff, device, batch_size)
    t_preds = _knn_predict_batched(t_feats, lbls, k_eff, device, batch_size)
    plot_confusion(s_preds, t_preds, lbls, knn_k, out_dir, tag,
                   f"{prefix}_confusion.png", title)

    print("  Running dimensionality reduction ...")
    emb_all, rname = _reduce_2d(np.concatenate([s_feats, t_feats], axis=0), method=reducer)
    N = len(lbls)
    plot_scatter(emb_all[:N], emb_all[N:], lbls, rname, out_dir, tag,
                 f"{prefix}_scatter.png", title,
                 alpha=scatter_alpha, s=scatter_s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    npz_path: Path,
    out_dir: Path,
    tag: str,
    n_images: int,
    search_row: int,
    search_col: int,
    search_h: int,
    search_w: int,
    density_r: float,
    box_h: int,
    box_w: int,
    ks: list,
    knn_k: int,
    reducer: str,
    device: torch.device,
    batch_size: int,
    seed: int = 42,
):
    print(f"Loading {npz_path}")
    data      = np.load(npz_path)
    s_feats   = data["student_features"]
    t_feats   = data["teacher_features"]
    labels    = data["labels"].astype(int)
    offsets   = data["offsets"]
    positions = data["positions"]

    print(f"  Valid pixels : {s_feats.shape[0]}   Feature dim: {s_feats.shape[1]}")
    print(f"  Images       : {len(labels)}")
    print(f"  Class counts : { {CLASS_NAMES[c]: int((labels==c).sum()) for c in np.unique(labels)} }")
    print(f"\n  Search region : rows [{search_row}, {search_row+search_h})  "
          f"cols [{search_col-search_w}, {search_col+search_w})")
    print(f"  Density radius: {density_r} px")
    print(f"  Eval box      : ±{box_h} rows × ±{box_w} cols around refined vertex")
    print(f"  Images to use : {n_images}")
    print(f"  Device        : {device}")

    print("\nCollecting vertex pixels ...")
    s_img, t_img, img_lbls, s_pix, t_pix, pix_lbls, n_zero = _collect(
        s_feats, t_feats, positions, offsets, labels,
        n_images=n_images,
        search_row=search_row, search_col=search_col,
        search_h=search_h, search_w=search_w,
        density_r=density_r,
        box_h=box_h, box_w=box_w,
        seed=seed,
    )
    print(f"  Images used  : {len(img_lbls)}  ({n_zero} skipped — empty search region)")
    print(f"  Total pixels : {len(pix_lbls)}")

    knn_kwargs = dict(ks=ks, knn_k=knn_k, reducer=reducer,
                      out_dir=out_dir, tag=tag, device=device, batch_size=batch_size)

    _run_knn(
        s_img, t_img, img_lbls,
        prefix="knn_vertex_image",
        title="Vertex region (image-level pool)",
        scatter_alpha=0.4, scatter_s=4.0,
        **knn_kwargs,
    )

    _run_knn(
        s_pix, t_pix, pix_lbls,
        prefix="knn_vertex_pixel",
        title="Vertex region (pixel-level)",
        scatter_alpha=0.15, scatter_s=2.0,
        **knn_kwargs,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vertex-focused k-NN analysis of DINO features."
    )
    parser.add_argument("npz_path",
                        help="Path to features .npz produced by extract_features")
    parser.add_argument("--out_dir", default="",
                        help="Output directory (default: same dir as .npz)")

    # Vertex finding
    parser.add_argument("--search_row", type=int, default=0,
                        help="Top row of the vertex search region (default: 0)")
    parser.add_argument("--search_col", type=int, default=250,
                        help="Centre column of the vertex search region (default: 250)")
    parser.add_argument("--search_h", type=int, default=80,
                        help="Height of the search region in rows (default: 80)")
    parser.add_argument("--search_w", type=int, default=100,
                        help="Half-width of the search region in columns (default: 100)")
    parser.add_argument("--density_r", type=float, default=10.0,
                        help="Neighbour-counting radius for density estimate (default: 10)")

    # Evaluation box
    parser.add_argument("--box_h", type=int, default=30,
                        help="Half-height of the evaluation box (default: 30)")
    parser.add_argument("--box_w", type=int, default=30,
                        help="Half-width of the evaluation box (default: 30)")

    # k-NN
    parser.add_argument("--ks", default="1,5,10,20",
                        help="Comma-separated k values for purity (default: 1,5,10,20)")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="k for the confusion-matrix majority vote (default: 5)")
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne"],
                        help="Dimensionality reduction method (default: auto)")

    # Compute
    parser.add_argument("--device", default="",
                        help="Torch device: 'cuda', 'cpu', etc. "
                             "(default: cuda if available, else cpu)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Query batch size for the GPU k-NN (default: 2048)")

    # Scale
    parser.add_argument("--n_images", type=int, default=2000,
                        help="Max images to process, stratified by class (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(
        npz_path=npz_path,
        out_dir=out_dir,
        tag=npz_path.stem,
        n_images=args.n_images,
        search_row=args.search_row,
        search_col=args.search_col,
        search_h=args.search_h,
        search_w=args.search_w,
        density_r=args.density_r,
        box_h=args.box_h,
        box_w=args.box_w,
        ks=[int(k) for k in args.ks.split(",")],
        knn_k=args.knn_k,
        reducer=args.reducer,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
