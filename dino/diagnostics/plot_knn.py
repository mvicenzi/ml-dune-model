"""
k-NN analysis of DINO features in 64-D cosine space.

Two analysis levels:
  1. Image-level (always): mean-pool pixel features per image, then k-NN
     purity, k-NN classifier accuracy, confusion matrix, and a 2-D
     UMAP / t-SNE scatter.
  2. Pixel-level (optional, --pixel_knn): a random subsample of pixels
     (each inheriting its parent image's label), same purity + scatter.

k-NN similarity is computed in batches on GPU (or CPU) via PyTorch matmuls,
so the full N×N matrix is never materialised in memory.

Produces:
  knn_image_purity.png      — per-class k-NN label purity at k=1,5,10,20
  knn_image_purity_hist.png — per-sample purity histograms by class
  knn_image_confusion.png   — confusion matrix (k-NN majority vote, k=5)
  knn_image_scatter.png     — UMAP or t-SNE 2-D scatter, student + teacher
  knn_pixel_purity.png      — (--pixel_knn) pixel-level purity
  knn_pixel_purity_hist.png — (--pixel_knn) pixel-level purity histograms
  knn_pixel_scatter.png     — (--pixel_knn) pixel-level scatter

Usage:
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --out_dir=./plots
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --pixel_knn --n_pixel_samples=50000
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --ks=1,5,10,20 --knn_k=10
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --device=cuda --batch_size=2048
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CLASS_NAMES = ["numu", "nue", "nutau", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation — cosine similarity becomes a dot product."""
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    return X / norms


def _mean_pool(feats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean-pool pixel features per image → [N_images, D]."""
    n_images = len(offsets) - 1
    D = feats.shape[1]
    out = np.empty((n_images, D), dtype=np.float32)
    for i in range(n_images):
        sl = slice(offsets[i], offsets[i + 1])
        if offsets[i] == offsets[i + 1]:
            out[i] = 0.0  # empty image — fill with zeros to avoid NaN
        else:
            out[i] = feats[sl].mean(axis=0)
    return out


def _pixel_labels(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Expand per-image labels to per-pixel labels."""
    n_pixels = offsets[-1]
    out = np.empty(n_pixels, dtype=np.int64)
    for i in range(len(labels)):
        out[offsets[i]:offsets[i + 1]] = labels[i]
    return out


def _subsample(feats: np.ndarray, labels: np.ndarray, n: int, rng):
    """Random subsample of n rows, stratified by class if possible."""
    if len(feats) <= n:
        return feats, labels
    idx = rng.choice(len(feats), size=n, replace=False)
    return feats[idx], labels[idx]


def _reduce_2d(X: np.ndarray, method: str = "auto"):
    """Reduce [N, D] to [N, 2].  method: 'umap', 'tsne', or 'auto'."""
    if method in ("umap", "auto"):
        try:
            import umap  # noqa: F401
            from umap import UMAP
            print("    Using UMAP for 2-D reduction.")
            return UMAP(n_components=2, metric="cosine",
                        random_state=42, verbose=False).fit_transform(X), "UMAP"
        except ImportError:
            if method == "umap":
                raise
            print("    umap-learn not found — falling back to t-SNE.")
    from sklearn.manifold import TSNE
    print("    Using t-SNE for 2-D reduction.")
    emb = TSNE(n_components=2, metric="cosine", init="pca",
               random_state=42, n_jobs=-1).fit_transform(X)
    return emb, "t-SNE"


# ---------------------------------------------------------------------------
# Batched GPU k-NN  (no full N×N matrix)
# ---------------------------------------------------------------------------

def _l2_normalise_t(X: torch.Tensor) -> torch.Tensor:
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
    similarities against the full set without ever materialising the N×N matrix.

    Returns {k: (overall_purity, per_class_array)}.
    """
    X    = _l2_normalise_t(torch.from_numpy(feats.astype(np.float32)).to(device))
    N    = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)

    max_k = min(max(ks), N - 1)

    # Accumulate top-max_k neighbour labels for every sample  [N, max_k]
    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T                                           # [B, N]
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf         # mask self
        _, idx = sim.topk(max_k, dim=1)                                    # [B, max_k]
        nn_labels[start:end] = lbls[idx]

    results = {}
    for k in ks:
        k_eff   = min(k, N - 1)
        same    = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)  # [N]
        overall = float(same.mean())
        per_class = np.full(len(CLASS_NAMES), np.nan)
        for c in np.unique(labels):
            per_class[c] = float(same[lbls == c].mean())
        results[k] = (overall, per_class, same.cpu().numpy())

    return results


def _knn_predict_batched(
    feats: np.ndarray,
    labels: np.ndarray,
    k: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Majority-vote k-NN prediction, batched.  Returns [N] int predictions."""
    X     = _l2_normalise_t(torch.from_numpy(feats.astype(np.float32)).to(device))
    N     = X.shape[0]
    lbls  = torch.from_numpy(labels.astype(np.int64)).to(device)
    n_cls = len(CLASS_NAMES)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T                                           # [B, N]
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf
        _, idx   = sim.topk(k_eff, dim=1)                                  # [B, k]
        nn_lbls  = lbls[idx]                                               # [B, k]

        # Vectorised majority vote via batched bincount
        offsets_bc = torch.arange(B, device=device).unsqueeze(1) * n_cls  # [B, 1]
        flat       = (nn_lbls + offsets_bc).reshape(-1)                    # [B*k]
        counts     = torch.bincount(flat, minlength=B * n_cls).reshape(B, n_cls)
        preds[start:end] = counts.argmax(dim=1)

    return preds.cpu().numpy()


# ---------------------------------------------------------------------------
# Plot: k-NN purity bar chart
# ---------------------------------------------------------------------------

def plot_purity(
    s_purity_k: dict,   # {k: (overall, per_class)} for student
    t_purity_k: dict,
    out_dir: Path,
    tag: str,
    fname: str,
    title_prefix: str,
):
    ks = sorted(s_purity_k.keys())
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    x = np.arange(n_classes + 1)        # one bar per class + overall
    width = 0.8 / len(ks)
    tick_labels = CLASS_NAMES + ["Overall"]

    for ax, purity_k, name in zip(axes, [s_purity_k, t_purity_k], ["Student", "Teacher"]):
        for ki, k in enumerate(ks):
            overall, per_class, _ = purity_k[k]
            values = list(per_class) + [overall]
            ax.bar(
                x + ki * width - 0.4 + width / 2,
                values,
                width=width,
                label=f"k={k}",
            )
        chance = 1.0 / n_classes
        ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
                   label=f"chance (1/{n_classes})")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel("Label purity")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} k-NN label purity  [{tag}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: per-sample purity histograms
# ---------------------------------------------------------------------------

def plot_purity_hist(
    s_purity_k: dict,   # {k: (overall, per_class, same_per_sample)}
    t_purity_k: dict,
    labels: np.ndarray,
    out_dir: Path,
    tag: str,
    fname: str,
    title_prefix: str,
):
    ks = sorted(s_purity_k.keys())
    n_classes = len(CLASS_NAMES)
    cmap = plt.get_cmap("tab10")

    # One row per k, two columns (student / teacher)
    fig, axes = plt.subplots(len(ks), 2, figsize=(14, 4 * len(ks)), squeeze=False)

    for row, k in enumerate(ks):
        # Purity is j/k for j in 0..k — align bin edges to the midpoints between
        # consecutive discrete values so each bar captures exactly one value.
        # Edges: -0.5/k, 0.5/k, 1.5/k, ..., (k+0.5)/k
        bins = (np.arange(k + 2) - 0.5) / k
        xticks = np.arange(k + 1) / k

        for col, (purity_k, name) in enumerate([(s_purity_k, "Student"), (t_purity_k, "Teacher")]):
            ax = axes[row, col]
            _, _, same = purity_k[k]

            for c in range(n_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                ax.hist(
                    same[mask],
                    bins=bins,
                    histtype="step",
                    color=cmap(c),
                    lw=2,
                    label=f"{CLASS_NAMES[c]} (μ={same[mask].mean():.2f})",
                    density=True,
                )

            ax.set_xlim(-0.5 / k, 1 + 0.5 / k)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:.2g}" for v in xticks], fontsize=7, rotation=45)
            ax.set_xlabel("Per-sample purity")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}  k={k}")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} k-NN purity distribution  [{tag}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion(
    s_preds: np.ndarray,
    t_preds: np.ndarray,
    labels: np.ndarray,
    k: int,
    out_dir: Path,
    tag: str,
    fname: str,
    title_prefix: str,
):
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, name in zip(axes, [s_preds, t_preds], ["Student", "Teacher"]):
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

        row_sums = cm.sum(axis=1, keepdims=True).clip(1)
        cm_norm = cm / row_sums

        acc = (labels == preds).mean()
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.6 else "black")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  (acc={acc:.3f})")

    fig.suptitle(f"{title_prefix} k-NN confusion matrix (k={k})  [{tag}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: 2-D scatter
# ---------------------------------------------------------------------------

def plot_scatter(
    s_emb: np.ndarray,
    t_emb: np.ndarray,
    labels: np.ndarray,
    reducer_name: str,
    out_dir: Path,
    tag: str,
    fname: str,
    title_prefix: str,
    alpha: float = 0.4,
    s: float = 4.0,
):
    n_classes = len(CLASS_NAMES)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, name in zip(axes, [s_emb, t_emb], ["Student", "Teacher"]):
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       color=colors[c], label=CLASS_NAMES[c],
                       alpha=alpha, s=s, linewidths=0)
        ax.set_title(f"{name}")
        ax.set_xlabel(f"{reducer_name} dim 1")
        ax.set_ylabel(f"{reducer_name} dim 2")
        ax.legend(markerscale=3, fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title_prefix} {reducer_name} scatter  [{tag}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Image-level pipeline
# ---------------------------------------------------------------------------

def run_image_level(
    s_feats: np.ndarray,
    t_feats: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    out_dir: Path,
    tag: str,
    ks: list,
    knn_k: int,
    reducer: str,
    device: torch.device,
    batch_size: int,
):
    print("\n[Image-level analysis]")
    print("  Mean-pooling pixel features per image ...")
    s_img = _mean_pool(s_feats, offsets)   # [N_images, D]
    t_img = _mean_pool(t_feats, offsets)

    print(f"  Image embeddings: {s_img.shape}  |  classes: {np.unique(labels)}")

    print(f"  Computing k-NN purity (device={device}, batch={batch_size}) ...")
    s_purity_k = _knn_purity_batched(s_img, labels, ks, device, batch_size)
    t_purity_k = _knn_purity_batched(t_img, labels, ks, device, batch_size)
    for k in ks:
        print(f"  k={k:2d}  student purity={s_purity_k[k][0]:.3f}  "
              f"teacher purity={t_purity_k[k][0]:.3f}")

    plot_purity(s_purity_k, t_purity_k, out_dir, tag,
                "knn_image_purity.png", "Image-level")
    plot_purity_hist(s_purity_k, t_purity_k, labels, out_dir, tag,
                     "knn_image_purity_hist.png", "Image-level")

    k_eff = min(knn_k, len(labels) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    s_preds = _knn_predict_batched(s_img, labels, k_eff, device, batch_size)
    t_preds = _knn_predict_batched(t_img, labels, k_eff, device, batch_size)
    plot_confusion(s_preds, t_preds, labels, knn_k, out_dir, tag,
                   "knn_image_confusion.png", "Image-level")

    print("  Running dimensionality reduction on image embeddings ...")
    # Uncomment the block below to exclude nutau from the scatter plot:
    # nutau_idx = CLASS_NAMES.index("nutau")
    # scatter_mask = labels != nutau_idx
    # s_img_plot, t_img_plot, labels_plot = s_img[scatter_mask], t_img[scatter_mask], labels[scatter_mask]
    s_img_plot, t_img_plot, labels_plot = s_img, t_img, labels
    all_feats = np.concatenate([s_img_plot, t_img_plot], axis=0)
    emb_all, rname = _reduce_2d(all_feats, method=reducer)
    N = len(s_img_plot)
    plot_scatter(emb_all[:N], emb_all[N:], labels_plot, rname, out_dir, tag,
                 "knn_image_scatter.png", "Image-level")


# ---------------------------------------------------------------------------
# Pixel-level pipeline
# ---------------------------------------------------------------------------

def run_pixel_level(
    s_feats: np.ndarray,
    t_feats: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    out_dir: Path,
    tag: str,
    ks: list,
    knn_k: int,
    n_samples: int,
    reducer: str,
    device: torch.device,
    batch_size: int,
    seed: int = 42,
):
    print("\n[Pixel-level analysis]")
    rng = np.random.default_rng(seed)

    px_labels = _pixel_labels(labels, offsets)

    n = min(n_samples, len(s_feats))
    idx = rng.choice(len(s_feats), size=n, replace=False)
    s_sub = s_feats[idx]
    t_sub = t_feats[idx]
    pl    = px_labels[idx]
    print(f"  Pixel subsample: {len(s_sub)} pixels  (requested {n_samples})")

    print(f"  Computing k-NN purity (device={device}, batch={batch_size}) ...")
    s_purity_k = _knn_purity_batched(s_sub, pl, ks, device, batch_size)
    t_purity_k = _knn_purity_batched(t_sub, pl, ks, device, batch_size)
    for k in ks:
        print(f"  k={k:2d}  student purity={s_purity_k[k][0]:.3f}  "
              f"teacher purity={t_purity_k[k][0]:.3f}")

    plot_purity(s_purity_k, t_purity_k, out_dir, tag,
                "knn_pixel_purity.png", "Pixel-level")
    plot_purity_hist(s_purity_k, t_purity_k, pl, out_dir, tag,
                     "knn_pixel_purity_hist.png", "Pixel-level")

    k_eff = min(knn_k, len(pl) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    s_preds = _knn_predict_batched(s_sub, pl, k_eff, device, batch_size)
    t_preds = _knn_predict_batched(t_sub, pl, k_eff, device, batch_size)
    plot_confusion(s_preds, t_preds, pl, knn_k, out_dir, tag,
                   "knn_pixel_confusion.png", "Pixel-level")

    print("  Running dimensionality reduction on pixel subsample ...")
    all_feats = np.concatenate([s_sub, t_sub], axis=0)
    emb_all, rname = _reduce_2d(all_feats, method=reducer)
    N = len(pl)
    plot_scatter(emb_all[:N], emb_all[N:], pl, rname, out_dir, tag,
                 "knn_pixel_scatter.png", "Pixel-level",
                 alpha=0.15, s=2.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="k-NN analysis of DINO features in cosine space."
    )
    parser.add_argument("npz_path", help="Path to features .npz produced by extract_features")
    parser.add_argument("--out_dir", default="",
                        help="Output directory (default: same dir as .npz)")
    parser.add_argument("--ks", default="1,5,10,20",
                        help="Comma-separated list of k values for purity (default: 1,5,10,20)")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="k used for the confusion-matrix majority vote (default: 5)")
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne"],
                        help="Dimensionality reduction method (default: auto → UMAP then t-SNE)")
    parser.add_argument("--pixel_knn", action="store_true",
                        help="Also run the pixel-level k-NN analysis")
    parser.add_argument("--n_pixel_samples", type=int, default=30000,
                        help="Number of pixels to subsample for pixel-level analysis (default: 30000)")
    parser.add_argument("--device", default="",
                        help="Torch device: 'cuda', 'cpu', etc. "
                             "(default: cuda if available, else cpu)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Query batch size for the GPU k-NN (default: 2048)")
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

    ks = [int(k) for k in args.ks.split(",")]

    print(f"Loading {npz_path}")
    data = np.load(npz_path)
    s_feats  = data["student_features"]   # [N_valid, D]
    t_feats  = data["teacher_features"]
    labels   = data["labels"].astype(int) # [N_images]
    offsets  = data["offsets"]            # [N_images+1]
    positions = data["positions"]         # [N_valid, 2]  (not used here but validated)

    print(f"  Valid pixels : {s_feats.shape[0]}   Feature dim: {s_feats.shape[1]}")
    print(f"  Images       : {len(labels)}")
    print(f"  Class counts : { {CLASS_NAMES[c]: int((labels==c).sum()) for c in np.unique(labels)} }")
    print(f"  Device       : {device}")

    tag = npz_path.stem
    print(f"  Output dir   : {out_dir}/")

    run_image_level(
        s_feats, t_feats, labels, offsets,
        out_dir, tag, ks=ks, knn_k=args.knn_k, reducer=args.reducer,
        device=device, batch_size=args.batch_size,
    )

    if args.pixel_knn:
        run_pixel_level(
            s_feats, t_feats, labels, offsets,
            out_dir, tag, ks=ks, knn_k=args.knn_k,
            n_samples=args.n_pixel_samples, reducer=args.reducer,
            device=device, batch_size=args.batch_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
