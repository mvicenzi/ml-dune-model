"""
k-NN analysis of DINO features in 64-D cosine space.

Two analysis levels:
  1. Image-level (always): mean-pool pixel features per image, then k-NN
     purity, k-NN classifier accuracy, confusion matrix, and a 2-D
     UMAP / t-SNE scatter.
  2. Pixel-level (optional, --pixel_knn): a random subsample of pixels
     (each inheriting its parent image's label), same purity + scatter.

Produces:
  knn_image_purity.png     — per-class k-NN label purity at k=1,5,10,20
  knn_image_confusion.png  — confusion matrix (k-NN majority vote, k=5)
  knn_image_scatter.png    — UMAP or t-SNE 2-D scatter, student + teacher
  knn_pixel_purity.png     — (--pixel_knn) pixel-level purity
  knn_pixel_scatter.png    — (--pixel_knn) pixel-level scatter

Usage:
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --out_dir=./plots
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --pixel_knn --n_pixel_samples=50000
    python -m dino.diagnostics.plot_knn path/to/features_ep10.npz --ks=1,5,10,20 --knn_k=10
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        out[i] = feats[sl].mean(axis=0)
    return out


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for normalised rows → dot product."""
    Xn = _l2_norm(X)
    return Xn @ Xn.T   # [N, N]


def _knn_purity(sim: np.ndarray, labels: np.ndarray, k: int) -> tuple:
    """
    For each sample, find the k nearest neighbours (excluding self) and
    compute the fraction that share the same label.

    Returns:
        overall_purity : float
        per_class_purity : ndarray [n_classes]  (NaN if class absent)
    """
    N = len(labels)
    # Exclude self by setting the diagonal to -inf before argsort
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)

    # Indices of k nearest neighbours per row (descending similarity)
    nn_idx = np.argpartition(-S, kth=min(k, N - 1), axis=1)[:, :k]

    purities = np.empty(N, dtype=float)
    for i in range(N):
        nbr_labels = labels[nn_idx[i]]
        purities[i] = (nbr_labels == labels[i]).mean()

    classes = np.unique(labels)
    per_class = np.full(len(CLASS_NAMES), np.nan)
    for c in classes:
        mask = labels == c
        per_class[c] = purities[mask].mean()

    return float(purities.mean()), per_class


def _knn_predict(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Majority-vote k-NN prediction (leave-one-out via -inf diagonal)."""
    N = len(labels)
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    nn_idx = np.argpartition(-S, kth=min(k, N - 1), axis=1)[:, :k]

    preds = np.empty(N, dtype=int)
    for i in range(N):
        nbr_labels = labels[nn_idx[i]]
        counts = np.bincount(nbr_labels, minlength=len(CLASS_NAMES))
        preds[i] = int(counts.argmax())
    return preds


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
            overall, per_class = purity_k[k]
            values = list(per_class) + [overall]
            bars = ax.bar(
                x + ki * width - 0.4 + width / 2,
                values,
                width=width,
                label=f"k={k}",
            )
        # Random-chance baseline
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

        # Row-normalise → recall per class
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
    fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
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
):
    print("\n[Image-level analysis]")
    print("  Mean-pooling pixel features per image ...")
    s_img = _mean_pool(s_feats, offsets)   # [N_images, D]
    t_img = _mean_pool(t_feats, offsets)

    print(f"  Image embeddings: {s_img.shape}  |  classes: {np.unique(labels)}")

    print("  Computing pairwise cosine similarity matrices ...")
    s_sim = _cosine_sim_matrix(s_img)
    t_sim = _cosine_sim_matrix(t_img)

    # --- Purity for each k ---
    s_purity_k, t_purity_k = {}, {}
    for k in ks:
        k_eff = min(k, len(labels) - 1)
        s_overall, s_pc = _knn_purity(s_sim, labels, k_eff)
        t_overall, t_pc = _knn_purity(t_sim, labels, k_eff)
        s_purity_k[k] = (s_overall, s_pc)
        t_purity_k[k] = (t_overall, t_pc)
        print(f"  k={k:2d}  student purity={s_overall:.3f}  teacher purity={t_overall:.3f}")

    plot_purity(s_purity_k, t_purity_k, out_dir, tag,
                "knn_image_purity.png", "Image-level")

    # --- Confusion matrix at knn_k ---
    k_eff = min(knn_k, len(labels) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    s_preds = _knn_predict(s_sim, labels, k_eff)
    t_preds = _knn_predict(t_sim, labels, k_eff)
    plot_confusion(s_preds, t_preds, labels, knn_k, out_dir, tag,
                   "knn_image_confusion.png", "Image-level")

    # --- 2-D scatter ---
    print("  Running dimensionality reduction on image embeddings ...")
    all_feats = np.concatenate([s_img, t_img], axis=0)
    emb_all, rname = _reduce_2d(all_feats, method=reducer)
    N = len(labels)
    s_emb = emb_all[:N]
    t_emb = emb_all[N:]
    plot_scatter(s_emb, t_emb, labels, rname, out_dir, tag,
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
    seed: int = 42,
):
    print("\n[Pixel-level analysis]")
    rng = np.random.default_rng(seed)

    # Expand per-image labels to per-pixel
    px_labels = _pixel_labels(labels, offsets)

    # Subsample — draw indices once so student and teacher share the same pixels
    n = min(n_samples, len(s_feats))
    idx = rng.choice(len(s_feats), size=n, replace=False)
    s_sub, t_sub = s_feats[idx], t_feats[idx]
    pl_s = pl_t = px_labels[idx]
    print(f"  Pixel subsample: {len(s_sub)} pixels  (requested {n_samples})")

    # For purity we need pairwise similarity — expensive at large N.
    # Split into student and teacher similarity independently.
    print("  Computing pairwise cosine similarity matrices ...")
    s_sim = _cosine_sim_matrix(s_sub)
    t_sim = _cosine_sim_matrix(t_sub)

    s_purity_k, t_purity_k = {}, {}
    for k in ks:
        k_eff = min(k, len(pl_s) - 1)
        s_overall, s_pc = _knn_purity(s_sim, pl_s, k_eff)
        t_overall, t_pc = _knn_purity(t_sim, pl_t, k_eff)
        s_purity_k[k] = (s_overall, s_pc)
        t_purity_k[k] = (t_overall, t_pc)
        print(f"  k={k:2d}  student purity={s_overall:.3f}  teacher purity={t_overall:.3f}")

    plot_purity(s_purity_k, t_purity_k, out_dir, tag,
                "knn_pixel_purity.png", "Pixel-level")

    # Confusion matrix at knn_k
    k_eff = min(knn_k, len(pl_s) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    s_preds = _knn_predict(s_sim, pl_s, k_eff)
    t_preds = _knn_predict(t_sim, pl_t, k_eff)
    plot_confusion(s_preds, t_preds, pl_s, knn_k, out_dir, tag,
                   "knn_pixel_confusion.png", "Pixel-level")

    # 2-D scatter
    print("  Running dimensionality reduction on pixel subsample ...")
    all_feats = np.concatenate([s_sub, t_sub], axis=0)
    emb_all, rname = _reduce_2d(all_feats, method=reducer)
    N = len(pl_s)
    s_emb = emb_all[:N]
    t_emb = emb_all[N:]
    plot_scatter(s_emb, t_emb, pl_s, rname, out_dir, tag,
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
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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

    tag = npz_path.stem
    print(f"  Output dir   : {out_dir}/")

    run_image_level(
        s_feats, t_feats, labels, offsets,
        out_dir, tag, ks=ks, knn_k=args.knn_k, reducer=args.reducer,
    )

    if args.pixel_knn:
        run_pixel_level(
            s_feats, t_feats, labels, offsets,
            out_dir, tag, ks=ks, knn_k=args.knn_k,
            n_samples=args.n_pixel_samples, reducer=args.reducer,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
