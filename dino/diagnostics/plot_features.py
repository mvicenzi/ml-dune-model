"""
Analyse feature covariance structure from a saved extract_features .npz file.

Produces two figures saved alongside the input file:

  features_cov_corr.png  —  covariance and correlation matrices for student
                             and teacher (2 × 2 grid)
  features_eigen.png     —  eigenvalue spectrum and covariance in eigenbasis
                             for student and teacher (2 × 2 grid)

Usage:
    python -m dino.diagnostics.plot_features path/to/features_ep10.npz
    python -m dino.diagnostics.plot_features path/to/features_ep10.npz --out_dir=./plots
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers (shared style with plot_histories.py)
# ---------------------------------------------------------------------------

def _compute_cov(feats: np.ndarray) -> np.ndarray:
    """Mean-centre and compute the [D, D] sample covariance matrix."""
    X = feats - feats.mean(axis=0)
    return (X.T @ X) / (len(X) - 1)


def _to_corr(cov: np.ndarray) -> np.ndarray:
    """Normalise a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov)).clip(1e-8)
    return cov / np.outer(std, std)


def _eigh(mat: np.ndarray):
    """Eigendecomposition for symmetric matrix; eigenvalues in ascending order."""
    return np.linalg.eigh(mat)


def _pr(vals: np.ndarray) -> float:
    """Participation ratio: (Σλ)² / Σλ²  — effective rank in [1, D]."""
    v = vals.clip(0)
    denom = float((v ** 2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Plot 1: covariance + correlation matrices
# ---------------------------------------------------------------------------

def plot_cov_and_corr(s_cov: np.ndarray, t_cov: np.ndarray, out_dir: Path, tag: str):
    s_corr = _to_corr(s_cov)
    t_corr = _to_corr(t_cov)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Row 0: covariance matrices — each uses its own colour scale so small
    # values (e.g. student) are not crushed by a larger-magnitude counterpart
    for ax, mat, name in zip(axes[0], [s_cov, t_cov], ["Student", "Teacher"]):
        vmax_cov = np.abs(mat).max()
        im = ax.imshow(mat, vmin=-vmax_cov, vmax=vmax_cov, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name} covariance matrix")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")

    # Row 1: correlation matrices
    for ax, mat, name in zip(axes[1], [s_corr, t_corr], ["Student", "Teacher"]):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name} correlation matrix")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")

    fig.suptitle(f"Feature covariance / correlation  [{tag}]", fontsize=13)
    fig.tight_layout()
    fname = "features_cov_corr.png"
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2: eigenvalue spectrum + covariance in eigenbasis
# ---------------------------------------------------------------------------

def plot_eigen(s_cov: np.ndarray, t_cov: np.ndarray, out_dir: Path, tag: str):
    s_vals, s_vecs = _eigh(s_cov)
    t_vals, t_vecs = _eigh(t_cov)

    s_pr = _pr(s_vals)
    t_pr = _pr(t_vals)
    print(f"  Participation ratio — student: {s_pr:.2f}  teacher: {t_pr:.2f}  (max={s_cov.shape[0]})")

    # Covariance in eigenbasis, normalised by the largest eigenvalue so the
    # colour scale is comparable across runs (identical to plot_histories.py)
    s_cov_eig = (s_vecs.T @ s_cov @ s_vecs) / s_vals[-1]
    t_cov_eig = (t_vecs.T @ t_cov @ t_vecs) / t_vals[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 0: eigenvalue spectra
    for ax, vals, pr, name in zip(axes[0], [s_vals, t_vals], [s_pr, t_pr], ["Student", "Teacher"]):
        ax.bar(np.arange(len(vals)), vals, width=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Eigenvalue index (ascending)")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"{name} eigenvalue spectrum  (PR = {pr:.1f})")
        ax.grid(True, alpha=0.3)

    # Row 1: covariance in eigenbasis — per-matrix scale for the same reason
    for ax, mat, name in zip(axes[1], [s_cov_eig, t_cov_eig], ["Student", "Teacher"]):
        vmax = np.abs(mat).max()
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name} covariance in eigenbasis  (normalised)")
        ax.set_xlabel("Eigenvector index")
        ax.set_ylabel("Eigenvector index")

    fig.suptitle(f"Eigenvalue decomposition  [{tag}]", fontsize=13)
    fig.tight_layout()
    fname = "features_eigen.png"
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot feature covariance diagnostics from a .npz file."
    )
    parser.add_argument("npz_path", help="Path to features .npz produced by extract_features")
    parser.add_argument(
        "--out_dir", default="",
        help="Output directory for plots (default: same directory as the .npz file)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {npz_path}")
    data = np.load(npz_path)
    s_feats = data["student_features"]   # [N, D]
    t_feats = data["teacher_features"]   # [N, D]
    print(f"  Pixels: {s_feats.shape[0]}   Feature dim: {s_feats.shape[1]}")

    tag = npz_path.stem   # e.g. "features_ep10"

    print("Computing covariance matrices ...")
    s_cov = _compute_cov(s_feats)
    t_cov = _compute_cov(t_feats)

    print(f"Saving plots to {out_dir}/")
    plot_cov_and_corr(s_cov, t_cov, out_dir, tag)
    plot_eigen(s_cov, t_cov, out_dir, tag)

    print("Done.")


if __name__ == "__main__":
    main()
