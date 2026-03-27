"""
Reproduce all debug plots from a saved histories.json.

Usage:
    python dino/plot_histories.py path/to/histories.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compute_eigen(mats: list) -> list:
    """Return [(eigenvalues, eigenvectors), ...] for each covariance matrix.
    Uses eigh (symmetric), so eigenvalues are real and in ascending order.
    """
    return [np.linalg.eigh(m) for m in mats]


def pr_from_eigen(vals: np.ndarray) -> float:
    """Participation ratio from eigenvalues: (sum λ)² / sum λ².
    Clips to zero to guard against tiny negative values from floating-point rounding.
    """
    v = vals.clip(0)
    denom = float((v ** 2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else 1.0


def plot_loss(data: dict, out_dir: Path):
    loss = data.get("loss", [])
    if not loss:
        return
    val = data.get("val", {})

    # Extract non-null teacher/student entropy and KL values along with their iteration indices.
    # Values are null (None after JSON load) for non-dino loss types.
    raw_t_ent = data.get("teacher_entropy", [])
    raw_s_ent = data.get("student_entropy", [])
    raw_kl    = data.get("kl", [])
    t_ent_iters = [i for i, v in enumerate(raw_t_ent) if v is not None]
    t_ent_vals  = [raw_t_ent[i] for i in t_ent_iters]
    s_ent_iters = [i for i, v in enumerate(raw_s_ent) if v is not None]
    s_ent_vals  = [raw_s_ent[i] for i in s_ent_iters]
    kl_iters    = [i for i, v in enumerate(raw_kl) if v is not None]
    kl_vals     = [raw_kl[i] for i in kl_iters]
    has_components = bool(t_ent_vals and kl_vals)

    # Extract non-null covariance penalty values.
    raw_cov = data.get("cov_penalty", [])
    cov_iters = [i for i, v in enumerate(raw_cov) if v is not None]
    cov_vals  = [raw_cov[i] for i in cov_iters]
    has_cov = bool(cov_vals)

    n_rows = 1 + int(has_components) + int(has_cov)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 + 4 * (n_rows - 1)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    ax_loss = axes[0]
    ax_loss.plot(loss, linewidth=1.0, alpha=0.8, label="Train (per batch)")
    if val and val.get("iter"):
        ax_loss.plot(
            val["iter"], val["loss"],
            "o-", linewidth=2.0, markersize=5, label="Val (per epoch)",
        )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    #ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    next_row = 1
    if has_components:
        ax_comp = axes[next_row]
        next_row += 1
        K = 128  # feature dimension (number of softmax categories)
        h_max = np.log(K)  # -log(1/K) = log(K): entropy of a uniform distribution over K dims
        ax_comp.plot(t_ent_iters, t_ent_vals, linewidth=1.0, alpha=0.8,
                     color="C2", label="Teacher entropy  H(P_t)")
        if s_ent_vals:
            ax_comp.plot(s_ent_iters, s_ent_vals, linewidth=1.0, alpha=0.8,
                         color="C0", label="Student entropy  H(P_s)")
        ax_comp.plot(kl_iters, kl_vals, linewidth=1.0, alpha=0.8,
                     color="C3", label="KL divergence  KL(P_t|P_s)")
        ax_comp.axhline(h_max, color="C2", linewidth=1.2, linestyle="--", alpha=0.7,
                        label=f"H_max = log({K})")
        ax_comp.set_xlabel("Iteration")
        ax_comp.set_ylabel("Nats")
        ax_comp.set_title("Loss decomposition: H(P_t, P_s) = H(P_t) + KL(P_t|P_s)  |  H(P_s)")
        #ax_comp.set_yscale("log")
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)

    if has_cov:
        ax_cov = axes[next_row]
        ax_cov.plot(cov_iters, cov_vals, linewidth=1.0, alpha=0.8,
                    color="C4", label="Covariance penalty (raw, unweighted)")
        ax_cov.set_xlabel("Iteration")
        ax_cov.set_ylabel("Penalty")
        ax_cov.set_title("VICReg covariance decorrelation penalty  (low → less dimensional correlation)")
        ax_cov.legend()
        ax_cov.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved loss_curve.png")


def plot_stats(data: dict, out_dir: Path, label: str = "backbone", mat_key: str = ""):
    h = data.get("stats", {})
    if not h or len(h.get("iter", [])) == 0:
        return

    key_s = f"s_{mat_key}cov_mat" if mat_key else "s_cov_mat"
    key_t = f"t_{mat_key}cov_mat" if mat_key else "t_cov_mat"
    raw_s = h.get(key_s, [])
    raw_t = h.get(key_t, [])
    if not raw_s or not raw_s[0]:  # head mats are [] when no head was used
        return

    iters = h["iter"]
    s_mats = [np.array(m) for m in raw_s]
    t_mats = [np.array(m) for m in raw_t]

    s_eigen = compute_eigen(s_mats)
    t_eigen = compute_eigen(t_mats)

    s_pr = [pr_from_eigen(vals) for vals, _ in s_eigen]
    t_pr = [pr_from_eigen(vals) for vals, _ in t_eigen]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    def draw_hist2d(ax, dataset, ylabel, title):
        """2-D histogram of per-feature variance across iterations."""
        x_arr = np.array(iters, dtype=float)
        y_vals = np.concatenate(dataset)

        # Build x-bin edges centred on each iteration snapshot
        if len(x_arr) > 1:
            mids = (x_arr[:-1] + x_arr[1:]) / 2.0
            half_left  = mids[0]  - x_arr[0]
            half_right = x_arr[-1] - mids[-1]
            x_edges = np.concatenate([[x_arr[0] - half_left], mids, [x_arr[-1] + half_right]])
        else:
            x_edges = np.array([x_arr[0] - 0.5, x_arr[0] + 0.5])

        y_edges = np.linspace(y_vals.min(), y_vals.max(), 51)

        x_vals = np.concatenate([np.full(len(d), it) for d, it in zip(dataset, iters)])
        H, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])

        # Mask empty bins → white
        H_masked = np.ma.masked_where(H.T == 0, H.T)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("white")
        ax.pcolormesh(xedges, yedges, H_masked, cmap=cmap)

        # Median overlay per iteration
        medians = [np.median(d) for d in dataset]
        ax.plot(x_arr, medians, color="red", linewidth=1.5, label="Median")

        ax.legend(fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    # Row 0: per-pixel L2 norm — median line with min/max band
    for col, (prefix, color, name) in enumerate([
        (f"s_{mat_key}norm", "C0", "Student"), (f"t_{mat_key}norm", "C1", "Teacher")
    ]):
        ax = axes[0, col]
        mins    = h.get(f"{prefix}_min",    [])
        maxs    = h.get(f"{prefix}_max",    [])
        medians = h.get(f"{prefix}_median", [])
        ax.fill_between(iters, mins, maxs, alpha=0.3, color=color, label="Min–Max")
        ax.plot(iters, medians, linewidth=1.5, color=color, label="Median")
        ax.set_ylabel("Per-pixel L2 norm")
        ax.set_title(f"{name} feature magnitude  (high → potential divergence)")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    # Row 1: per-feature variance 2D histogram
    draw_hist2d(axes[1, 0], [np.diag(m) for m in s_mats], "Per-feature variance", f"Student Variance [{label}]  (low → dimensional collapse)")
    draw_hist2d(axes[1, 1], [np.diag(m) for m in t_mats], "Per-feature variance", f"Teacher Variance [{label}]  (low → dimensional collapse)")

    # Row 2: participation ratio (scalar)
    for col, (pr, role) in enumerate([(s_pr, "Student"), (t_pr, "Teacher")]):
        ax = axes[2, col]
        ax.plot(iters, pr, linewidth=1.5, color=f"C{col}")
        ax.set_ylabel("Effective rank  [1, D]")
        ax.set_title(f"{role} Participation Ratio [{label}]  (low → few dominant channels)")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    fname = f"feature_stats_{label}.png"
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def plot_cov_heatmap(data: dict, out_dir: Path, label: str = "backbone", mat_key: str = ""):
    h = data.get("stats", {})
    key_s = f"s_{mat_key}cov_mat" if mat_key else "s_cov_mat"
    key_t = f"t_{mat_key}cov_mat" if mat_key else "t_cov_mat"
    s_mats = h.get(key_s, [])
    t_mats = h.get(key_t, [])
    if not s_mats or not s_mats[0]:
        return

    iters = h.get("iter", [])

    def to_corr(mat):
        std = np.sqrt(np.diag(mat)).clip(1e-8)
        return mat / np.outer(std, std)

    # Pick first and last snapshots (or just one if only one exists)
    snapshots = [(0, "first")] if len(s_mats) == 1 else [(0, "first"), (-1, "last")]

    for idx, snap in snapshots:
        s_corr = to_corr(np.array(s_mats[idx]))
        t_corr = to_corr(np.array(t_mats[idx]))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        title_suffix = f"iter {iters[idx]}" if iters else snap

        for ax, corr, name in zip(axes, [s_corr, t_corr], ["Student", "Teacher"]):
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name} [{label}] feature correlation ({title_suffix})")
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Feature index")

        fig.tight_layout()
        fname = f"cov_heatmap_{label}_{snap}.png"
        fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


def plot_eigen(data: dict, out_dir: Path, label: str = "backbone", mat_key: str = ""):
    h = data.get("stats", {})
    key_s = f"s_{mat_key}cov_mat" if mat_key else "s_cov_mat"
    key_t = f"t_{mat_key}cov_mat" if mat_key else "t_cov_mat"
    s_mats = h.get(key_s, [])
    t_mats = h.get(key_t, [])
    if not s_mats or not s_mats[0]:
        return

    iters = h.get("iter", [])
    snapshots = [(0, "first")] if len(s_mats) == 1 else [(0, "first"), (-1, "last")]

    s_eigen = compute_eigen([np.array(m) for m in s_mats])
    t_eigen = compute_eigen([np.array(m) for m in t_mats])

    for idx, snap in snapshots:
        s_cov = np.array(s_mats[idx])
        t_cov = np.array(t_mats[idx])
        s_vals, s_vecs = s_eigen[idx]
        t_vals, t_vecs = t_eigen[idx]

        s_cov_eigen = (s_vecs.T @ s_cov @ s_vecs) / s_vals[-1]
        t_cov_eigen = (t_vecs.T @ t_cov @ t_vecs) / t_vals[-1]

        title_suffix = f"iter {iters[idx]}" if iters else snap
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax, vals, name in zip(axes[0], [s_vals, t_vals], ["Student", "Teacher"]):
            ax.bar(np.arange(len(vals)), vals, width=1.0)
            ax.set_yscale("log")
            ax.set_xlabel("Eigenvalue index (ascending)")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"{name} [{label}] eigenvalue spectrum ({title_suffix})")
            ax.grid(True, alpha=0.3)

        vmax = max(np.abs(s_cov_eigen).max(), np.abs(t_cov_eigen).max())
        for ax, mat, name in zip(axes[1], [s_cov_eigen, t_cov_eigen], ["Student", "Teacher"]):
            im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name} [{label}] covariance in eigenbasis ({title_suffix})")
            ax.set_xlabel("Eigenvector index")
            ax.set_ylabel("Eigenvector index")

        fig.tight_layout()
        fname = f"eigen_{label}_{snap}.png"
        fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


def plot_center_stats(data: dict, out_dir: Path):
    h = data.get("stats", {})
    iters       = h.get("iter", [])
    center_norm = h.get("center_norm", [])
    center_var  = h.get("center_var", [])

    if not iters or not center_norm:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(iters, center_norm, linewidth=1.5, color="C2")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("Teacher center norm  (large → strong mean bias being corrected)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, center_var, linewidth=1.5, color="C3")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Variance across dims")
    axes[1].set_title("Teacher center variance  (low → center concentrated in few dims)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "center_stats.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved center_stats.png")


def _expand_groups(grad: dict) -> list[tuple[str, dict]]:
    """
    Return (title, params_dict) pairs for each subplot in forward-pass execution order.

    If the 'bottleneck' group contains attention layers it is split into two:
      - 'bottleneck (attention)': pre_proj + attn.*   [attention path]
      - 'bottleneck (MLP)':       mlp.* + post_proj   [feed-forward + output]
    Otherwise 'bottleneck' (plain residual) stays as a single subplot.
    Any groups not in the known order are appended at the end.
    """
    _EXEC_ORDER = [
        "conv0", "conv1", "block1", "conv2", "block2",
        "bottleneck",
        "convtr5", "block6", "convtr7", "block8",
        "final",
    ]
    _ATTN_PREFIXES = ("pre_proj.", "attn.")
    _MLP_PREFIXES  = ("mlp.", "post_proj.")

    ordered_keys = [k for k in _EXEC_ORDER if k in grad]
    ordered_keys += [k for k in grad if k not in _EXEC_ORDER]

    result = []
    for group in ordered_keys:
        params = grad[group]
        if group == "bottleneck":
            attn_params = {s: d for s, d in params.items() if s.startswith(_ATTN_PREFIXES)}
            mlp_params  = {s: d for s, d in params.items() if s.startswith(_MLP_PREFIXES)}
            if attn_params and mlp_params:
                result.append(("bottleneck (attention)", attn_params))
                result.append(("bottleneck (MLP)",       mlp_params))
                continue
        result.append((group, params))
    return result


def plot_grads(data: dict, out_dir: Path):
    grad = data.get("grad", {})
    if not grad:
        return

    panels = _expand_groups(grad)
    n = len(panels)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3),
                             sharex=False, sharey=False)
    axes_flat = np.array(axes).reshape(-1) if n > 1 else [axes]

    for i, (title, params) in enumerate(panels):
        ax = axes_flat[i]
        for suffix, d in params.items():
            ax.plot(d["iter"], d["norm"], linewidth=1.5, label=suffix)
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Grad L2 Norm")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if i >= n_cols * (n_rows - 1):
            ax.set_xlabel("Iteration")

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Gradient Norms per Backbone Module", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norms.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved grad_norms.png")


def main():
    if len(sys.argv) != 2:
        print("Usage: python dino/plot_histories.py path/to/histories.json")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    out_dir = json_path.parent
    print(f"Reading {json_path}")
    print(f"Saving plots to {out_dir}/")

    with open(json_path) as f:
        data = json.load(f)

    plot_loss(data, out_dir)
    plot_stats(data, out_dir, label="backbone", mat_key="")
    plot_stats(data, out_dir, label="head", mat_key="head_")
    plot_center_stats(data, out_dir)
    plot_cov_heatmap(data, out_dir, label="backbone", mat_key="")
    plot_cov_heatmap(data, out_dir, label="head", mat_key="head_")
    plot_eigen(data, out_dir, label="backbone", mat_key="")
    plot_eigen(data, out_dir, label="head", mat_key="head_")
    plot_grads(data, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
