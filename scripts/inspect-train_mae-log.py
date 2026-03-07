#!/usr/bin/env python3
"""
inspect-train_mae-log.py — Parse train_mae.py log and produce summary plots.

Plots produced
--------------
1. ssl_loss.png
   SSL L1 reconstruction loss vs. global training step (running mean logged
   every 50 steps, concatenated across all SSL epochs).

2. sft_loss_acc.png
   Top panel : focal loss  | Bottom panel : accuracy (%)
   X-axis: sequential SFT sub-epoch index (e.g. 1–100 for 10 SSL × 10 SFT).
   Each SSL epoch gets a distinct color; solid = SSL features, dashed = raw charge.
   Vertical lines mark the start of each SSL epoch's SFT block.

Usage
-----
    python scripts/inspect-train_mae-log.py               # default: log -> viz_eval/
    python scripts/inspect-train_mae-log.py --log_path=log --out_dir=viz_eval
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SSL_RE = re.compile(
    r"\[SSL\] Epoch (\d+)\s+step \[(\d+)/(\d+)\]\s+loss=([\d.]+)"
)
_SFT_RE = re.compile(
    r"SFT epoch (\d+)/\d+\s+\|\s+SSL-feat: CE=([\d.]+) acc=([\d.]+)%"
    r"\s+\|\s+raw-charge: CE=([\d.]+) acc=([\d.]+)%"
)


def parse_log(path: str) -> dict:
    """
    Returns
    -------
    {
        "ssl_steps"   : list of (global_step: int, loss: float),
        "sft_records" : list of (ssl_epoch: int, sft_epoch: int,
                                  sft_loss, sft_acc, ref_loss, ref_acc),
    }
    """
    ssl_steps   = []
    sft_records = []
    current_ssl = 0

    with open(path) as fh:
        for line in fh:
            m = _SSL_RE.search(line)
            if m:
                epoch       = int(m.group(1))
                step        = int(m.group(2))
                total       = int(m.group(3))
                loss        = float(m.group(4))
                global_step = (epoch - 1) * total + step
                ssl_steps.append((global_step, loss))
                current_ssl = epoch
                continue

            m = _SFT_RE.search(line)
            if m:
                sft_epoch = int(m.group(1))
                sft_records.append((
                    current_ssl, sft_epoch,
                    float(m.group(2)), float(m.group(3)),   # sft loss, acc
                    float(m.group(4)), float(m.group(5)),   # ref loss, acc
                ))

    return {"ssl_steps": ssl_steps, "sft_records": sft_records}


# ---------------------------------------------------------------------------
# Plot 1 — SSL loss
# ---------------------------------------------------------------------------

def plot_ssl_loss(ssl_steps: list, out_path: Path) -> None:
    if not ssl_steps:
        print("  [warn] no SSL step data — skipping ssl_loss.png")
        return

    xs = [s for s, _ in ssl_steps]
    ys = [l for _, l in ssl_steps]

    # Find SSL epoch boundaries (step resets to a small value at each epoch start)
    boundaries = [xs[0]]
    for i in range(1, len(xs)):
        if xs[i] < xs[i - 1]:
            boundaries.append(xs[i])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, ys, linewidth=0.9, color="steelblue", zorder=3)

    for bx in boundaries[1:]:
        ax.axvline(bx, color="gray", linewidth=0.7, linestyle="--", alpha=0.5, zorder=2)

    # Annotate epoch numbers
    all_bounds = boundaries + [xs[-1] + 1]
    for i, (bstart, bend) in enumerate(zip(all_bounds[:-1], all_bounds[1:]), start=1):
        mid = (bstart + bend) / 2
        ax.text(mid, ax.get_ylim()[1], f"E{i}", ha="center", va="bottom",
                fontsize=7, color="gray")

    ax.set_xlabel("Global training step")
    ax.set_ylabel("SSL L1 loss (running mean)")
    ax.set_title("SSL reconstruction loss vs. training step")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — SFT loss + accuracy
# ---------------------------------------------------------------------------

def plot_sft(sft_records: list, out_path: Path) -> None:
    if not sft_records:
        print("  [warn] no SFT records — skipping sft_loss_acc.png")
        return

    # Group by SSL epoch, preserving insertion order
    groups = defaultdict(list)
    for ssl_ep, sft_ep, sl, sa, rl, ra in sft_records:
        groups[ssl_ep].append((sft_ep, sl, sa, rl, ra))
    ssl_epochs = sorted(groups.keys())
    n_ssl      = len(ssl_epochs)

    # Sequential x-axis: each SSL epoch's SFT sub-epochs appended end-to-end
    offset         = 0
    epoch_x_start  = {}
    for ssl_ep in ssl_epochs:
        epoch_x_start[ssl_ep] = offset
        offset += len(groups[ssl_ep])
    total_sft = offset

    # Color palette — one color per SSL epoch
    cmap   = cm.get_cmap("tab10" if n_ssl <= 10 else "tab20")
    colors = [cmap(i / max(n_ssl - 1, 1)) for i in range(n_ssl)]

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    for idx, ssl_ep in enumerate(ssl_epochs):
        color = colors[idx]
        recs  = groups[ssl_ep]
        start = epoch_x_start[ssl_ep]
        xs    = [start + i + 1 for i in range(len(recs))]

        sft_loss = [r[1] for r in recs]
        sft_acc  = [r[2] for r in recs]
        ref_loss = [r[3] for r in recs]
        ref_acc  = [r[4] for r in recs]

        ax_loss.plot(xs, sft_loss, color=color, linewidth=1.6, linestyle="-",  zorder=3)
        ax_loss.plot(xs, ref_loss, color=color, linewidth=1.6, linestyle="--", zorder=3)
        ax_acc.plot( xs, sft_acc,  color=color, linewidth=1.6, linestyle="-",  zorder=3)
        ax_acc.plot( xs, ref_acc,  color=color, linewidth=1.6, linestyle="--", zorder=3)

        # Vertical boundary line at the start of each SSL epoch block
        bx = xs[0] - 0.5
        ax_loss.axvline(bx, color=color, linewidth=0.7, linestyle=":", alpha=0.6, zorder=2)
        ax_acc.axvline( bx, color=color, linewidth=0.7, linestyle=":", alpha=0.6, zorder=2)

        # Label at top of loss panel
        mid = (xs[0] + xs[-1]) / 2
        ax_loss.text(mid, ax_loss.get_ylim()[1], f"E{ssl_ep}",
                     ha="center", va="bottom", fontsize=7, color=color)

    # ── Legends ──────────────────────────────────────────────────────────
    # Color legend: one patch per SSL epoch
    color_handles = [
        Patch(facecolor=colors[i], label=f"SSL epoch {ep}")
        for i, ep in enumerate(ssl_epochs)
    ]
    ncol = max(1, n_ssl // 5)
    ax_loss.legend(handles=color_handles, loc="upper right",
                   fontsize=7, ncol=ncol, title="SSL epoch", title_fontsize=7)

    # Style legend: solid vs dashed
    style_handles = [
        Line2D([0], [0], color="gray", linewidth=1.6, linestyle="-",  label="SSL features"),
        Line2D([0], [0], color="gray", linewidth=1.6, linestyle="--", label="raw charge"),
    ]
    ax_acc.legend(handles=style_handles, loc="lower right", fontsize=8)

    ax_loss.set_ylabel("Focal loss")
    ax_loss.set_title("SFT training — solid: SSL features  |  dashed: raw charge reference")
    ax_loss.grid(True, alpha=0.25)

    ax_acc.set_xlabel("Sequential SFT epoch (across all SSL epochs)")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.25)

    # X-ticks: mark every SSL epoch boundary
    tick_positions = [epoch_x_start[ep] + 0.5 for ep in ssl_epochs]
    tick_labels    = [f"E{ep}\nSFT1" for ep in ssl_epochs]
    ax_acc.set_xticks(tick_positions)
    ax_acc.set_xticklabels(tick_labels, fontsize=7)
    ax_acc.set_xlim(0.5, total_sft + 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(log_path: str = "log", out_dir: str = "viz_eval"):
    log_path = Path(log_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {log_path}")
    data = parse_log(str(log_path))
    print(f"  SSL steps   : {len(data['ssl_steps'])}")
    print(f"  SFT records : {len(data['sft_records'])}")

    plot_ssl_loss(data["ssl_steps"], out_dir / "ssl_loss.png")
    plot_sft(data["sft_records"],    out_dir / "sft_loss_acc.png")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
