"""2-D embedding of frozen features, coloured by pixel PID or by event flavor.

Two questions, one per mode:

  pid     Do pixels of the same particle type land together?  Samples pixels,
          reduces their 64-d features to 2-D, colours by `pixel_labels`.
  event   Do events of the same interaction flavor land together?  Mean-pools
          each event, reduces, colours by `labels` (numuCC / nueCC / NC).

This is the one plot that cannot be drawn from a result JSON: it needs the
feature vectors themselves, so it reads the extraction .npz. That file is ~7 GB
and decompresses in full, which a login node will not survive — run the reducing
step on a worker, then redraw from the cached embedding as often as you like:

  # heavy, once (condor)
  python -m probes.plot_embedding FEATURES.npz --mode pid --embed_out emb_pid.npz

  # light, instant, iterate on the styling
  python -m probes.plot_embedding emb_pid.npz --out_dir figures/

The cache holds only the 2-D points and their labels, so it is a few MB and the
second command is the one to keep running. Passing a features file always writes
the cache, so the expensive pass is never repeated by accident.

`--mode both` does the pid and the event picture from a single read. The read is
most of the wall clock — the .npz is ~7 GB of single-threaded zlib, minutes of it
before any reducing starts — so running the two modes as separate commands pays
that cost twice for nothing.

Sampling is the part that decides whether the picture means anything, and it is
not uniform. For `pid` it reuses `probe_knn_pid.collect`, which caps how many
pixels of one class come from any single event: without that cap an abundant
class fills its quota from a handful of events and the plot shows one shower's
pixels sitting together rather than showers in general. For `event` it reuses
`probe_event.mean_pool` and its per-event pixel cap.

A 2-D reduction is a picture, not a measurement: neighbourhoods are distorted by
construction and distances between clusters carry no meaning. Read it alongside
the k-NN and probe numbers, never instead of them.
"""

import argparse
from pathlib import Path

import numpy as np

from probes.features import load_features
from probes.probe_event import FLAVOR_NAMES, mean_pool
from probes.probe_knn_pid import PIXEL_CLASS_NAMES, _label_to_class, collect

# Okabe-Ito, colourblind-safe, in a fixed order so a class keeps its colour
# between figures. Yellow (#F0E442) is deliberately absent: too low-contrast.
CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                "#000000"]

DEFAULT_MAX_PIXELS_PER_CLASS = 5000
DEFAULT_MAX_EVENTS = 4000


def reduce_2d(X: np.ndarray, method: str = "auto", seed: int = 0):
    """Reduce to 2-D. Returns (embedding, name of the method actually used).

    PCA first when the input is wide: both UMAP and t-SNE are far slower on 64
    raw dimensions than on the ~20 that hold the variance, and neither is
    sensitive to the difference.
    """
    X = np.asarray(X, dtype=np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"

    if X.shape[1] > 24:
        from sklearn.decomposition import PCA
        X = PCA(n_components=24, random_state=seed).fit_transform(X)

    if method in ("auto", "umap"):
        try:
            import umap
            red = umap.UMAP(n_components=2, random_state=seed)
            return red.fit_transform(X), "UMAP"
        except ImportError:
            if method == "umap":
                raise SystemExit("umap-learn is not installed; use --reducer=tsne")

    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=seed,
                init="pca").fit_transform(X), "t-SNE"


def embed_pid(fx, args):
    """(points, class index per point, class names) for the pixel PID mode."""
    fx.require("pixel_labels")
    cls = _label_to_class(fx.truth["pixel_labels"])
    pools, labels, counts, n_events, cap = collect(
        {"f": fx.feat}, cls, fx.offsets, args.max_pixels_per_class,
        seed=args.seed, max_pixels_per_image=args.max_pixels_per_image)
    print(f"  pooled {len(labels)} pixels, per-image cap "
          f"{'uncapped' if cap is None else cap}")
    for i, name in enumerate(PIXEL_CLASS_NAMES):
        print(f"    {name:<9s} {int(counts[i]):>7d} px from {int(n_events[i]):>5d} events")
    return pools["f"], labels, list(PIXEL_CLASS_NAMES)


def embed_event(fx, args):
    """(points, flavor index per event, flavor names) for the event mode."""
    pooled = mean_pool(fx.feat, fx.offsets, args.max_pixels_per_event, args.seed)
    y = fx.labels
    keep = np.isfinite(pooled).all(1) & (y >= 0)
    n_drop = int((~keep).sum())
    pooled, y = pooled[keep], y[keep]
    if len(pooled) > args.max_events:
        sel = np.random.RandomState(args.seed).choice(
            len(pooled), args.max_events, replace=False)
        pooled, y = pooled[sel], y[sel]
    names = list(FLAVOR_NAMES)
    print(f"  pooled {len(pooled)} events ({n_drop} dropped: empty or unknown flavor)")
    for i, name in enumerate(names):
        print(f"    {name:<9s} {int((y == i).sum()):>6d} events")
    return pooled, y, names


def draw(emb, y, names, title: str, out_path: Path, point_size: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    # Drawn largest class first so the rare ones stay visible on top of it.
    order = sorted(range(len(names)), key=lambda c: -int((y == c).sum()))
    for c in order:
        m = y == c
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=point_size, alpha=0.5,
                   color=CLASS_COLORS[c % len(CLASS_COLORS)],
                   label=f"{names[c]} ({int(m.sum())})", linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    # Legend markers sized independently of the points, which are deliberately
    # tiny in a dense scatter.
    leg = ax.legend(fontsize=9, markerscale=4, loc="best")
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def load_cache(path: Path):
    """(embedding, labels, names, mode, reducer) from a cached embedding file."""
    d = np.load(path, allow_pickle=True)
    return (d["embedding"], d["labels"], [str(s) for s in d["names"]],
            str(d["mode"]), str(d["reducer"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="features .npz, or a cached embedding .npz to redraw")
    ap.add_argument("--mode", default="pid", choices=["pid", "event", "both"],
                    help="colour pixels by particle type, or events by flavor; "
                         "both = one read, two figures")
    ap.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne", "pca"],
                    help="auto = UMAP when installed, else t-SNE")
    ap.add_argument("--embed_out", default="",
                    help="where to cache the 2-D points (default: beside --out_dir)")
    ap.add_argument("--out_dir", default="figures", help="directory for the PNG")
    ap.add_argument("--name", default="", help="output file stem (default: embedding_<mode>)")
    ap.add_argument("--source", default="student", choices=["student", "teacher"])
    ap.add_argument("--max_pixels_per_class", type=int,
                    default=DEFAULT_MAX_PIXELS_PER_CLASS,
                    help=f"pid mode: pixels per class (default "
                         f"{DEFAULT_MAX_PIXELS_PER_CLASS}; a scatter saturates "
                         f"long before a k-NN pool would)")
    ap.add_argument("--max_pixels_per_image", type=int, default=0,
                    help="pid mode: cap per class per event (0 = auto, spreading "
                         "each pool over ~200 events; negative = uncapped)")
    ap.add_argument("--max_pixels_per_event", type=int, default=2000,
                    help="event mode: pixels pooled per event (default 2000)")
    ap.add_argument("--max_events", type=int, default=DEFAULT_MAX_EVENTS,
                    help=f"event mode: events plotted (default {DEFAULT_MAX_EVENTS})")
    ap.add_argument("--point_size", type=float, default=0.0,
                    help="scatter marker size (0 = auto: 2 for the dense pixel "
                         "scatter, 12 for the far sparser event one)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"{path}: not found")
    out_dir = Path(args.out_dir)

    with np.load(path, allow_pickle=True) as probe:
        cached = "embedding" in probe.files

    if cached:
        emb, y, names, mode, reducer = load_cache(path)
        print(f"redrawing {len(emb)} cached points from {path} ({mode}, {reducer})")
        render(emb, y, names, mode, reducer, path.stem, out_dir, args)
        return

    modes = ["pid", "event"] if args.mode == "both" else [args.mode]
    # Both name a single file, so they cannot describe two figures at once.
    if len(modes) > 1 and (args.name or args.embed_out):
        raise SystemExit("--name and --embed_out name one output; they need "
                         "--mode pid or --mode event, not --mode both")

    # One read serves every mode: it is the expensive part of this program.
    fx = load_features(path, source=args.source)
    print(f"{run_label_of(path)}  modes={','.join(modes)}  events={fx.n_events} "
          f"pixels={fx.n_pixels} D={fx.feat.shape[1]}")

    for mode in modes:
        print(f"[{mode}]")
        X, y, names = (embed_pid(fx, args) if mode == "pid"
                       else embed_event(fx, args))
        print(f"  reducing {X.shape[0]}x{X.shape[1]} to 2-D ...")
        emb, reducer = reduce_2d(X, args.reducer, args.seed)
        cache = (Path(args.embed_out) if args.embed_out
                 else out_dir / f"embedding_{mode}_{path.stem}.npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embedding=emb, labels=y, names=np.array(names),
                 mode=mode, reducer=reducer)
        print(f"  cached embedding -> {cache}   (redraw from this, not the .npz)")
        render(emb, y, names, mode, reducer, path.stem, out_dir, args)


def render(emb, y, names, mode, reducer, label, out_dir: Path, args) -> None:
    stem = args.name or f"embedding_{mode}"
    unit = "pixels by particle type" if mode == "pid" else "events by flavor"
    size = args.point_size or (2.0 if mode == "pid" else 12.0)
    draw(emb, y, names, f"{reducer} of frozen features — {unit}\n{label}",
         out_dir / f"{stem}.png", size)


def run_label_of(path: Path) -> str:
    from probes.results import run_label
    return run_label(path, "student")


if __name__ == "__main__":
    main()
