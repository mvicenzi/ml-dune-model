"""Class balancing - how the scored population is constructed.

Two stages, in the order they must happen:

1. Split at event level, before anything else.
   Given some metrics require training of specialized heads, events
   are splitted into training/validation samples (80/20) with a chosen seed. 
   Since the number of pixels per event varies, this means that the pixels
   might not be splitted at 80/20 in the end.
   
   NOTE: a direct pixel-level split leaks: neighbouring pixels of the same track 
   land on both sides and even an untrained backbone could score high.

2. Class-balanced pools, drawn for either training/validation set with different seeds.
   Class frequencies span orders of magnitude; an unbalanced macro-F1 is dominated by
   sampling noise on the rare classes.

Balancing is a training concern first (it stops a head learning the prior).
But it can also be needed for evaluation, e.g. to compare the performance of two heads on a balanced sample.

"""

import numpy as np

from probes.features import Features


def event_split(n_events: int, seed: int, train_frac: float = 0.8) -> np.ndarray:
    """Boolean [n_events] mask, True = train. Seeded permutation of events."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_events)
    n_train = int(train_frac * n_events)
    is_train = np.zeros(n_events, dtype=bool)
    is_train[perm[:n_train]] = True
    return is_train


def pixel_split(fx: Features, seed: int, train_frac: float = 0.8) -> np.ndarray:
    """Per-pixel train mask induced by the event-level split.
       Pixel is train if its event is train. Seeded permutation of events.
    """
    return event_split(fx.n_events, seed, train_frac)[fx.pixel_event]


def balanced_pool(candidates: np.ndarray, y: np.ndarray, classes, per_class: int,
                  seed: int) -> np.ndarray:
    """Draw up to `per_class` indices per class from `candidates` (seeded).
        If a class has fewer than `per_class` candidates, all of them are returned.
        The returned indices are shuffled, but not across classes.
    """
    rng = np.random.RandomState(seed)
    picked = []
    for c in classes:
        ci = candidates[y[candidates] == c]
        if len(ci) > per_class:
            ci = rng.choice(ci, per_class, replace=False)
        else:
            ci = ci.copy()
            rng.shuffle(ci)
        picked.append(ci)
    return np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)
