"""
test_shard_partition.py
───────────────────────
The DDP shard split, checked against the real shard set.

Under DDP the sharded reader hands each of `world_size * num_workers` readers its own
slice of the shard list. Two things have to hold or a multi-GPU run fails in ways that do
not look like data bugs:

  • the slices must cover every full shard. Readers are levelled up to a common count
    by repeating a few shards (as DistributedSampler pads), so a small, accounted-for
    overlap is expected -- but a shard that reaches no reader is data silently unused,
    and one that reaches many is data silently over-weighted;
  • every reader must yield the same number of batches. DDP's gradient all-reduce is a
    collective, so a rank that runs out early leaves the others blocked in it until the
    job is killed by wall clock -- with no error message.

The second is why the short trailing shard matters: this set is 199 shards of 1000 plus
one of 870, so a reader holding the short one would come up two batches short at
batch_size=100 and hang the job.

Runs on CPU: it only reads shard lengths and metadata, never samples.

Run:  python -u tests/test_shard_partition.py
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader.apa_sparse_sharded_dataset import APASparseShardedDataset

SHARD_DIR = "/gpfs01/lbne/users/fm/cffm-data/shards_prod-jay-2026-06-11_mixed_apa0W"
BATCH = 100


def _readers(world_size, num_workers, epoch=0, batch_size=BATCH):
    """Every reader's shard list, as {(rank, worker): [names]}, plus one dataset."""
    out = {}
    ds0 = None
    for rank in range(world_size):
        ds = APASparseShardedDataset(
            root_dir=SHARD_DIR, batch_size=batch_size,
            rank=rank, world_size=world_size, num_workers=num_workers,
        )
        ds.set_epoch(epoch)
        ds0 = ds0 or ds
        for w in range(max(num_workers, 1)):
            out[(rank, w)] = [p.name for p in ds._ddp_shards(w, max(num_workers, 1))]
    return out, ds0


def check_every_full_shard_is_covered(device=None):
    """All 199 full shards are read each epoch, and repeats are only the padding."""
    for world_size, num_workers in ((8, 5), (8, 4), (4, 4), (8, 1)):
        readers, ds = _readers(world_size, num_workers)
        assigned = [name for names in readers.values() for name in names]
        n_readers = world_size * max(num_workers, 1)
        per_reader = len(readers[(0, 0)])
        expected_pad = n_readers * per_reader - ds._n_full_shards
        dupes = sum(c - 1 for c in Counter(assigned).values() if c > 1)
        print(f"  world={world_size} workers={num_workers}: {len(set(assigned))}"
              f"/{ds._n_full_shards} covered, {dupes} repeated (padding {expected_pad})")
        assert len(set(assigned)) == ds._n_full_shards, "a full shard reached no reader"
        assert dupes == expected_pad, f"{dupes} repeats, expected exactly {expected_pad}"


def check_every_reader_gets_the_same_count(device=None):
    """Unequal counts deadlock the all-reduce; check several rank/worker splits."""
    for world_size, num_workers in ((8, 5), (8, 4), (4, 4), (2, 3), (8, 1)):
        readers, ds = _readers(world_size, num_workers)
        counts = {len(v) for v in readers.values()}
        n_readers = world_size * max(num_workers, 1)
        print(f"  world={world_size} workers={num_workers}: {n_readers} readers, "
              f"{sorted(counts)} shards each")
        assert len(counts) == 1, f"readers differ in shard count: {sorted(counts)}"


def check_short_shard_is_excluded(device=None):
    """The trailing remainder shard must never be assigned: its reader would run short.

    `create_shards.py:110` chunks the index list contiguously, so every shard holds
    exactly shard_size samples except the last, which takes the remainder. The lengths
    are read from the two files rather than taken from the reader's own `_n_full_shards`,
    so this checks the reader's arithmetic instead of restating it.
    """
    import h5py

    readers, ds = _readers(world_size=8, num_workers=5)

    def length(path):
        with h5py.File(path, "r") as f:
            return int(f["offsets"].shape[0]) - 1

    first, last = length(ds.shards[0]), length(ds.shards[-1])
    assigned = {name for names in readers.values() for name in names}

    print(f"  {len(ds.shards)} shards: first={first}, last={last}, "
          f"{len(assigned)} distinct assigned")
    assert last < first, "expected this set to end with a short remainder shard"
    assert ds.shards[-1].name not in assigned, "the short trailing shard was assigned"
    assert len(assigned) == len(ds.shards) - 1, (
        f"expected every shard but the last: {len(assigned)} of {len(ds.shards)}"
    )


def check_len_matches_the_partition(device=None):
    """__len__ is per rank, and the schedules are built from it."""
    for world_size, num_workers in ((8, 5), (8, 4), (4, 4)):
        readers, ds = _readers(world_size, num_workers)
        per_reader = len(readers[(0, 0)])
        expected = ((per_reader * ds._per_shard) // BATCH) * max(num_workers, 1)
        print(f"  world={world_size} workers={num_workers}: len={len(ds)} "
              f"(expected {expected})")
        assert len(ds) == expected, f"__len__ {len(ds)} != {expected}"


def check_single_process_path_is_unchanged(device=None):
    """world_size=1 must behave exactly as before: whole list, no exclusions."""
    ds = APASparseShardedDataset(root_dir=SHARD_DIR, batch_size=BATCH)
    with_meta = 199870 // BATCH
    print(f"  world=1: len={len(ds)} (expected {with_meta}), shards={len(ds.shards)}")
    assert len(ds) == with_meta, "single-process __len__ changed"
    assert len(ds.shards) == 200, "single-process shard list changed"


def check_epoch_changes_the_assignment(device=None):
    """A new epoch reshuffles, so a different shard sits out and readers swap work."""
    r0, _ = _readers(world_size=8, num_workers=5, epoch=0)
    r1, _ = _readers(world_size=8, num_workers=5, epoch=1)
    same = sum(1 for k in r0 if r0[k] == r1[k])
    covered0 = {n for v in r0.values() for n in v}
    covered1 = {n for v in r1.values() for n in v}
    print(f"  readers with an identical list across epochs: {same}/40; "
          f"shards used ep0={len(covered0)} ep1={len(covered1)}, "
          f"differing={len(covered0 ^ covered1)}")
    assert same < 40, "the epoch seed did not change the assignment"


def check_all_ranks_agree_on_the_shuffle(device=None):
    """Ranks must permute the full list identically, or the split is not a partition."""
    readers, ds = _readers(world_size=8, num_workers=5, epoch=3)
    assigned = [n for v in readers.values() for n in v]
    per_reader = len(readers[(0, 0)])
    # If ranks permuted differently, slices would collide and coverage would fall well
    # below the full-shard count while the repeat count climbed.
    print(f"  epoch 3: {len(set(assigned))} of {ds._n_full_shards} covered, "
          f"{len(assigned)} slots ({40} readers x {per_reader})")
    assert len(set(assigned)) == ds._n_full_shards, (
        "coverage is below the full-shard count — ranks disagree on the shuffle order"
    )


def check_too_many_readers_raises(device=None):
    """More readers than full shards cannot be split; say so rather than yield nothing."""
    try:
        APASparseShardedDataset(root_dir=SHARD_DIR, batch_size=BATCH,
                                rank=0, world_size=64, num_workers=8)
    except ValueError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("an unsplittable reader count was accepted")


CHECKS = [
    ("every_full_shard_is_covered", check_every_full_shard_is_covered),
    ("every_reader_gets_the_same_count", check_every_reader_gets_the_same_count),
    ("short_shard_is_excluded", check_short_shard_is_excluded),
    ("len_matches_the_partition", check_len_matches_the_partition),
    ("single_process_path_is_unchanged", check_single_process_path_is_unchanged),
    ("epoch_changes_the_assignment", check_epoch_changes_the_assignment),
    ("all_ranks_agree_on_the_shuffle", check_all_ranks_agree_on_the_shuffle),
    ("too_many_readers_raises", check_too_many_readers_raises),
]


def main():
    failures = []
    for name, fn in CHECKS:
        print(f"\n[{name}]")
        try:
            fn()
            print("  PASS")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print("  FAIL")

    print("\n" + "=" * 60)
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
