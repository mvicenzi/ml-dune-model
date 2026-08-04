# gridutils/datagen

Generating dataset containers (HDF5 shards / packed `.npz`) as CPU-only Condor jobs.
For the environment setup and the SDCC directory layout, see [../README.md](../README.md).

There are two relevant files:
- [datajob.sh](datajob.sh): worker-node script that runs an arbitrary `loader.*` module
- [submit_datagen.sh](submit_datagen.sh): submission script that prepares the `.sub` file and runs `condor_submit`.

```bash
./submit_datagen.sh <job_name> <module> [args...]
```

Everything after `<module>` is forwarded verbatim to `python -m <module>`.
Logs go to `${CONDOR_OUT}/datagen/<job_name>/`. Job requirements can be
overridden via env vars (`REQUEST_MEMORY`, `REQUEST_CPUS`, ...), as for training jobs.

## Examples

```bash
DATA=/gpfs01/lbne/users/bnayak/cffm-data/prod-jay-100k-truth-2026-06-11
OUT=/gpfs01/lbne/users/fm/${USER}/cffm-data

# full training shards (full truth; extraction reads the same set):
./submit_datagen.sh shards_mixed_apa0W \
    loader.create_shards --datadir $DATA --apa 0 --view W \
    --outdir $OUT/shards_prod-jay-2026-06-11_mixed_apa0W \
    --cache_dir /gpfs01/lbne/users/fm/${USER}/cache/data \
    --with_extra_truth --num_workers 8

# packed .npz (RAM-heavy: ~45 GB peak at 200k events):
REQUEST_MEMORY=64000 ./submit_datagen.sh pack_mixed_apa0W \
    loader.pack_dataset --datadir $DATA --apa 0 --view W \
    --out_path $OUT/packed/prod-jay-2026-06-11_mixed_apa0W.npz \
    --cache_dir /gpfs01/lbne/users/fm/${USER}/cache/data --num_workers 8
```

Note: jobs sharing a `--cache_dir` each scan the dataset if the index cache
is missing (the write is atomic, so concurrent jobs are safe — just wasteful).
If the cache doesn't exist yet, let one job build it before submitting the others.

See `docs/09` / `docs/10` for the sharded-vs-packed design discussion and
benchmark results, and `docs/02` for the container formats themselves.
