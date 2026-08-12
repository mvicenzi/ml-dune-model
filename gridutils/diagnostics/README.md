# gridutils/diagnostics

Post-training GPU jobs that operate on a completed run's checkpoints:
feature extraction first, then diagnostics plots on the extracted features.
For the environment setup and the SDCC directory layout, see [../README.md](../README.md).

## Feature extraction

- [extractjob.sh](extractjob.sh): worker-node script
- [submit_extract.sh](submit_extract.sh): submission script — one Condor job per checkpoint.

```bash
./submit_extract.sh <run_name> [epoch...] [extra_args...]

# examples:
./submit_extract.sh myrun                          # all checkpoints in the run
./submit_extract.sh myrun 10                       # epoch 10 only
./submit_extract.sh myrun 10 50 100                # epochs 10, 50, and 100
./submit_extract.sh myrun 10 --max_images=5000     # epoch 10, limit images
```

Bare integers are parsed as epochs; `--flag` style args are forwarded verbatim
to `probes.extract_features` (in addition to `--pixel_truth`, which
is always passed; see [extractjob.sh](extractjob.sh)).
Logs go to `${CONDOR_OUT}/<run_name>_extract/`.

## Diagnostics plots

- [diagnosticsjob.sh](diagnosticsjob.sh): worker-node script
- [submit_diagnostics.sh](submit_diagnostics.sh): submission script — one Condor job per epoch.

Requires the features extracted above (`submit_extract.sh` first).

```bash
./submit_diagnostics.sh <run_name> [epoch...] [extra_args...]

# examples:
./submit_diagnostics.sh myrun                          # epoch 100 (default)
./submit_diagnostics.sh myrun 10 50 100                # epochs 10, 50, and 100
./submit_diagnostics.sh myrun 100 --max_pixels_per_class=30000
```

Extra flags are forwarded verbatim to `plot_knn_pixel`.
Logs go to `${CONDOR_OUT}/<run_name>_diag/`.

## Probe suite

- [probesjob.sh](probesjob.sh): worker-node script
- [submit_probes.sh](submit_probes.sh): submission script — one Condor job per epoch.

Requires the features extracted above. CPU-only: the probes read the features
`.npz` and import neither warpconvnet nor a dataset reader, so these jobs skip
the GPU queue entirely.

```bash
./submit_probes.sh <run_name> [epoch...] [extra_args...]

# examples:
./submit_probes.sh myrun 100                                  # every measurement
FEATURES_PREFIX=features_10k_ ./submit_probes.sh myrun 100     # non-default extraction
PROBE_STAGES=embed ./submit_probes.sh myrun 100                # the 2-D maps only
```

`PROBE_STAGES` picks the stages: `pid,knn,overlap,instance,vertex,event` (the
default, all of them measurements) plus `embed`, which is opt-in.

`embed` runs [`probes.plot_embedding`](../../probes/plot_embedding.py) with
`--mode=both`, so one read of the 7 GB features file draws both the pixel-PID and
the event-flavor UMAP/t-SNE map. It writes to `probes/ep<N>/`:

```
embedding_pid.png    embedding_pid_<features stem>.npz
embedding_event.png  embedding_event_<features stem>.npz
```

The `.npz` holds just the 2-D points and their labels, a few MB. Restyle from
that on a login node — instantly, and as often as you like — rather than
resubmitting:

```bash
python -m probes.plot_embedding probes/ep100/embedding_pid_features_10k_ep100.npz \
    --out_dir figures/ --point_size=1
```

Per-probe knobs go through `<STAGE>_EXTRA_ARGS` (`EMBED_EXTRA_ARGS` for this one),
because `getenv = False` and a bare extra arg is forwarded to *every* probe.
Logs go to `${CONDOR_OUT}/<run_name>_probes/`.

## One-off submit files

The `*.sub` files in this directory (`diag_remaining_*.sub`,
`scatter_ep100_*.sub`) are hand-written one-off submit files from past
campaigns, kept for reference. They hardcode absolute paths.
