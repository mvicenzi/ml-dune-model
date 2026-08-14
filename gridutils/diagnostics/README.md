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

## Collecting a whole run

- [collect_probes.sh](collect_probes.sh): drives extraction and probes over many
  checkpoints, waiting for each.

Unlike the `submit_*.sh` scripts above, this one does **not** submit a job and
return — it runs on the login node and blocks, submitting work as each
checkpoint becomes available. That is what lets it be pointed at a training job
that is still running.

```bash
./collect_probes.sh <run_name> [epoch...] [options] [extraction flags...]

# examples:
./collect_probes.sh myrun                              # every checkpoint present now
./collect_probes.sh myrun 60 70 80 90 100              # wait for each in turn
./collect_probes.sh myrun 100 --max_images=10000 --features_prefix=features_10k_

# left running against a live training job:
nohup bash collect_probes.sh myrun 60 70 80 90 100 \
    --max_images=10000 --features_prefix=features_10k_ > collect.log 2>&1 &
```

Bare integers are epochs; unrecognised `--flags` are forwarded to
`submit_extract.sh`. `--features_prefix` is also passed on as
`--output_prefix`, so extraction and scoring cannot name different files.

It is safe to re-run: an epoch whose result JSONs already exist is skipped
unless `--force` is given. Epochs it gave up waiting for are listed as
*abandoned* and make the script exit non-zero, so a merge chained after it
cannot quietly tabulate an incomplete sweep.

Three guards are worth knowing about, all of them responses to failures seen in
practice:

- **Serialisation.** Only `--parallel` probe sweeps (default 1) are allowed in
  the queue at once. Two concurrent sweeps contend on GPFS badly: `probe_overlap`
  measured 2905 s and 4043 s with two jobs in flight against 1963 s alone.
- **`.npz` integrity.** A features file is checked by opening the zip and
  requiring the members every extraction writes — not by size or mtime, both of
  which look correct on a file that is still being written. The check must hold
  twice, `--settle` seconds apart, because a duplicate extraction overwriting the
  file in place looks complete both before and after but not during.
- **Duplicate extraction.** An epoch is skipped when an extraction for that
  checkpoint is already queued. Note this is tested with `condor_q -af Cmd Args`:
  `condor_q -nobatch` prints arguments truncated to the terminal width, and a
  checkpoint path sits far enough into `extractjob.sh`'s arguments to be cut off.
