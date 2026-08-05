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

## One-off submit files

The `*.sub` files in this directory (`diag_remaining_*.sub`,
`scatter_ep100_*.sub`) are hand-written one-off submit files from past
campaigns, kept for reference. They hardcode absolute paths.
