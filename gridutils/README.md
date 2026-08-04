# gridutils

Helpers for setting up the environment and submitting jobs to the SDCC HTCondor GPU pool.

## Layout

- `build_env.sh` / `build.sub` — one-time environment setup (shared by all job types, see below).
- [datagen/](datagen/README.md) — dataset container generation (shards / packs), CPU-only jobs.
- [train/](train/README.md) — training jobs, plus the run configuration JSONs.
- [diagnostics/](diagnostics/README.md) — post-training feature extraction and diagnostics plots.

Each subdirectory has its own README with usage instructions; all follow the
same pattern: a `submit_*.sh` script run from the login node that writes a
`.sub` file and calls `condor_submit`, and a worker-node `*job.sh` script.

## Directory structure (SDCC)

Both your `$HOME` and `/gpfs01/lbne/users/fm` (shared group area) are visible from the work nodes. However, quotas are very different. The suggested directory layout is:

- `$HOME/ml-dune-model`: clone the repo here. Keep all code in `$HOME`.
- `/gpfs01/lbne/users/fm/${USER}/`: your personal area on the group GPFS volume. Create it once. Inside it:
  - `/gpfs01/lbne/users/fm/${USER}/uvenv/`: python virtual environment. Lives here because it can get large (~10GB).
  - `/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT/`: training run outputs (checkpoints, debug, condor logs). Subdirectories are created automatically by `submit.sh` for each run.
  - `/gpfs01/lbne/users/fm/${USER}/cache/`: will store `warpconvnet/` and `data/` caches.
- `/gpfs01/lbne/users/fm/cffm-data/`: **shared** dataset area, available to anyone. Point `datadir` in your config here.

## Setting up the environment

One-time setup is handled by [build_env.sh](build_env.sh) and [build.sub](build.sub). 
Some packages can be installed directly from the interactive node (thanks to pre-built wheels), but GPU availability is needed when building from source and so the script must be run as a job (via `build.sub`).
See the comments in the script, and the instructions below:

```bash
# 0. get uv package
pip install uv

# 1. create the venv (only needed once)
uv venv /gpfs01/lbne/users/fm/${USER}/uvenv --python 3.11

# 2. CPU-only installs via script
./build_env.sh

# 2b. GPU installs (needed for flash-attn / local warpconvnet builds)
# edit build_env.sh to uncomment the relevant blocks, then:
condor_submit build.sub
```

Edit the env vars at the top of `build_env.sh` (`CUDA`, `TORCH_REL`, `WARPCONV_REL`, `USER`) to match your target stack before running.
