# gridutils/train

Submitting DINO training jobs to the SDCC HTCondor GPU pool.
For the environment setup and the SDCC directory layout, see [../README.md](../README.md).

There are two relevant files:
- [trainjob.sh](trainjob.sh): training script that runs on the worker node
- [submit.sh](submit.sh): submission script that prepares the `.sub` file and runs `condor_submit`.

Training jobs can be submitted by:

```bash
./submit.sh path/to/run_config.json
```

What it does:

1. Reads `run_name` from the JSON configuration; this is the campaign name and **must be unique**.
2. Creates `${CONDOR_OUT}/${run_name}/` (errors out if it already exists, always pick a fresh `run_name`).
3. Writes `${run_name}.sub` and submits it.
4. On the worker, [trainjob.sh](trainjob.sh) writes checkpoints/debug to `$_CONDOR_SCRATCH_DIR` and `rsync`s them back to `${CONDOR_OUT}/${run_name}/{checkpoints,debug}/` on exit (including SIGTERM).

## Job submission parameters

At the top of `submit.sh`, you can customize the directory locations as well as the job requirements for your case. Note that the dataset directory is specified directly in the `config.json` file (see below).

```bash
# output base directory on GPFS
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"

# code directory
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"

# python virtual environment
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"

# cache directory for warpconvnet and data index
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

# JOB REQUIREMENTS: memory, GPU type, etc.
REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"
```

## Multi-GPU (DDP)

Set `REQUEST_GPUS` to the number of ranks you want; nothing else about the submission
changes.

```bash
REQUEST_GPUS=6 REQUEST_CPUS=24 REQUEST_MEMORY=192000 \
  ./submit.sh path/to/run_config.json
```

`trainjob.sh` counts the GPUs Condor actually granted the slot (the comma-separated
`CUDA_VISIBLE_DEVICES`) and launches one rank per GPU under `torchrun --standalone` when
that count is above 1, or runs the interpreter directly when it is 1. Single node only.
Only rank 0 writes checkpoints, debug histories and the `[timing]` lines.

Three things the submitter does not do for you:

- `batch_size` in the config is per rank. N ranks at `batch_size: B` train at an
  effective batch of `N x B`. To hold the effective batch fixed while adding ranks,
  divide `batch_size` by N.
- `lr` is not adjusted. Raising the effective batch without scaling the learning rate
  puts the run in a different optimisation regime, so it is no longer comparable to runs
  at the old one.
- `REQUEST_CPUS` and `REQUEST_MEMORY` do not scale with `REQUEST_GPUS`. Every rank
  spawns its own `num_workers` dataloader workers, so raise both yourself.

With the sharded reader, shards are partitioned over `world_size x num_workers` readers
and every reader gets the same (ceiling) number of shards, so the padding waste grows
with the rank count — prefer fewer `num_workers` per rank at high `REQUEST_GPUS`. If
`world_size x num_workers` exceeds the number of full shards the container holds, the
dataset raises at startup (`N full shards cannot feed M readers`); lower `num_workers`
or use fewer ranks.

`NCCL_P2P_DISABLE` and `NCCL_IB_DISABLE` are exported by `trainjob.sh`. The L40S nodes
have no NVLink and their GPU-to-GPU P2P transport hangs: the process group initialises
and the startup broadcasts succeed, but the first AllReduce never returns and the job
sits until the watchdog kills it. Both are set to `1` unless already present in the
environment.

## Training configuration

An example training configuration is provided in [config.json](config.json).
The full list of accepted keys is whatever `dino.train_dino.from_config` consumes, see [dino/train_dino.py](../../dino/train_dino.py).

Copy [config.json](config.json) and edit. Key fields:

- `run_name` — **must be unique**; defines the output directory under `CONDOR_OUT`.
- `datadir`, `apa`, `view`, `image_h`, `image_w`, `n_subset` — dataset selection.
- `use_sharded`/`sharded_dir`, `use_packed`/`packed_path` — pre-built container selection (see [../datagen/](../datagen/)).
- `batch_size`, `num_workers` — dataloader. `batch_size` is **per rank**; see [Multi-GPU (DDP)](#multi-gpu-ddp).
- `backbone_name`, `feature_dim`, `proj_head_*`, `encoding_range` — model.
- `augmentation_mode`, `crop_*`, `mask_type`, `mask_ratio`, `mask_region_*` — augmentation
  pipeline. `mask_type` is `region` (whole cells of a fixed grid, the default), `block`
  (windows around active voxels) or `pixel` (per-voxel dropout). Under `region` the masked
  fraction comes from `mask_region_wipe_max`, not `mask_ratio`.
- `objective`, `lambda_charge`, `lambda_occ`, `occ_max_neg`, `occ_neg_per_pos`,
  `occ_grow_iters` — training objective. The last three are `mae` only, and which of them
  applies depends on `mask_type`: `region` enumerates its occupancy candidates and needs a
  cap on them (`occ_max_neg` / `occ_neg_per_pos`, measured with
  [../diagnostics/measure_occ_candidates.sub](../diagnostics/measure_occ_candidates.sub)),
  while the other mask types grow candidates and take `occ_grow_iters` instead. Setting a
  key that does not apply is rejected at submit rather than ignored.
- `epochs`, `lr`, `min_lr`, `weight_decay*`, `warmup_epochs`, `momentum_*` — schedule.
- `teacher_temp`, `student_temp`, `use_centering`, `use_cov_penalty`, `use_var_penalty` — loss.
- `save_every`, `debug`, `debug_every` — checkpointing / debug dumps.

The other `config_*.json` files in this directory are the configurations of past campaigns, kept for reference.

## Outputs

```
${CONDOR_OUT}/${run_name}/
├── ${run_name}.sub                    # generated submit file
├── <ClusterId>.<ProcId>.{out,err,log} # condor logs
├── checkpoints/                       # rsynced from scratch
└── debug/                             # rsynced from scratch, includes config.json
```
