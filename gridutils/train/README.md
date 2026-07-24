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

## Training configuration

An example training configuration is provided in [config.json](config.json).
The full list of accepted keys is whatever `dino.train_dino.from_config` consumes, see [dino/train_dino.py](../../dino/train_dino.py).

Copy [config.json](config.json) and edit. Key fields:

- `run_name` — **must be unique**; defines the output directory under `CONDOR_OUT`.
- `datadir`, `apa`, `view`, `image_h`, `image_w`, `n_subset` — dataset selection.
- `use_sharded`/`sharded_dir`, `use_packed`/`packed_path` — pre-built container selection (see [../datagen/](../datagen/)).
- `batch_size`, `num_workers` — dataloader.
- `backbone_name`, `feature_dim`, `proj_head_*`, `encoding_range` — model.
- `augmentation_mode`, `crop_*`, `mask_ratio` — augmentation pipeline.
- `epochs`, `lr`, `min_lr`, `weight_decay*`, `warmup_epochs`, `momentum_*` — schedule.
- `loss_type`, `teacher_temp`, `student_temp`, `use_centering`, `use_cov_penalty`, `use_var_penalty` — loss.
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
