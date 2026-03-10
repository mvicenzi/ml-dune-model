"""Configuration dataclass for DINO training."""

from dataclasses import dataclass


@dataclass
class DINOConfig:
    """Configuration for DUNE-DINO self-supervised training."""

    # ============ Backbone ============
    backbone_name: str = "attn_default"  # key into BACKBONE_REGISTRY (e.g., "base", "attn_default")
    feature_dim: int = 64             # backbone output channels; must match model's output
    image_size: int = 500             # spatial resolution (H = W)

    # ============ Masking ============
    mask_ratio: float = 0.5  # fraction of active pixels to mask for student

    # ============ EMA teacher ============
    momentum_start: float = 0.996      # EMA momentum schedule start (slow teacher update)
    momentum_end: float = 0.9999       # EMA momentum schedule end (very slow update)

    # ============ Optimizer and LR schedule ============
    lr: float = 1e-4                 # base learning rate
    min_lr: float = 1e-6             # minimum learning rate (after cosine annealing)
    weight_decay: float = 0.04       # L2 weight decay
    weight_decay_end: float = 0.4    # weight decay at end of training (cosine annealed)
    warmup_epochs: int = 5           # linear LR warmup duration
    epochs: int = 100                # total training epochs

    # ============ Loss ============
    loss_type: str = "cosine"    # "cosine", "mse", or "dino"
    center_momentum: float = 0.9  # EMA decay for teacher centering
    use_centering: bool = True    # subtract running center from teacher before loss
    teacher_temp: float = 1.0    # teacher softmax temperature (only used for "dino")
    student_temp: float = 1.0    # student softmax temperature (only used for "dino")

    # ============ Data ============
    rootdir: str = "/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd"
    view_index: int = 2              # which wire plane view to use (0, 1, or 2)
    n_subset: int = -1               # -1 = full dataset; otherwise use first N samples for testing
    batch_size: int = 16
    num_workers: int = 4

    # ============ Checkpointing ============
    output_dir: str = "./dino_checkpoints"
    save_every: int = 10  # save checkpoint every N epochs

    # ============ Debug / Visualization ============
    debug: bool = False               # enable detailed logging and history tracking
    debug_every: int = 100            # log scalars / stats / grad norms every N batches
    debug_dir: str = "./dino_debug"   # base directory for debug outputs
    run_name: str = ""                # optional label; outputs go to debug_dir/run_name/ if set
