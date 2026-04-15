"""Configuration dataclass for DINO training."""

from dataclasses import dataclass


@dataclass
class DINOConfig:
    """Configuration for DUNE-DINO self-supervised training."""

    # ============ Backbone ============
    backbone_name: str = "attn_default"  # key into BACKBONE_REGISTRY (e.g., "base", "attn_default")
    feature_dim: int = 64             # backbone output channels; must match model's output
    image_size: int = 500             # spatial resolution (H = W)
    encoding_range: float = 125.0     # sinusoidal positional encoding range (bottleneck coordinate extent)

    # ============ Augmentation mode ============
    augmentation_mode: str = "masking"  # "masking" or "cropping"

    # ============ Masking ============
    mask_ratio: float = 0.5  # fraction of active pixels to mask for student (masking mode only)

    # ============ Cropping ============
    crop_n_global: int = 2               # number of global crops per image
    crop_n_local: int = 4                # number of local crops per image
    crop_global_scale: tuple = (0.4, 1.0)  # global crop area as fraction of image area
    crop_local_scale: tuple = (0.05, 0.2)  # local crop area as fraction of image area
    crop_aspect_ratio: tuple = (0.75, 1.333)  # width-to-height aspect ratio range
    crop_blur_sigma_px: float = 10.0     # Gaussian blur sigma for activity heatmap (px)
    crop_heatmap_power: float = 1.0      # exponent applied to heatmap before sampling
    crop_min_active_pixels: int = 10     # minimum active voxels required inside a crop

    # ============ EMA teacher ============
    momentum_start: float = 0.996      # EMA momentum schedule start (slow teacher update)
    momentum_end: float = 0.9999       # EMA momentum schedule end (very slow update)

    # ============ Optimizer and LR schedule ============
    lr: float = 1e-4                 # base learning rate
    min_lr: float = 1e-6             # minimum learning rate (after cosine annealing)
    weight_decay: float = 0.04       # L2 weight decay
    weight_decay_end: float = 0.4    # weight decay at end of training (cosine annealed)
    warmup_epochs: int = 1           # linear LR warmup duration
    epochs: int = 10                 # total training epochs

    # ============ Projection head ============
    use_proj_head: bool = True         # attach MLP projection head between backbone and loss
    proj_head_hidden_dim: int = 256     # inner MLP width (DINO paper uses 2048)
    proj_head_output_dim: int = 128     # output dimension of the final FC layer
    proj_head_n_layers: int = 2         # number of MLP layers before the final FC

    # ============ Loss ============
    loss_type: str = "dino"         # "cosine", "mse", or "dino"
    center_momentum: float = 0.996  # EMA decay for teacher centering
    use_centering: bool = True      # subtract running center from teacher before loss
    teacher_temp: float = 0.04      # teacher softmax temperature (only used for "dino")
    student_temp: float = 0.1       # student softmax temperature (only used for "dino")
    normalize_features: bool = False  # L2-normalise student+teacher features before loss; set to False when use_proj_head=True (head normalises internally)
    use_cov_penalty: bool = True      # add VICReg covariance decorrelation penalty
    cov_penalty_weight: float = 10  # weight for the covariance penalty term
    use_var_penalty: bool = False     # add VICReg variance penalty (hinge on per-dim std)
    var_penalty_weight: float = 1.0  # weight for the variance penalty term
    var_gamma: float = 1.0           # target minimum std per feature dimension

    # ============ Data ============
    rootdir: str = "/nfs/data/1/yuhw/cffm-data/prod-jay-100k-truth-2026-02-27"
    apa: int = 0                    # which APA to train on
    view: str = "W"                 # which wire plane view to use
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
