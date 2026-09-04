"""Configuration for DINO training, and the rules a configuration must satisfy.

The rules live here rather than in the training script because they are properties of a
configuration, not of a run: they read a DINOConfig and nothing else, so they can be
checked at submit time, in a test, or from a notebook without building anything.
"""

import inspect
from dataclasses import dataclass, fields

from models import BACKBONE_REGISTRY


@dataclass
class DINOConfig:
    """Configuration for DUNE-DINO self-supervised training."""

    # ============ Run ============
    run_name: str = ""                # optional label; outputs go to debug_dir/run_name/ if set
    seed: int = 42                    # torch RNG, loader shuffles and the DDP shard order;
                                      # change it to repeat a run as an independent sample

    # ============ Data ============
    # Production root: mixes nominal + nueswap flavors (point at a flavor
    # subdirectory for a single-flavor dataset).
    datadir: str = "/gpfs01/lbne/users/bnayak/cffm-data/prod-jay-100k-truth-2026-06-11"
    apa: int = 0                      # which APA to train on
    view: str = "W"                   # which wire plane view ("U", "V", or "W")
    image_h: int = 1500               # spatial resolution: height (time ticks)
    image_w: int = 1050               # spatial resolution: width (wire channels)
    n_subset: int = -1                # -1 = full dataset; otherwise cap at N samples
    batch_size: int = 100         # per rank under DDP: effective batch is batch_size * world_size
    num_workers: int = 5          # 5 x 8 ranks divides the 199 full shards with one repeat; 4 costs 25
    cache_dir: str = "./data"           # directory for cached dataset index .pt file
    use_sharded: bool = False           # use pre-sharded HDF5 dataset (loader/create_shards.py)
    sharded_dir: str = ""               # path to directory containing shard_*.h5 files
    buffer_size: int = 3000             # shuffle-buffer size (samples) for sharded dataset
    use_packed: bool = False            # use packed .npz dataset (loader/pack_dataset.py)
    packed_path: str = ""               # path to the packed .npz file
    use_log_transform: bool = True     # apply FeatureLogTransform to raw charge before model
    feat_min_val: float = 3.75        # 2nd percentile of raw charge [ADC]; anchors y = -1
    feat_max_val: float = 83861.2     # 99.999th percentile of raw charge [ADC]; anchors y = +1

    # ============ Backbone ============
    backbone_name: str = "attn_mae"      # key into BACKBONE_REGISTRY
    encoding_range: float = 125.0        # sinusoidal positional encoding range
    encoding_dim: int = 32               # sinusoidal encoding channels per spatial axis
    feature_dim: int = 64                # backbone output channels; must match model's output
    use_proj_head: bool = True           # attach MLP projection head between backbone and loss
    proj_head_hidden_dim: int = 256      # inner MLP width
    proj_head_output_dim: int = 128      # output dimension of the final FC layer
    proj_head_n_layers: int = 2          # number of MLP layers before the final FC

    # ============ Augmentation ============
    use_cropping: bool = True             # enable activity-aware multi-crop augmentation
    use_masking: bool = True              # enable masking on student views
    # "region" (whole cells of a fixed grid), "block" (windows around active voxels) or
    # "pixel" (independent per-voxel dropout). Only "region" enumerates its own
    # occupancy candidates, so it is the default.
    mask_type: str = "region"
    mask_ratio: float = 0.5             # fraction of active voxels to mask ("pixel"/"block")
    mask_block_win_ch: int = 5          # half-window radius in channel direction ("block" only)
    mask_block_win_tick: int = 5        # half-window radius in tick direction ("block" only)
    # "region" only. The cell must tile the canvas evenly and divide by the candidate
    # stride; 70 x 100 gives an exact 15 x 15 grid on the 1050 x 1500 canvas.
    mask_region_cell_w: int = 70         # cell width (wire channels)
    mask_region_cell_h: int = 100        # cell height (time ticks)
    mask_region_flavor: str = "wipe"     # "wipe" (empty the cell) or "randomize" (thin it)
    mask_region_wipe_max: float = 0.75   # "wipe": ceiling on the fraction of voxels removed
    mask_region_r1: float = 0.5          # "randomize": fraction of active cells selected
    mask_region_r2: float = 0.75         # "randomize": voxel dropout rate inside them
    crop_n_global: int = 2               # number of global crops per image
    crop_n_local: int = 4                # number of local crops per image
    crop_global_scale: tuple = (0.4, 1.0)    # global crop area as fraction of image area
    crop_local_scale: tuple = (0.05, 0.2)    # local crop area as fraction of image area
    crop_aspect_ratio: tuple = (0.75, 1.333) # width-to-height aspect ratio range
    crop_blur_sigma_px: float = 10.0     # Gaussian blur sigma for activity heatmap (px)
    crop_heatmap_power: float = 1.0      # exponent applied to heatmap before sampling
    crop_min_active_pixels: int = 10     # minimum active voxels required inside a crop

    # ============ Training ============
    epochs: int = 10
    momentum_start: float = 0.996        # EMA teacher momentum schedule start
    momentum_end: float = 0.9999         # EMA teacher momentum schedule end
    lr: float = 1e-4                     # base learning rate
    min_lr: float = 1e-6                 # minimum learning rate (cosine annealing floor)
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4        # weight decay at end of training (cosine annealed)
    warmup_epochs: int = 1               # linear LR warmup duration

    # ============ Objective ============
    # Which training objective runs. The masker removes active pixels in every mode; what
    # differs is whether the backbone puts a learnable placeholder back at those coordinates
    # ("injection") and what the loss is asked to predict there.
    #   "dino"   cross-entropy on the surviving pixels only; no injection.
    #   "hybrid" cross-entropy on surviving AND re-injected pixels.
    #   "mae"    no teacher at all; the loss is charge + occupancy reconstruction.
    objective: str = "hybrid"
    lambda_charge: float = 0.1           # mae: weight on the charge term (count-normalised first)
    lambda_occ: float = 1.0              # mae: weight on the occupancy term
    occ_grow_iters: int = 1              # mae: 3x3 dilations when growing occupancy candidates
    # mae with enumerated candidates: per-image caps on the empty candidates, which are
    # what drive memory. 0 means the cap is unset. See dino.masking.cap_negatives.
    occ_max_neg: int = 0                 # mae: absolute cap on empties per image
    occ_neg_per_pos: float = 0.0         # mae: cap on empties per positive

    # ============ Loss ============
    use_centering: bool = True           # subtract running center from teacher before loss
    center_momentum: float = 0.996       # EMA decay for teacher centering
    teacher_temp: float = 0.04           # teacher softmax temperature
    student_temp: float = 0.1            # student softmax temperature
    normalize_features: bool = False     # L2-normalise features before loss; False when use_proj_head=True
    use_cov_penalty: bool = True         # VICReg covariance decorrelation penalty
    cov_penalty_weight: float = 10.0
    use_var_penalty: bool = False        # VICReg variance penalty (hinge on per-dim std)
    var_penalty_weight: float = 1.0
    var_gamma: float = 1.0               # target minimum std per feature dimension

    # ============ Checkpointing & debug ============
    output_dir: str = "./dino_checkpoints"
    save_every: int = 10
    debug: bool = False                  # enable detailed logging and history tracking
    debug_every: int = 100               # log scalars / stats / grad norms every N batches
    debug_dir: str = "./dino_debug"


OBJECTIVES = ("dino", "hybrid", "mae")

# Keys that only mean something under objective="mae". 
# Setting one anywhere else throws an error.
MAE_ONLY_KEYS = ("lambda_charge", "lambda_occ", "occ_grow_iters",
                 "occ_max_neg", "occ_neg_per_pos")


def objective_injects(objective: str) -> bool:
    """Does this objective put placeholders back at the coordinates the masker removed?

    "dino" supervises only the pixels that survived masking, so nothing is injected and the
    removed coordinates stay absent from the student's output. 
    "hybrid" and "mae" both need something to predict at the removed coordinates, so both inject.
    """
    return objective in ("hybrid", "mae")


MASK_TYPES = ("region", "block", "pixel")


def occ_read_stride(backbone_name: str):
    """The decoder resolution this backbone's occupancy head reads at, or None if it has
    no such head.

    Enumerated occupancy candidates are reported, injected and scored in these units, so
    the value is a property of where the head sits in the architecture -- not a tuning
    knob, and not something a config may disagree with. Read from the backbone class so
    exactly one place decides it.
    """
    return getattr(BACKBONE_REGISTRY[backbone_name], "OCC_READ_STRIDE", None)


def _backbone_can_inject(backbone_name: str) -> bool:
    """Does this backbone accept `masked_coords`, i.e. can it reinject placeholders?

    `attn_default` and `base` drop the masked voxels and never restore them, 
    so they can only run the dino objective.
    """
    backbone_cls = BACKBONE_REGISTRY[backbone_name]
    return "masked_coords" in inspect.signature(backbone_cls.forward).parameters


def validate_config(cfg) -> None:
    """Reject conflicting configurations."""

    defaults = {f.name: f.default for f in fields(DINOConfig)}

    if cfg.objective not in OBJECTIVES:
        raise ValueError(
            f"objective={cfg.objective!r} is not one of {', '.join(OBJECTIVES)}"
        )

    if objective_injects(cfg.objective):
        if not cfg.use_masking:
            raise ValueError(
                f"objective={cfg.objective!r} needs use_masking=true — with nothing masked "
                f"there is nothing to reinject, which is objective='dino'"
            )
        if not _backbone_can_inject(cfg.backbone_name):
            raise ValueError(
                f"backbone {cfg.backbone_name!r} cannot reinject masked coordinates, so it "
                f"cannot run objective={cfg.objective!r} — use 'attn_mae', or set "
                f"objective='dino'"
            )

    if cfg.mask_type not in MASK_TYPES:
        raise ValueError(
            f"mask_type={cfg.mask_type!r} is not one of {', '.join(MASK_TYPES)}"
        )

    if cfg.mask_type == "region":
        # Caught here rather than in the masker so a submitted job fails before it builds
        # a model; the arithmetic is the same either way.
        stride = occ_read_stride(cfg.backbone_name)
        for name, cell, canvas in (("mask_region_cell_w", cfg.mask_region_cell_w, cfg.image_w),
                                   ("mask_region_cell_h", cfg.mask_region_cell_h, cfg.image_h)):
            if canvas % cell:
                raise ValueError(
                    f"{name}={cell} does not tile a canvas of {canvas} evenly — partial "
                    f"edge cells densify to a different candidate count than interior "
                    f"ones, which skews the occupancy positive rate"
                )
            # Only meaningful on a backbone that scores occupancy at all; the others
            # never enumerate candidates, so there is no footprint to misalign.
            if stride is not None and cell % stride:
                raise ValueError(
                    f"{name}={cell} must divide by the occupancy read stride {stride} of "
                    f"backbone {cfg.backbone_name!r}, or a wiped cell's footprint at that "
                    f"scale reaches into a neighbouring cell that still holds charge"
                )
        if cfg.mask_region_flavor not in ("wipe", "randomize"):
            raise ValueError(
                f"mask_region_flavor={cfg.mask_region_flavor!r} is not 'wipe' or 'randomize'"
            )
        # region masking removes whole cells, so how much it removes is set by wipe_max
        # (or r1/r2), never by mask_ratio. A config carrying both reads as though the
        # ratio applied, and it does not.
        if cfg.mask_ratio != defaults["mask_ratio"]:
            knob = ("mask_region_wipe_max" if cfg.mask_region_flavor == "wipe"
                    else "mask_region_r1 / mask_region_r2")
            raise ValueError(
                f"mask_ratio does not apply to mask_type='region' — the masked fraction "
                f"is set by {knob}; remove mask_ratio from the config"
            )

    if cfg.objective == "mae":
        if cfg.mask_type == "region":
            if cfg.mask_region_flavor != "wipe":
                raise ValueError(
                    "objective='mae' with mask_type='region' needs "
                    "mask_region_flavor='wipe': 'randomize' leaves active voxels inside a "
                    "selected cell that the occupancy label calls empty, so the target is "
                    "wrong wherever it matters most"
                )
            if cfg.occ_grow_iters != defaults["occ_grow_iters"]:
                raise ValueError(
                    "occ_grow_iters only applies when the occupancy candidates are grown "
                    "from surviving structure; mask_type='region' enumerates them instead "
                    "— remove it from the config"
                )
            if not (cfg.occ_max_neg or cfg.occ_neg_per_pos):
                raise ValueError(
                    "objective='mae' with mask_type='region' needs occ_max_neg or "
                    "occ_neg_per_pos set — densifying whole cells enumerates every pixel "
                    "of each one, and uncapped that is far more empty candidates than the "
                    "decoder can hold. Measure the counts for your cell size with "
                    "gridutils/diagnostics/measure_occ_candidates.sub, then set the cap"
                )
        elif cfg.occ_max_neg or cfg.occ_neg_per_pos:
            raise ValueError(
                f"occ_max_neg / occ_neg_per_pos cap the enumerated occupancy candidates, "
                f"which only mask_type='region' produces (this run is "
                f"{cfg.mask_type!r}) — remove them from the config"
            )
        if cfg.use_cropping:
            raise ValueError(
                "objective='mae' reconstructs the masked region of the full image; set "
                "use_cropping=false"
            )
        if cfg.use_proj_head:
            raise ValueError(
                "objective='mae' has no DINO term to project for; set use_proj_head=false"
            )
    else:
        # A weight that looks applied but is inert is worse than a rejected config.
        for key in MAE_ONLY_KEYS:
            if getattr(cfg, key) != defaults[key]:
                raise ValueError(
                    f"{key} only applies to objective='mae' (this run is "
                    f"{cfg.objective!r}) — remove it from the config"
                )

    if cfg.objective == "dino" and not cfg.use_masking and not cfg.use_cropping:
        raise ValueError(
            "objective='dino' with neither masking nor cropping compares each view against "
            "itself and learns nothing — enable at least one"
        )
