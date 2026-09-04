"""
test_objective_validation.py
────────────────────────────
Does a configuration that cannot mean what it says get rejected before the run starts?

`objective` decides two things at once: what the loss is, and whether the coordinates the
masker removed are put back as placeholders for the network to predict at. Several
combinations are contradictory -- asking for a mode that reinjects while masking is off, or
asking an injection-incapable backbone to reinject -- and several are merely inert, like
setting an mae-only weight on a dino run. Inert is the dangerous one: the config records a
weight, the log echoes it, and nothing applies it.

So all of them raise, at submit time, before the model or dataset is built. This checks the
rejections fire, that each message names the fix, and -- equally important -- that the
combinations we actually run are still accepted.

No GPU, no data, no training: `validate_config` reads a config object and nothing else.

Run:  python -u tests/test_objective_validation.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dino.config import (DINOConfig, OBJECTIVES, objective_injects,
                         validate_config, _backbone_can_inject)


def cfg(**kw) -> DINOConfig:
    """A config built from the SHIPPED defaults; keyword arguments break one thing at a time.

    Deliberately does not pin backbone_name or objective: an earlier version of this file
    did, and so passed while the shipped defaults themselves were an invalid combination
    (a backbone that cannot reinject, under an objective that requires it).
    """
    return DINOConfig(**kw)


def expect_raises(name: str, expect_in_message: str, **kw) -> None:
    try:
        validate_config(cfg(**kw))
    except ValueError as exc:
        assert expect_in_message in str(exc), (
            f"{name}: raised, but the message does not mention {expect_in_message!r}\n"
            f"  got: {exc}"
        )
        return
    raise AssertionError(f"{name}: expected a ValueError, none raised")


# ── the combinations we actually run must stay accepted ──────────────────────

def check_shipped_defaults_are_valid() -> None:
    """The bare defaults must be a runnable configuration.

    This is the check the first version of this file was missing: `objective` defaults to
    hybrid, which requires a backbone that can reinject, so the default backbone has to be
    one that can. It is `attn_mae`.
    """
    c = DINOConfig()
    assert c.backbone_name == "attn_mae", c.backbone_name
    assert c.objective == "hybrid", c.objective
    validate_config(c)


def check_dino_on_either_augmentation() -> None:
    """dino takes cropping and/or masking; masking is an augmentation there, not a task."""
    validate_config(cfg(objective="dino", use_masking=True, use_cropping=False))
    validate_config(cfg(objective="dino", use_masking=False, use_cropping=True))
    validate_config(cfg(objective="dino", use_masking=True, use_cropping=True))


def check_dino_runs_on_a_non_injecting_backbone() -> None:
    """A backbone with no placeholder mechanism can still run dino, and only dino."""
    assert not _backbone_can_inject("attn_default")
    validate_config(cfg(objective="dino", backbone_name="attn_default"))


def mae_cfg(**kw):
    """A valid mae config: no cropping, no head, and the candidate cap the default
    mask_type='region' requires."""
    base = dict(objective="mae", use_cropping=False, use_proj_head=False, occ_max_neg=20000)
    base.update(kw)
    return cfg(**base)


def check_mae_shape_is_accepted() -> None:
    validate_config(mae_cfg())


def check_mae_weights_allowed_under_mae() -> None:
    validate_config(mae_cfg(lambda_charge=0.5, lambda_occ=2.0))


def check_mae_on_grown_candidates_is_accepted() -> None:
    """A masker that cannot enumerate candidates grows them, and takes occ_grow_iters."""
    validate_config(mae_cfg(mask_type="block", occ_max_neg=0, occ_grow_iters=3))


def check_region_grid_must_tile_the_canvas() -> None:
    expect_raises("indivisible cell", "evenly", mask_region_cell_w=64)


def check_region_cell_must_divide_the_stride() -> None:
    expect_raises("odd cell", "stride", mask_region_cell_w=75, image_w=1050)


def check_region_rejects_unknown_flavor() -> None:
    expect_raises("bad flavor", "'wipe' or 'randomize'", mask_region_flavor="soften")


def check_unknown_mask_type() -> None:
    expect_raises("bad mask_type", "not one of", mask_type="chunk")


def check_mask_ratio_is_inert_under_region() -> None:
    """The masked fraction comes from wipe_max here; a ratio would look applied and not be."""
    expect_raises("mask_ratio under region", "mask_region_wipe_max", mask_ratio=0.9)


def check_mae_region_needs_a_negative_cap() -> None:
    expect_raises("uncapped region candidates", "occ_max_neg",
                  objective="mae", use_cropping=False, use_proj_head=False)


def check_mae_region_rejects_randomize() -> None:
    expect_raises("randomize as a recon target", "wipe",
                  objective="mae", use_cropping=False, use_proj_head=False,
                  occ_max_neg=20000, mask_region_flavor="randomize")


def check_grow_iters_is_inert_under_region() -> None:
    expect_raises("occ_grow_iters under region", "enumerates them instead",
                  objective="mae", use_cropping=False, use_proj_head=False,
                  occ_max_neg=20000, occ_grow_iters=3)


def check_caps_rejected_without_enumerated_candidates() -> None:
    expect_raises("cap on a grown-candidate run", "only mask_type='region' produces",
                  objective="mae", use_cropping=False, use_proj_head=False,
                  mask_type="block", occ_max_neg=20000)


# ── contradictions must be rejected ──────────────────────────────────────────

def check_unknown_objective() -> None:
    expect_raises("unknown objective", "not one of", objective="MAE")


def check_injecting_without_masking() -> None:
    for mode in ("hybrid", "mae"):
        expect_raises(f"{mode} without masking", "use_masking=true",
                      objective=mode, use_masking=False,
                      use_cropping=(mode != "mae"),
                      use_proj_head=(mode != "mae"))


def check_injecting_on_incapable_backbone() -> None:
    expect_raises("hybrid on attn_default", "cannot reinject",
                  objective="hybrid", backbone_name="attn_default")


def check_mae_rejects_cropping_and_head() -> None:
    # cap set, so the failure under test is the one being named
    expect_raises("mae with cropping", "use_cropping=false",
                  objective="mae", use_cropping=True, use_proj_head=False,
                  occ_max_neg=20000)
    expect_raises("mae with proj head", "use_proj_head=false",
                  objective="mae", use_cropping=False, use_proj_head=True,
                  occ_max_neg=20000)


def check_mae_weights_rejected_elsewhere() -> None:
    """An inert weight must be an error, not a silently ignored number in the log."""
    for key, value in (("lambda_charge", 0.5), ("lambda_occ", 2.0), ("occ_grow_iters", 3)):
        for mode in ("dino", "hybrid"):
            expect_raises(f"{key} under {mode}", "only applies to objective='mae'",
                          objective=mode, **{key: value})


def check_dino_with_no_augmentation_at_all() -> None:
    expect_raises("dino with nothing on", "learns nothing",
                  objective="dino", use_masking=False, use_cropping=False)


# ── the helper the rest of the code branches on ──────────────────────────────

def check_injects_mapping() -> None:
    assert objective_injects("hybrid") and objective_injects("mae")
    assert not objective_injects("dino")
    assert set(OBJECTIVES) == {"dino", "hybrid", "mae"}


CHECKS = (
    check_mae_on_grown_candidates_is_accepted,
    check_unknown_mask_type,
    check_region_grid_must_tile_the_canvas,
    check_region_cell_must_divide_the_stride,
    check_region_rejects_unknown_flavor,
    check_mask_ratio_is_inert_under_region,
    check_mae_region_needs_a_negative_cap,
    check_mae_region_rejects_randomize,
    check_grow_iters_is_inert_under_region,
    check_caps_rejected_without_enumerated_candidates,
    check_shipped_defaults_are_valid,
    check_dino_on_either_augmentation,
    check_dino_runs_on_a_non_injecting_backbone,
    check_mae_shape_is_accepted,
    check_mae_weights_allowed_under_mae,
    check_unknown_objective,
    check_injecting_without_masking,
    check_injecting_on_incapable_backbone,
    check_mae_rejects_cropping_and_head,
    check_mae_weights_rejected_elsewhere,
    check_dino_with_no_augmentation_at_all,
    check_injects_mapping,
)


def main() -> int:
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {check.__name__}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
