"""Smoke-test config — verifies env + scene layout train without errors.

Inherits from ``arguments/vrugz/basketball.py`` (the LocalDyGS preset
closest to a multi-view sync rig captured outdoors), then overrides
the iteration budget down to 500 so the job finishes in minutes.

Use for first-time validation on a new scene. For real training, point
``--configs`` at the upstream basketball.py directly.
"""

_base_ = "/home/yubo/github/LocalDyGS/arguments/vrugz/basketball.py"


OptimizationParams = dict(
    iterations = 500,

    # Cap every LR scheduler at 500 too, otherwise the LR is barely
    # off its initial value when training stops — fine for a smoke
    # test, but means the loss curve isn't representative.
    position_lr_max_steps = 500,
    offset_lr_max_steps = 500,
    mlp_opacity_lr_max_steps = 500,
    mlp_cov_lr_max_steps = 500,
    mlp_offset_lr_max_steps = 500,
    mlp_color_lr_max_steps = 500,
    mlp_featurebank_lr_max_steps = 500,
    appearance_lr_max_steps = 500,
    update_until = 500,
)
