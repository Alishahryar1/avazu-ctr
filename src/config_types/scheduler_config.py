from typing import TypedDict, Literal


class SchedulerConfig(TypedDict, total=False):
    """LR scheduler configuration for warmup and decay.

    Attributes:
        warmup_epoch_ratio: Fraction of first epoch for linear warmup (0.0-1.0).
        min_lr: Minimum learning rate at end of decay (default: 1e-6).
        decay_type: Type of decay after warmup - 'none', 'cosine', or 'linear'.
    """

    warmup_epoch_ratio: float
    min_lr: float
    decay_type: Literal["none", "cosine", "linear"]
