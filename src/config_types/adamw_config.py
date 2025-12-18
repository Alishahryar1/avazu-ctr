from typing import TypedDict, Literal


class AdamWConfig(TypedDict, total=False):
    """AdamW optimizer config with lr, warmup, and weight decay."""
    type: Literal["adamw"]
    lr: float
    warmup_epoch_ratio: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
