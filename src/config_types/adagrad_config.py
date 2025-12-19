from typing import TypedDict, Literal


class AdagradConfig(TypedDict, total=False):
    """Adagrad optimizer config with lr, warmup, and weight decay."""

    type: Literal["adagrad"]
    lr: float
    warmup_epoch_ratio: float
    weight_decay: float
