from typing import TypedDict, Literal

from .scheduler_config import SchedulerConfig


class AdagradConfig(TypedDict, total=False):
    """Adagrad optimizer config with lr, weight decay, and scheduler settings."""

    type: Literal["adagrad"]
    lr: float
    weight_decay: float
    scheduler: SchedulerConfig
