from typing import TypedDict, Literal

from .scheduler_config import SchedulerConfig


class AdamWConfig(TypedDict, total=False):
    """AdamW optimizer config with lr, weight decay, and scheduler settings."""

    type: Literal["adamw"]
    lr: float
    weight_decay: float
    scheduler: SchedulerConfig
