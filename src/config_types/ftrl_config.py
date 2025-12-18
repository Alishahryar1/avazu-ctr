from typing import TypedDict, Literal


class FTRLConfig(TypedDict, total=False):
    """FTRL optimizer config (no lr/warmup/weight_decay - uses own params)."""
    type: Literal["ftrl"]
    alpha: float  # Learning rate proportionality constant
    beta: float   # Learning rate smoothing parameter
    l1: float     # L1 regularization (enables sparsity)
    l2: float     # L2 regularization
