"""Evaluation training and fixed-budget production refitting."""

from avazu_ctr.training.candidate import CandidateResult, CandidateTrainer
from avazu_ctr.training.refit import ProductionRefitter, RefitResult

__all__ = [
    "CandidateResult",
    "CandidateTrainer",
    "ProductionRefitter",
    "RefitResult",
]
