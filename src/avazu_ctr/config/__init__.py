"""Validated experiment configuration."""

from avazu_ctr.config.loader import load_experiment
from avazu_ctr.config.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "load_experiment"]
