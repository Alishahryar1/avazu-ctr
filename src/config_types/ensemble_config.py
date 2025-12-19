from typing import TypedDict


class EnsembleConfig(TypedDict):
    models: list  # List of ModelConfig (GatedDCNConfig | STECConfig | EnsembleConfig)
    ensemble_aggregation: str  # Aggregation method: 'mean' or 'median'
