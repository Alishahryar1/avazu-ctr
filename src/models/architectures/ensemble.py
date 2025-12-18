"""Ensemble model for CTR prediction."""
import torch
import torch.nn as nn

from src.config.config import ConfigType, ModelConfig
from src.models.types import ModelOutput
from src.models.architectures.base import BaseCTRModel
from src.models.losses import KBCELoss


def _create_model_from_config(
    model_config: ModelConfig,
    vocab_sizes: dict[str, int],
    feature_names: list[str],
    parent_config: ConfigType,
    seed: int
) -> BaseCTRModel:
    """
    Factory function to create a model from a ModelConfig.
    
    Args:
        model_config: The model-specific config (GatedDCNConfig, STECConfig, or EnsembleConfig)
        vocab_sizes: Feature vocabulary sizes
        feature_names: List of feature names
        parent_config: The parent ConfigType for global settings (embedding_dim, etc.)
        seed: Random seed for this model's initialization
        
    Returns:
        Instantiated model
    """
    # Import here to avoid circular imports
    from src.models.architectures.gated_dcn import GatedDCNModel
    from src.models.architectures.stec import STECModel
    
    # Set seed for reproducible initialization
    torch.manual_seed(seed)
    
    # Create a full config by combining parent config with model-specific config
    full_config: ConfigType = {**parent_config, 'model': model_config}  # type: ignore
    
    # Detect model type by checking for unique keys
    if 'models' in model_config:
        # EnsembleConfig - recursive!
        return EnsembleModel(vocab_sizes, feature_names, full_config, base_seed=seed)
    elif 'stec_num_layers' in model_config:
        # STECConfig
        return STECModel(vocab_sizes, feature_names, full_config)
    elif 'use_dcn' in model_config:
        # GatedDCNConfig
        return GatedDCNModel(vocab_sizes, feature_names, full_config)
    else:
        raise ValueError(f"Unknown model config type. Keys: {model_config.keys()}")


class EnsembleModel(BaseCTRModel):
    """
    Ensemble of heterogeneous models for CTR prediction.

    Supports any combination of GatedDCNModel, STECModel, or nested EnsembleModel.
    Each model in the ensemble can have different architectures.
    During inference, predictions are aggregated across all models.

    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with `model.models` list of model configs.
        base_seed: Base seed for reproducibility. Model i uses seed = base_seed + i.
    """
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        feature_names: list[str],
        config: ConfigType,
        base_seed: int | None = None
    ):
        super().__init__()
        model_config = config['model']
        
        # Get ensemble settings
        self.base_seed: int = base_seed if base_seed is not None else config['seed']
        self.ensemble_aggregation: str = model_config['ensemble_aggregation']
        
        # Get list of model configs
        model_configs: list[ModelConfig] = model_config['models']
        self.k: int = len(model_configs)
        
        if self.k == 0:
            raise ValueError("Ensemble must have at least one model in 'models' list")

        # Create models from configs with different seeds
        self.models = nn.ModuleList()
        for i, sub_model_config in enumerate(model_configs):
            model = _create_model_from_config(
                model_config=sub_model_config,
                vocab_sizes=vocab_sizes,
                feature_names=feature_names,
                parent_config=config,
                seed=self.base_seed + i
            )
            self.models.append(model)

        # Reset to base seed after initialization
        torch.manual_seed(self.base_seed)
        
        # Internal loss for multi-branch architecture
        self._kbce_loss = KBCELoss()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass through all models in the ensemble.

        Args:
            x: Input tensor of shape [Batch, Num_Features]

        Returns:
            ModelOutput with aggregated logits and list of branch logits
        """
        all_logits = []
        for model in self.models:
            output = model(x)
            all_logits.append(output["logits"])

        # Stack all logits: [K, Batch, 1]
        stacked_logits = torch.stack(all_logits, dim=0)

        # Aggregate predictions
        if self.ensemble_aggregation == 'mean':
            aggregated = stacked_logits.mean(dim=0)
        elif self.ensemble_aggregation == 'median':
            aggregated = stacked_logits.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {self.ensemble_aggregation}")

        return {
            "logits": aggregated,
            "aux_logits": all_logits  # List of k branch logits
        }

    def compute_loss(
        self, 
        output: ModelOutput, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute K-BCE loss for ensemble architecture."""
        return self._kbce_loss(output["logits"], output["aux_logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        """Return model name for registry."""
        return "ensemble"

    def forward_single(self, x: torch.Tensor, model_idx: int) -> ModelOutput:
        """
        Forward pass through a single model in the ensemble.

        Args:
            x: Input tensor of shape [Batch, Num_Features]
            model_idx: Index of the model to use (0 to k-1)

        Returns:
            ModelOutput from the specified model
        """
        if model_idx < 0 or model_idx >= self.k:
            raise ValueError(f"model_idx must be in range [0, {self.k-1}], got {model_idx}")
        return self.models[model_idx](x)

    def get_model(self, idx: int) -> BaseCTRModel:
        """Get a specific model from the ensemble."""
        return self.models[idx]  # type: ignore

    def num_models(self) -> int:
        """Return the number of models in the ensemble."""
        return self.k

