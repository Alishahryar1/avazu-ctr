import torch
import torch.nn as nn
from typing import List, Dict

from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.config_types import ConfigType, MultiHeadDiversityConfig, ResidualMLPConfig
from src.models.losses.diversity_loss import DiversityBCELoss
from src.models.layers.mlp import ResidualMLP

class MultiHeadDiversityModel(BaseCTRModel):
    """
    Single model that mimics an ensemble using a shared backbone and 
    multiple diverse heads.
    """
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_names: List[str],
        config: ConfigType
    ):
        super().__init__()
        
        # Extract specific config
        # The create_model factory passes the full config, so we look inside 'model'
        # We assume the factory logic for MultiHeadDiversityModel passes validation 
        # that 'model' is MultiHeadDiversityConfig.
        model_config: MultiHeadDiversityConfig = config['model'] # type: ignore
        
        # 1. Initialize the Shared Backbone
        # We create a temporary config for the backbone to use existing factory or class
        backbone_type = model_config['backbone_type']
        
        # Construct a config for the backbone. 
        # We need to ensure the backbone model class can be instantiated.
        # We'll use the specific model class directly to avoid recursion issues if we used generic create_model
        # But for 'gated_dcn', we can likely use a targeted import.
        # Ideally, we should use a registry or the create_model function with a modified config.
        # To reuse create_model, we need to fake the config.
        
        backbone_full_config = config.copy()
        # Inject backbone config into 'model' key for the factory
        backbone_full_config['model'] = model_config['backbone_config']
        
        # We need to instantiate the backbone.
        # We can use the create_model from . (lazy import to avoid circular dependency if possible, 
        # but better to import specific classes if known).
        # Since 'gated_dcn' is the primary candidate:
        from src.models.architectures.gated_dcn import GatedDCNModel
        
        if backbone_type == 'gated_dcn':
            self.backbone = GatedDCNModel(vocab_sizes, feature_names, backbone_full_config)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # 2. Neutralize Backbone's Final Output Layer
        # & Determine Input Dimension for Heads
        input_dim = 0
        
        # For GatedDCN, the structure is:
        # embeddings -> projection -> senet/gating -> dcn -> mlp -> output
        # We want: embeddings ... -> dcn -> [SPLIT] -> Our Heads
        # So we should remove the backbone's MLP entirely.
        
        if hasattr(self.backbone, 'mlp') and isinstance(self.backbone.mlp, nn.Module):
            # Check if it is a ResidualMLP
            if hasattr(self.backbone.mlp, 'layers') and len(self.backbone.mlp.layers) > 0:
                # The input to the first layer of the existing MLP is what we want.
                first_layer = self.backbone.mlp.layers[0]
                if isinstance(first_layer, nn.Linear):
                    input_dim = first_layer.in_features
                else:
                    # Try to infer? Or just run a dummy forward?
                    # Safer to require standard structure
                    pass
            
            # If we couldn't find input_dim from layers (e.g. empty MLP), check attributes
            if input_dim == 0:
                # Fallback logic based on GatedDCN internals
                # It calculates working_dim in __init__
                if hasattr(self.backbone, 'dcn') and hasattr(self.backbone.dcn, 'input_dim'):
                    input_dim = self.backbone.dcn.input_dim # type: ignore
                # This might be unreliable if DCN is not used.
                # Let's rely on valid inspection of the neutralized MLP or a dummy pass if needed.
                # But we are in __init__, so dummy pass is expensive/complex.
                 
            # Recover input_dim from the first layer of the MLP we are about to delete
            if input_dim == 0 and len(self.backbone.mlp.layers) > 0:
                input_dim = self.backbone.mlp.layers[0].in_features
            
            # Neutralize
            self.backbone.mlp = nn.Identity()
            
        else:
            raise AttributeError("Backbone must have an 'mlp' attribute to be replaced.")

        if input_dim == 0:
            raise ValueError("Could not determine input dimension from backbone.")

        self.input_dim = input_dim

        # 3. Create Multiple Independent Heads
        heads_config = model_config['heads']
        self.num_heads = len(heads_config)
        self.heads = nn.ModuleList()
        
        for head_cfg in heads_config:
            self.heads.append(ResidualMLP(
                input_dim=self.input_dim,
                hidden_dims=head_cfg['hidden_dims'],
                output_dim=1,
                activation=head_cfg['activation'],
                dropout=head_cfg['dropout'],
                use_layer_norm=head_cfg['use_layer_norm'],
                use_skip_connections=head_cfg['use_skip_connections']
            ))
        
        # 4. Diversity Loss
        self.loss_fn = DiversityBCELoss(diversity_weight=model_config['diversity_weight'])

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # 1. Get Shared Representation
        # The backbone's mlp is Identity, so backbone(x) returns the features
        # GatedDCN returns {"logits": ..., "aux_logits": ...}
        # "logits" will contain the interaction features [B, Dim]
        backbone_out = self.backbone(x)
        shared_features = backbone_out["logits"]
        
        # 2. Pass through all heads
        head_logits = []
        for head in self.heads:
            # head returns [B, 1]
            head_logits.append(head(shared_features))
            
        # Stack: [K, Batch, 1]
        stacked_logits = torch.stack(head_logits, dim=0)
        
        # 3. Aggregate (Mean) for final prediction
        # [Batch, 1]
        avg_logits = stacked_logits.mean(dim=0)
        
        return {
            "logits": avg_logits,
            "aux_logits": stacked_logits, # Pass stacked logits to loss
        }

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        # Use the custom diversity loss on the stacked logits
        return self.loss_fn(output["aux_logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        return "multi_head_diversity"
