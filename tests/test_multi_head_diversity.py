
import pytest
import torch
import torch.nn as nn
from src.models.architectures.multi_head_diversity import MultiHeadDiversityModel
from src.models.losses.diversity_loss import DiversityBCELoss
from src.models.architectures import create_model

@pytest.fixture
def mock_vocab_sizes():
    return {
        "f1": 100,
        "f2": 50
    }

@pytest.fixture
def mock_feature_names():
    return ["f1", "f2"]

@pytest.fixture
def mock_gated_dcn_config():
    return {
        "use_dcn": True,
        "dcn_num_layers": 1,
        "dcn_use_layernorm": True,
        "dcn_low_rank": 8,
        "use_senet": False,
        "senet_squeeze_funcs": ["mean"],
        "senet_reduction_ratio": 3,
        "senet_hidden_activation": "relu",
        "senet_excitation_activation": "sigmoid",
        "senet_num_groups": 1,
        "senet_reweight_mode": "feature",
        "senet_use_fuse": False,
        "senet_use_layer_norm": False,
        "use_feature_gating": False,
        "feature_gating_activation": "sigmoid",
        "feature_gating_low_rank": None,
        "mlp_hidden_dims": [16], # Original backbone MLP (should be ignored)
        "mlp_activation": "relu",
        "mlp_use_skip_connections": False,
        "mlp_dropout": 0.0,
        "use_layer_norm": True,
    }

@pytest.fixture
def multi_head_config(mock_gated_dcn_config):
    head_config_1 = {
        "hidden_dims": [16, 8],
        "activation": "relu",
        "dropout": 0.0,
        "use_layer_norm": True,
        "use_skip_connections": True
    }
    head_config_2 = {
        "hidden_dims": [12],
        "activation": "tanh",
        "dropout": 0.1,
        "use_layer_norm": False,
        "use_skip_connections": False
    }
    
    return {
        "embedding_dim": 8,
        "device": "cpu",
        "model": {
            "backbone_type": "gated_dcn",
            "backbone_config": mock_gated_dcn_config,
            "heads": [head_config_1, head_config_2],
            "diversity_weight": 0.1
        },
        "feature_embeddings": {
             "f1": {"type": "standard", "dim": 8},
             "f2": {"type": "standard", "dim": 8}
        }
    }

def test_multi_head_diversity_initialization(multi_head_config, mock_vocab_sizes, mock_feature_names):
    model = create_model(multi_head_config, mock_vocab_sizes, mock_feature_names)
    assert isinstance(model, MultiHeadDiversityModel)
    assert len(model.heads) == 2
    assert isinstance(model.backbone.mlp, nn.Identity)

def test_multi_head_diversity_forward(multi_head_config, mock_vocab_sizes, mock_feature_names):
    model = MultiHeadDiversityModel(mock_vocab_sizes, mock_feature_names, multi_head_config)
    
    batch_size = 4
    x = torch.zeros((batch_size, 2), dtype=torch.long)
    
    output = model(x)
    
    assert "logits" in output
    assert "aux_logits" in output
    
    # Check shapes
    # Logits: [B, 1]
    assert output["logits"].shape == (batch_size, 1)
    # Aux logits: [K, B, 1]
    assert output["aux_logits"].shape == (2, batch_size, 1)

def test_diversity_loss():
    loss_fn = DiversityBCELoss(diversity_weight=0.1)
    
    # 2 Heads, Batch 4
    aux_logits = torch.randn(2, 4, 1)
    y_true = torch.randint(0, 2, (4, 1)).float()
    
    loss = loss_fn(aux_logits, y_true)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar

def test_model_training_step(multi_head_config, mock_vocab_sizes, mock_feature_names):
    model = MultiHeadDiversityModel(mock_vocab_sizes, mock_feature_names, multi_head_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.randint(0, 50, (4, 2))
    y = torch.randint(0, 2, (4, 1)).float()
    
    optimizer.zero_grad()
    output = model(x)
    loss = model.compute_loss(output, y)
    loss.backward()
    optimizer.step()
    
    # Check if weights updated (simple check)
    head_param = next(model.heads[0].parameters())
    assert head_param.grad is not None

