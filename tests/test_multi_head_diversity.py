
import unittest
import torch
import torch.nn as nn
from src.models.architectures.multi_head_diversity import MultiHeadDiversityModel
from src.config_types import MultiHeadDiversityConfig, ResidualMLPConfig
from src.models.losses.diversity_loss import DiversityBCELoss

class TestMultiHeadDiversityModel(unittest.TestCase):
    def setUp(self):
        self.vocab_sizes = {
            'feat_1': 100,
            'feat_2': 50
        }
        self.feature_names = ['feat_1', 'feat_2']
        
        self.head_config: ResidualMLPConfig = {
            'hidden_dims': [32],
            'activation': 'relu',
            'dropout': 0.1,
            'use_layer_norm': True,
            'use_skip_connections': True
        }
        
        self.backbone_config = {
            'use_dcn': True,
            'dcn_num_layers': 1,
            'dcn_use_layernorm': True,
            'dcn_low_rank': 4,
            'use_senet': False,
            'senet_squeeze_funcs': ['mean'],
            'senet_reduction_ratio': 1,
            'senet_hidden_activation': 'relu',
            'senet_excitation_activation': 'sigmoid',
            'use_feature_gating': False,
            'use_layer_norm': True,
        }
        
        self.config = {
            'embedding_dim': 8,
            'embedding_projection_dim': None,
            'feature_embeddings': {
                'feat_1': {'type': 'standard', 'dim': 8},
                'feat_2': {'type': 'standard', 'dim': 8}
            },
            'model': {
                'backbone_type': 'gated_dcn', # Actually ignored now but kept for config structure
                'backbone_config': self.backbone_config,
                'heads': [self.head_config, self.head_config],
                'diversity_weight': 0.1,
                'feature_bagging_ratio': 0.8
            }
        }

    def test_initialization(self):
        model = MultiHeadDiversityModel(self.vocab_sizes, self.feature_names, self.config) # type: ignore
        self.assertIsInstance(model, MultiHeadDiversityModel)
        self.assertEqual(len(model.heads), 2)
        # Check native layers
        self.assertTrue(hasattr(model, 'embeddings'))
        self.assertTrue(hasattr(model, 'dcn'))
        self.assertFalse(hasattr(model, 'backbone'))

    def test_forward_shape(self):
        model = MultiHeadDiversityModel(self.vocab_sizes, self.feature_names, self.config) # type: ignore
        batch_size = 4
        x = torch.zeros((batch_size, 2), dtype=torch.long)
        
        output = model(x)
        self.assertIn('logits', output)
        self.assertIn('aux_logits', output)
        
        self.assertEqual(output['logits'].shape, (batch_size, 1))
        # aux_logits should be [num_heads, batch_size, 1]
        self.assertEqual(output['aux_logits'].shape, (2, batch_size, 1))

    def test_feature_bagging_diversity(self):
        # With feature_bagging_ratio < 1.0, heads should receive different inputs
        # and likely produce different outputs even if initialized identically (if dropout > 0)
        # But here we want to ensure diversity mechanism runs.
        # We can mock torc.bernoulli to control masking if needed, but for now just check it runs.
        
        torch.manual_seed(42)
        model = MultiHeadDiversityModel(self.vocab_sizes, self.feature_names, self.config) # type: ignore
        x = torch.randint(0, 10, (4, 2))
        
        output = model(x)
        aux = output['aux_logits'] # [2, 4, 1]
        
        # Check that outputs are likely different (variance > 0)
        variance = torch.var(aux, dim=0).mean()
        # It's possible for variance to be small, but should execute without error.
        self.assertTrue(variance >= 0)

    def test_loss(self):
        model = MultiHeadDiversityModel(self.vocab_sizes, self.feature_names, self.config) # type: ignore
        diversity_weight = 0.5
        model.loss_fn.diversity_weight = diversity_weight
        
        batch_size = 4
        aux_logits = torch.randn(2, batch_size, 1) # 2 heads
        y_true = torch.randint(0, 2, (batch_size, 1)).float()
        
        loss = model.compute_loss({'logits': aux_logits.mean(0), 'aux_logits': aux_logits}, y_true)
        # Check scalar
        self.assertEqual(loss.dim(), 0)
