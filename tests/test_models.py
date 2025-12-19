"""
Test suite for model architecture and structure.

This module tests model components, DCN layers, variable embeddings,
and various model configurations.
"""

import unittest
from typing import Any, cast
import torch
import torch.nn as nn
from config import CONFIG, ConfigType
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.layers.cross_network import DCNv2


def make_test_config(**overrides) -> ConfigType:
    """Create a test config with optional overrides.
    
    The config follows the production pattern:
    - Global settings at top level (embedding_dim, feature_embeddings, etc.)
    - Model-specific settings nested under 'model' key
    
    Overrides can be passed for both levels. For model-level overrides,
    pass keys like 'use_dcn', 'mlp_hidden_dims' directly - they'll be
    placed in the model dict automatically.
    """
    # Define which keys belong in the model config
    model_keys = {
        'use_dcn', 'dcn_num_layers', 'dcn_use_layernorm', 'dcn_low_rank',
        'use_senet', 'senet_squeeze_funcs', 'senet_reduction_ratio',
        'senet_hidden_activation', 'senet_excitation_activation',
        'senet_num_groups', 'senet_reweight_mode', 'senet_use_fuse', 'senet_use_layer_norm',
        'use_feature_gating', 'feature_gating_activation', 'feature_gating_low_rank',
        'mlp_hidden_dims', 'mlp_activation', 'mlp_use_skip_connections', 'mlp_dropout',
        'use_layer_norm', 'focal_loss_gamma', 'label_smoothing',
        # STEC-specific keys
        'stec_num_layers', 'stec_num_heads', 'stec_hidden_dim', 'stec_dropout',
        'stec_use_ffn', 'stec_mlp_hidden_dims',
    }
    
    # Base model config (GatedDCN defaults)
    model_config: dict[str, object] = {
        'use_dcn': True,
        'dcn_num_layers': 2,
        'dcn_use_layernorm': False,
        'dcn_low_rank': None,
        'use_senet': True,
        'senet_squeeze_funcs': ['mean'],
        'senet_reduction_ratio': 3,
        'senet_hidden_activation': 'relu',
        'senet_excitation_activation': 'sigmoid',
        'use_feature_gating': False,
        'feature_gating_activation': 'sigmoid',
        'feature_gating_low_rank': None,
        'mlp_hidden_dims': [32, 16],
        'mlp_activation': 'relu',
        'mlp_use_skip_connections': False,
        'mlp_dropout': 0.1,
        'use_layer_norm': True,
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.0,
    }
    
    # Apply model-level overrides
    for key, value in overrides.items():
        if key in model_keys:
            model_config[key] = value
    
    test_config: dict[str, object] = {
        # General
        'seed': 42,
        'device': 'cpu',

        # Data Loading
        'batch_size': 32,
        'num_workers': 0,
        'min_freq': 5,
        'validation_split': 0.1,
        'shuffle_train': False,

        # Model Architecture - Embeddings (top-level)
        'embedding_dim': overrides.get('embedding_dim', 16),
        'feature_embeddings': overrides.get('feature_embeddings', {}),
        'embedding_projection_dim': overrides.get('embedding_projection_dim', None),
        
        # Model config (nested)
        'model': model_config,

        # Training
        'epochs': 1,
        'early_stopping_patience': 3,
        'grad_clip': 1.0,
        'use_tensorboard': False,
        'tensorboard_logdir': './runs',
        'tensorboard_log_interval': 1000,

        # Optimizer Configuration
        'dense_optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'warmup_epoch_ratio': 0.1,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
        'embedding_optimizer': {
            'type': 'adagrad',
            'lr': 1.0,
            'warmup_epoch_ratio': 0.0,
            'weight_decay': 0.0,
            'eps': 1e-10,
            'lr_decay': 0.0,
        },

        # Automatic Mixed Precision (AMP)
        'auto_amp': True,  # Disabled for tests (CPU)
        'amp_dtype': 'float16',
        
        # Model Compilation
        'compile_model': False,

        # Loss settings (for test config reference)
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.0,

        # Paths
        'train_path': './data/train.gz',
        'test_path': './data/test.gz',
        'sub_path': 'submission.csv',
        'processed_path': './data',
        'models_path': './models',
    }
    
    # Apply non-model overrides directly to test_config
    for key, value in overrides.items():
        if key not in model_keys and key not in ('embedding_dim', 'feature_embeddings', 'embedding_projection_dim'):
            test_config[key] = value
    
    return test_config  # type: ignore[return-value]


class TestModelStructure(unittest.TestCase):
    """Tests for model architecture and configuration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up mock data used across all tests in this class."""
        cls.vocab_sizes = {'f1': 100, 'f2': 100}
        cls.feature_names = ['f1', 'f2']
        cls.config = make_test_config()
        cls.model = GatedDCNModel(cls.vocab_sizes, cls.feature_names, cls.config)
    
    def test_dcn_layers(self):
        """Verify DCN has the correct number of layers (if enabled)."""
        model_config = cast(dict[str, Any], self.config['model'])
        if not model_config['use_dcn']:
            self.skipTest("DCN is disabled in config")
        expected_layers = model_config['dcn_num_layers']
        # Check either full-rank W or low-rank U (depending on config)
        if model_config['dcn_low_rank'] is not None:
            actual_layers = len(self.model.dcn.U)
        else:
            actual_layers = len(self.model.dcn.W)
        self.assertEqual(
            actual_layers, 
            expected_layers,
            f"Expected {expected_layers} DCN layers, got {actual_layers}"
        )
    
    def test_mlp_structure(self):
        """Verify MLP has layers for each hidden dim.
        
        MLP typically contains Linear, BatchNorm/ReLU, Dropout layers.
        The exact structure depends on use_layer_norm config.
        """
        # Import ResidualMLP to check type
        from src.models.model import ResidualMLP
        
        # Check MLP is a ResidualMLP
        self.assertIsInstance(self.model.mlp, ResidualMLP, "MLP should be ResidualMLP")
        
        # Check it has layers (at least one hidden layer)
        self.assertGreater(len(self.model.mlp.layers), 0, "MLP should have layers")
        
        # Check final output layer exists and outputs single value
        output_layer = self.model.mlp.output_layer
        self.assertIsInstance(output_layer, nn.Linear, "Output layer should be Linear")
        self.assertEqual(output_layer.out_features, 1, "Output layer should output 1")
    
    def test_model_forward_pass(self):
        """Verify model can perform a forward pass."""
        batch_size = 4
        num_features = len(self.feature_names)
        x = torch.randint(0, 100, (batch_size, num_features))
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertEqual(output['logits'].shape, (batch_size, 1), "Output shape mismatch")
    
    def test_embedding_dim(self):
        """Verify embeddings have correct dimension."""
        expected_dim = self.config['embedding_dim']
        for name in self.feature_names:
            embedding = self.model.embeddings[name]
            actual_dim = embedding.embedding_dim
            self.assertEqual(
                actual_dim,
                expected_dim,
                f"Embedding '{name}' has dim {actual_dim}, expected {expected_dim}"
            )


class TestModelWithProductionConfig(unittest.TestCase):
    """Tests using the actual production CONFIG."""
    
    @classmethod
    def setUpClass(cls):
        """Set up model with production config using factory function."""
        from src.models.architectures import create_model
        cls.vocab_sizes = {'f1': 100, 'f2': 100}
        cls.feature_names = ['f1', 'f2']
        cls.model = create_model(CONFIG, cls.vocab_sizes, cls.feature_names)
    
    def test_forward_pass_with_production_config(self):
        """Verify model works with production config."""
        batch_size = 4
        num_features = len(self.feature_names)
        x = torch.randint(0, 100, (batch_size, num_features))
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertEqual(output['logits'].shape, (batch_size, 1), "Output shape mismatch")
    
    def test_production_config_model_type(self):
        """Verify production config creates the expected model type."""
        model_config = CONFIG['model']
        
        # Check if ensemble or single model
        if 'models' in model_config:
            from src.models.architectures.ensemble import EnsembleModel
            self.assertIsInstance(self.model, EnsembleModel)
        elif 'stec_num_layers' in model_config:
            from src.models.architectures.stec import STECModel
            self.assertIsInstance(self.model, STECModel)
        elif 'use_dcn' in model_config:
            self.assertIsInstance(self.model, GatedDCNModel)


class TestDCNv2LowRank(unittest.TestCase):
    """Tests for DCNv2 low-rank decomposition."""
    
    def test_full_rank_forward(self):
        """Test DCNv2 with full-rank (low_rank=None)."""
        dcn = DCNv2(input_dim=64, num_layers=2, low_rank=None)
        x = torch.randn(4, 64)
        out = dcn(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_low_rank_forward(self):
        """Test DCNv2 with low-rank decomposition."""
        dcn = DCNv2(input_dim=64, num_layers=2, low_rank=16)
        x = torch.randn(4, 64)
        out = dcn(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_low_rank_parameter_reduction(self):
        """Verify low-rank reduces parameters."""
        input_dim = 128
        num_layers = 2
        low_rank = 32
        
        dcn_full = DCNv2(input_dim=input_dim, num_layers=num_layers, low_rank=None)
        dcn_low = DCNv2(input_dim=input_dim, num_layers=num_layers, low_rank=low_rank)
        
        full_params = sum(p.numel() for p in dcn_full.parameters())
        low_params = sum(p.numel() for p in dcn_low.parameters())
        
        self.assertLess(low_params, full_params, "Low-rank should have fewer parameters")
    
    def test_low_rank_numerical_stability(self):
        """Verify no NaN/Inf in low-rank output."""
        dcn = DCNv2(input_dim=64, num_layers=4, use_layernorm=True, low_rank=16)
        for _ in range(10):
            x = torch.randn(32, 64)
            out = dcn(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")
    
    def test_low_rank_gradient_flow(self):
        """Verify gradients flow through low-rank DCNv2."""
        dcn = DCNv2(input_dim=64, num_layers=2, low_rank=16)
        x = torch.randn(4, 64, requires_grad=True)
        out = dcn(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in dcn.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
    
    def test_low_rank_has_U_V_matrices(self):
        """Verify low-rank DCNv2 has U and V matrices."""
        dcn = DCNv2(input_dim=64, num_layers=3, low_rank=16)
        self.assertTrue(hasattr(dcn, 'U'), "Low-rank DCNv2 should have U matrices")
        self.assertTrue(hasattr(dcn, 'V'), "Low-rank DCNv2 should have V matrices")
        self.assertEqual(len(dcn.U), 3, "Should have 3 U matrices")
        self.assertEqual(len(dcn.V), 3, "Should have 3 V matrices")
    
    def test_full_rank_has_W_matrices(self):
        """Verify full-rank DCNv2 has W matrices."""
        dcn = DCNv2(input_dim=64, num_layers=3, low_rank=None)
        self.assertTrue(hasattr(dcn, 'W'), "Full-rank DCNv2 should have W matrices")
        self.assertEqual(len(dcn.W), 3, "Should have 3 W matrices")


class TestModelVariants(unittest.TestCase):
    """Test model with different config combinations."""
    
    def test_model_without_dcn(self):
        """Test model with DCN disabled."""
        config = make_test_config(use_dcn=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_model_without_senet(self):
        """Test model with SENET disabled."""
        config = make_test_config(use_senet=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_model_without_batch_norm(self):
        """Test model with batch norm disabled."""
        config = make_test_config(use_layer_norm=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_model_with_low_rank_dcn(self):
        """Test model with low-rank DCN."""
        config = make_test_config(dcn_low_rank=8)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_model_minimal(self):
        """Test model with minimal config (no DCN, no SENET)."""
        config = make_test_config(use_dcn=False, use_senet=False, use_layer_norm=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
class TestMutualExclusivity(unittest.TestCase):
    """Tests for mutual exclusivity between SENET and Feature Gating."""
    
    def test_both_senet_and_feature_gating_raises_error(self):
        """Verify that enabling both SENET and Feature Gating raises ValueError."""
        config = make_test_config(use_senet=True, use_feature_gating=True)
        with self.assertRaises(ValueError) as context:
            GatedDCNModel({'f1': 100}, ['f1'], config)
        self.assertIn("Cannot enable both SENET and Feature Gating", str(context.exception))
    
    def test_model_with_feature_gating_only(self):
        """Test model with Feature Gating enabled and SENET disabled."""
        config = make_test_config(use_senet=False, use_feature_gating=True)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
        self.assertTrue(hasattr(model, 'feature_gating'), "Model should have feature_gating layer")
    
    def test_model_with_neither_senet_nor_feature_gating(self):
        """Test model with both SENET and Feature Gating disabled."""
        config = make_test_config(use_senet=False, use_feature_gating=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_model_with_senet_only(self):
        """Test model with SENET enabled and Feature Gating disabled (default)."""
        config = make_test_config(use_senet=True, use_feature_gating=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
        self.assertTrue(hasattr(model, 'senet'), "Model should have senet layer")


class TestFeatureEmbeddings(unittest.TestCase):
    """Tests for per-feature embedding configurations including hash embeddings."""
    
    def test_standard_embeddings_per_feature(self):
        """Test that per-feature standard embeddings work correctly."""
        config = make_test_config(
            feature_embeddings={
                'small': {'type': 'standard', 'dim': 8},
                'medium': {'type': 'standard', 'dim': 16},
                'large': {'type': 'standard', 'dim': 32},
            },
            use_senet=False,  # SENET needs uniform dims
            use_feature_gating=True
        )
        vocab_sizes = {
            'small': 5,
            'medium': 50,
            'large': 500,
        }
        feature_names = ['small', 'medium', 'large']
        
        model = GatedDCNModel(vocab_sizes, feature_names, config)
        
        # Verify each feature has correct embedding dimension
        self.assertEqual(model.embeddings['small'].embedding_dim, 8)
        self.assertEqual(model.embeddings['medium'].embedding_dim, 16)
        self.assertEqual(model.embeddings['large'].embedding_dim, 32)
    
    def test_hash_embeddings_per_feature(self):
        """Test that hash embeddings work correctly."""
        from src.models.layers.hash_embedding import HashEmbedding
        
        config = make_test_config(
            feature_embeddings={
                'hash_feat': {'type': 'hash', 'dim': 16, 'num_buckets': 100, 'num_hashes': 2},
                'std_feat': {'type': 'standard', 'dim': 16},
            },
            use_senet=False,
            use_feature_gating=True
        )
        vocab_sizes = {'hash_feat': 1000, 'std_feat': 100}
        model = GatedDCNModel(vocab_sizes, ['hash_feat', 'std_feat'], config)
        
        # Verify hash embedding uses HashEmbedding
        self.assertIsInstance(model.embeddings['hash_feat'], HashEmbedding)
        # Verify standard embedding uses nn.Embedding
        self.assertIsInstance(model.embeddings['std_feat'], nn.Embedding)
    
    def test_mixed_embeddings_forward_pass(self):
        """Test forward pass with mixed embedding types."""
        config = make_test_config(
            feature_embeddings={
                'hash_feat': {'type': 'hash', 'dim': 16, 'num_buckets': 50},
                'std_feat': {'type': 'standard', 'dim': 16},
            },
            use_senet=False,
            use_feature_gating=True
        )
        vocab_sizes = {'hash_feat': 500, 'std_feat': 100}
        model = GatedDCNModel(vocab_sizes, ['hash_feat', 'std_feat'], config)
        
        x = torch.randint(0, 100, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_projection_layer(self):
        """Test that projection layer unifies feature embedding dimensions."""
        config = make_test_config(
            feature_embeddings={
                'small': {'type': 'standard', 'dim': 8},
                'medium': {'type': 'standard', 'dim': 16},
            },
            embedding_projection_dim=64,  # Project to 64 dims
            use_senet=False,
            use_feature_gating=False
        )
        vocab_sizes = {'small': 5, 'medium': 50}
        model = GatedDCNModel(vocab_sizes, ['small', 'medium'], config)
        
        self.assertTrue(model.use_projection)
        self.assertTrue(hasattr(model, 'projection'))
        
        # Forward pass should work
        x = torch.randint(0, 5, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_projection_with_senet(self):
        """Test that SENET works with projection layer."""
        config = make_test_config(
            feature_embeddings={
                'feat1': {'type': 'standard', 'dim': 8},
                'feat2': {'type': 'standard', 'dim': 16},
            },
            embedding_projection_dim=64,  # Must divide evenly by num_fields for SENET
            use_senet=True,
            use_feature_gating=False
        )
        vocab_sizes = {'feat1': 5, 'feat2': 50}  # 2 fields
        model = GatedDCNModel(vocab_sizes, ['feat1', 'feat2'], config)
        
        self.assertTrue(hasattr(model, 'senet'))
        x = torch.randint(0, 5, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_senet_with_variable_embeddings_no_projection(self):
        """Test that SENET works with non-uniform embeddings without projection."""
        config = make_test_config(
            feature_embeddings={
                'small': {'type': 'standard', 'dim': 8},
                'large': {'type': 'standard', 'dim': 32},
            },
            embedding_projection_dim=None,
            use_senet=True,
            use_feature_gating=False
        )
        vocab_sizes = {'small': 5, 'large': 500}
        
        # Should now work without raising an error
        model = GatedDCNModel(vocab_sizes, ['small', 'large'], config)
        self.assertTrue(hasattr(model, 'senet'), "Model should have senet layer")
        
        # Forward pass should work
        x = torch.randint(0, 5, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_default_embedding_fallback(self):
        """Test that features not in feature_embeddings use default embedding_dim."""
        config = make_test_config(
            embedding_dim=64,  # Default for unspecified features
            feature_embeddings={
                'specified': {'type': 'standard', 'dim': 16},
            },
            use_senet=False
        )
        vocab_sizes = {'specified': 5, 'unspecified': 100}
        model = GatedDCNModel(vocab_sizes, ['specified', 'unspecified'], config)
        
        self.assertEqual(model.embeddings['specified'].embedding_dim, 16)
        self.assertEqual(model.embeddings['unspecified'].embedding_dim, 64)  # Default
    
    def test_hash_embedding_aggregation_modes(self):
        """Test that hash embedding aggregation modes work correctly."""
        from src.models.layers.hash_embedding import HashEmbedding
        
        for mode in ['sum', 'concatenate', 'median']:
            config = make_test_config(
                feature_embeddings={
                    'feat': {'type': 'hash', 'dim': 16, 'num_buckets': 50, 'aggregation_mode': mode},
                },
                use_senet=False,
                use_feature_gating=True
            )
            vocab_sizes = {'feat': 100}
            model = GatedDCNModel(vocab_sizes, ['feat'], config)
            
            # Verify embedding was created with correct mode
            emb = model.embeddings['feat']
            self.assertIsInstance(emb, HashEmbedding)
            self.assertEqual(emb.aggregation_mode, mode)
            
            # Forward pass should work
            x = torch.randint(0, 100, (4, 1))
            out = model(x)
            self.assertEqual(out['logits'].shape, (4, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
