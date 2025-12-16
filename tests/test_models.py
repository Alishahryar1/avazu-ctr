"""
Test suite for model architecture and structure.

This module tests model components, DCN layers, variable embeddings,
and various model configurations.
"""

import unittest
import torch
import torch.nn as nn
from src.config.config import CONFIG, ConfigType
from src.models.model import GatedDCNModel, DCNv2


def make_test_config(**overrides) -> ConfigType:
    """Create a test config with optional overrides."""
    test_config: dict[str, object] = {
        # General
        'seed': 42,
        'device': 'cpu',

        # Data Loading
        'batch_size': 32,
        'num_workers': 0,
        'min_freq': 5,
        'validation_split': 0.1,

        # Model Architecture - Embeddings
        'embedding_dim': 16,
        'use_variable_embeddings': False,  # Disabled for backward compatibility in tests
        'embedding_dim_rules': [
            (10, 8),
            (100, 16),
            (1000, 32),
        ],
        'embedding_projection_dim': None,
        'feature_embedding_overrides': {},

        # Model Architecture - DCN/Attention
        'use_dcn': True,
        'dcn_num_layers': 2,
        'dcn_use_layernorm': False,
        'dcn_low_rank': None,
        'use_senet': True,
        'senet_squeeze_funcs': ['mean'],
        'senet_reduction_ratio': 3,
        'senet_activation': 'sigmoid',
        'use_feature_gating': False,
        'feature_gating_activation': 'sigmoid',
        'feature_gating_low_rank': None,
        'mlp_hidden_dims': [32, 16],
        'mlp_activation': 'relu',
        'mlp_use_skip_connections': False,
        'use_layer_norm': True,

        # Training
        'lr': 1e-3,
        'embedding_lr': 1.0,
        'embedding_optimizer': 'adagrad',
        'epochs': 1,
        'lr_warmup_epoch_ratio': 0.1,
        'early_stopping_patience': 3,
        'use_tensorboard': False,
        'tensorboard_logdir': './runs',
        'tensorboard_log_interval': 1000,

        # Automatic Mixed Precision (AMP)
        'auto_amp': False,  # Disabled for tests (CPU)
        'amp_dtype': 'float16',

        # Regularization
        'mlp_dropout': 0.1,
        'grad_clip': 1.0,
        'weight_decay': 1e-5,
        'embedding_weight_decay': 0.0,
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.0,

        # Paths
        'train_path': './data/train.gz',
        'test_path': './data/test.gz',
        'sub_path': 'submission.csv',
        'processed_path': './data',
        'models_path': './models',
    }
    # Apply overrides
    for key, value in overrides.items():
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
        if not self.config['use_dcn']:
            self.skipTest("DCN is disabled in config")
        expected_layers = self.config['dcn_num_layers']
        # Check either full-rank W or low-rank U (depending on config)
        if self.config['dcn_low_rank'] is not None:
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
        for name, embedding in self.model.embeddings.items():
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
        """Set up model with production config."""
        cls.vocab_sizes = {'f1': 100, 'f2': 100}
        cls.feature_names = ['f1', 'f2']
        cls.model = GatedDCNModel(cls.vocab_sizes, cls.feature_names, CONFIG)
    
    def test_forward_pass_with_production_config(self):
        """Verify model works with production config."""
        batch_size = 4
        num_features = len(self.feature_names)
        x = torch.randint(0, 100, (batch_size, num_features))
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertEqual(output['logits'].shape, (batch_size, 1), "Output shape mismatch")
    
    def test_production_config_dcn_layers(self):
        """Verify DCN layers match production config (if enabled)."""
        if not CONFIG['use_dcn']:
            self.skipTest("DCN is disabled in production config")
        
        expected_layers = CONFIG['dcn_num_layers']
        if CONFIG['dcn_low_rank'] is not None:
            actual_layers = len(self.model.dcn.U)
        else:
            actual_layers = len(self.model.dcn.W)
        self.assertEqual(actual_layers, expected_layers)

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


class TestVariableEmbeddings(unittest.TestCase):
    """Tests for variable embedding dimensions based on feature cardinality."""
    
    def test_cardinality_based_embedding_dims(self):
        """Test that embeddings get different dimensions based on vocab size."""
        config = make_test_config(
            use_variable_embeddings=True,
            use_senet=False,  # SENET needs uniform dims
            use_feature_gating=True
        )
        # Create features with different cardinalities
        vocab_sizes = {
            'small': 5,      # Should get 8 dims (cardinality <= 10)
            'medium': 50,    # Should get 16 dims (cardinality <= 100)
            'large': 500,    # Should get 32 dims (cardinality <= 1000)
        }
        feature_names = ['small', 'medium', 'large']
        
        model = GatedDCNModel(vocab_sizes, feature_names, config)
        
        # Verify each feature has correct embedding dimension
        self.assertEqual(model.embeddings['small'].embedding_dim, 8)
        self.assertEqual(model.embeddings['medium'].embedding_dim, 16)
        self.assertEqual(model.embeddings['large'].embedding_dim, 32)
    
    def test_variable_embeddings_forward_pass(self):
        """Test forward pass with variable embedding dimensions."""
        config = make_test_config(
            use_variable_embeddings=True,
            use_senet=False,
            use_feature_gating=True
        )
        vocab_sizes = {'small': 5, 'medium': 50, 'large': 500}
        model = GatedDCNModel(vocab_sizes, ['small', 'medium', 'large'], config)
        
        x = torch.randint(0, 5, (4, 3))  # Use min vocab size for safety
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_projection_layer(self):
        """Test that projection layer unifies variable embedding dimensions."""
        config = make_test_config(
            use_variable_embeddings=True,
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
        """Test that SENET works with variable embeddings when projection is enabled."""
        config = make_test_config(
            use_variable_embeddings=True,
            embedding_projection_dim=64,  # Must divide evenly by num_fields for SENET
            use_senet=True,
            use_feature_gating=False
        )
        vocab_sizes = {'small': 5, 'medium': 50}  # 2 fields
        model = GatedDCNModel(vocab_sizes, ['small', 'medium'], config)
        
        self.assertTrue(hasattr(model, 'senet'))
        x = torch.randint(0, 5, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_senet_requires_uniform_or_projection(self):
        """Test that SENET raises error with variable embeddings and no projection."""
        config = make_test_config(
            use_variable_embeddings=True,
            embedding_projection_dim=None,
            use_senet=True,
            use_feature_gating=False
        )
        vocab_sizes = {'small': 5, 'large': 500}  # Different dims
        
        with self.assertRaises(ValueError) as context:
            GatedDCNModel(vocab_sizes, ['small', 'large'], config)
        self.assertIn("SENET requires uniform embedding dimensions", str(context.exception))
    
    def test_feature_embedding_overrides(self):
        """Test per-feature embedding dimension overrides."""
        config = make_test_config(
            use_variable_embeddings=True,
            feature_embedding_overrides={
                'special': {'embedding_dim': 128}  # Override to 128
            },
            use_senet=False,
            use_feature_gating=True
        )
        vocab_sizes = {'small': 5, 'special': 50}
        model = GatedDCNModel(vocab_sizes, ['small', 'special'], config)
        
        # 'small' should use cardinality rules (8 dims for <= 10)
        self.assertEqual(model.embeddings['small'].embedding_dim, 8)
        # 'special' should use override (128 dims)
        self.assertEqual(model.embeddings['special'].embedding_dim, 128)
    
    def test_default_embedding_fallback(self):
        """Test that high-cardinality features use default embedding_dim."""
        config = make_test_config(
            use_variable_embeddings=True,
            embedding_dim=64,  # Default for high cardinality
            embedding_dim_rules=[(10, 8), (100, 16)],  # No rule for > 100
            use_senet=False
        )
        vocab_sizes = {'small': 5, 'huge': 10000}  # 10000 > 100, uses default
        model = GatedDCNModel(vocab_sizes, ['small', 'huge'], config)
        
        self.assertEqual(model.embeddings['small'].embedding_dim, 8)
        self.assertEqual(model.embeddings['huge'].embedding_dim, 64)  # Default


if __name__ == "__main__":
    unittest.main(verbosity=2)
