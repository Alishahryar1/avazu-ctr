"""
Test suite for the Avazu CTR project.

Usage:
    Run all tests:         python tests.py
    Run specific test:     python tests.py TestModelStructure.test_dcn_layers
    Run test class:        python tests.py TestModelStructure
    Verbose output:        python tests.py -v
"""

import unittest
import sys
import torch
import torch.nn as nn
import numpy as np
from config import CONFIG, ConfigType
from model import GatedDCNModel, DCNv2, SENetLayer, FeatureGatingLayer


def make_test_config(**overrides) -> ConfigType:
    """Create a test config with optional overrides."""
    test_config: ConfigType = {
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
        test_config[key] = value  # type: ignore
    return test_config


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
        from model import ResidualMLP
        
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
        
        self.assertEqual(output.shape, (batch_size, 1), "Output shape mismatch")
    
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
        
        self.assertEqual(output.shape, (batch_size, 1), "Output shape mismatch")
    
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


class TestConfig(unittest.TestCase):
    """Tests for configuration validity."""
    
    def test_required_keys_exist(self):
        """Verify all required config keys are present."""
        required_keys = [
            'embedding_dim', 
            'dcn_num_layers', 
            'mlp_hidden_dims',
            'mlp_dropout',
            'lr',
            'batch_size',
            'epochs'
        ]
        for key in required_keys:
            self.assertIn(key, CONFIG, f"Missing required config key: {key}")
    
    def test_config_value_types(self):
        """Verify config values have correct types."""
        self.assertIsInstance(CONFIG['embedding_dim'], int)
        self.assertIsInstance(CONFIG['dcn_num_layers'], int)
        self.assertIsInstance(CONFIG['mlp_hidden_dims'], (list, tuple))
        self.assertIsInstance(CONFIG['mlp_dropout'], float)
    
    def test_config_value_ranges(self):
        """Verify config values are in valid ranges."""
        self.assertGreater(CONFIG['embedding_dim'], 0, "embedding_dim must be positive")
        self.assertGreater(CONFIG['dcn_num_layers'], 0, "dcn_num_layers must be positive")
        self.assertGreaterEqual(CONFIG['mlp_dropout'], 0.0, "mlp_dropout must be >= 0")
        self.assertLess(CONFIG['mlp_dropout'], 1.0, "mlp_dropout must be < 1")
        if CONFIG['dcn_low_rank'] is not None:
            self.assertGreater(CONFIG['dcn_low_rank'], 0, "dcn_low_rank must be positive if set")


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
        self.assertEqual(out.shape, (4, 1))
    
    def test_model_without_senet(self):
        """Test model with SENET disabled."""
        config = make_test_config(use_senet=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_model_without_batch_norm(self):
        """Test model with batch norm disabled."""
        config = make_test_config(use_layer_norm=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_model_with_low_rank_dcn(self):
        """Test model with low-rank DCN."""
        config = make_test_config(dcn_low_rank=8)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_model_minimal(self):
        """Test model with minimal config (no DCN, no SENET)."""
        config = make_test_config(use_dcn=False, use_senet=False, use_layer_norm=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))


class TestSENetLayer(unittest.TestCase):
    """Tests for SENetLayer (Squeeze-and-Excitation network)."""
    
    def test_senet_forward_single_squeeze(self):
        """Test SENET forward pass with single squeeze function."""
        num_fields = 5
        embed_dim = 16
        senet = SENetLayer(num_fields=num_fields, embedding_dim=embed_dim, squeeze_funcs=['mean'])
        x = torch.randn(4, num_fields * embed_dim)
        out = senet(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_senet_forward_multiple_squeeze(self):
        """Test SENET forward pass with multiple squeeze functions (mean + max)."""
        num_fields = 5
        embed_dim = 16
        senet = SENetLayer(num_fields=num_fields, embedding_dim=embed_dim, squeeze_funcs=['mean', 'max'])
        x = torch.randn(4, num_fields * embed_dim)
        out = senet(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_senet_numerical_stability(self):
        """Verify no NaN/Inf in SENET output."""
        senet = SENetLayer(num_fields=10, embedding_dim=32, squeeze_funcs=['mean', 'max'])
        for _ in range(10):
            x = torch.randn(32, 320)
            out = senet(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")
    
    def test_senet_gradient_flow(self):
        """Verify gradients flow through SENetLayer."""
        senet = SENetLayer(num_fields=5, embedding_dim=16, squeeze_funcs=['mean', 'max'])
        x = torch.randn(4, 80, requires_grad=True)
        out = senet(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in senet.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
    
    def test_senet_invalid_squeeze_func(self):
        """Verify SENET raises error for invalid squeeze function."""
        with self.assertRaises(ValueError):
            SENetLayer(num_fields=5, embedding_dim=16, squeeze_funcs=['invalid'])
    
    def test_senet_activations(self):
        """Test SENET with different activation functions."""
        for activation in ['sigmoid', 'tanh', 'relu', 'softmax']:
            senet = SENetLayer(num_fields=5, embedding_dim=16, excitation_activation=activation)
            x = torch.randn(4, 80)
            out = senet(x)
            self.assertEqual(out.shape, x.shape, f"Failed for activation: {activation}")


class TestFeatureGatingLayer(unittest.TestCase):
    """Tests for FeatureGatingLayer."""
    
    def test_feature_gating_forward(self):
        """Test Feature Gating forward pass (full-rank)."""
        input_dim = 80
        gating = FeatureGatingLayer(input_dim=input_dim, gating_activation='sigmoid')
        x = torch.randn(4, input_dim)
        out = gating(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_feature_gating_low_rank_forward(self):
        """Test Feature Gating forward pass with low-rank decomposition."""
        input_dim = 80
        gating = FeatureGatingLayer(input_dim=input_dim, gating_activation='sigmoid', low_rank=16)
        x = torch.randn(4, input_dim)
        out = gating(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_feature_gating_low_rank_parameter_reduction(self):
        """Verify low-rank reduces parameters in FeatureGatingLayer."""
        input_dim = 128
        low_rank = 32
        
        gating_full = FeatureGatingLayer(input_dim=input_dim, low_rank=None)
        gating_low = FeatureGatingLayer(input_dim=input_dim, low_rank=low_rank)
        
        full_params = sum(p.numel() for p in gating_full.parameters())
        low_params = sum(p.numel() for p in gating_low.parameters())
        
        self.assertLess(low_params, full_params, "Low-rank should have fewer parameters")
    
    def test_feature_gating_low_rank_has_U_V_matrices(self):
        """Verify low-rank FeatureGatingLayer has U, V, and bias."""
        gating = FeatureGatingLayer(input_dim=64, low_rank=16)
        self.assertTrue(hasattr(gating, 'U'), "Low-rank should have U matrix")
        self.assertTrue(hasattr(gating, 'V'), "Low-rank should have V matrix")
        self.assertTrue(hasattr(gating, 'bias'), "Low-rank should have bias")
        self.assertEqual(gating.U.shape, (64, 16), "U shape mismatch")
        self.assertEqual(gating.V.shape, (16, 64), "V shape mismatch")
    
    def test_feature_gating_full_rank_has_linear(self):
        """Verify full-rank FeatureGatingLayer has gate_linear."""
        gating = FeatureGatingLayer(input_dim=64, low_rank=None)
        self.assertTrue(hasattr(gating, 'gate_linear'), "Full-rank should have gate_linear")
    
    def test_feature_gating_numerical_stability(self):
        """Verify no NaN/Inf in Feature Gating output."""
        gating = FeatureGatingLayer(input_dim=320, gating_activation='sigmoid')
        for _ in range(10):
            x = torch.randn(32, 320)
            out = gating(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")
    
    def test_feature_gating_low_rank_numerical_stability(self):
        """Verify no NaN/Inf in low-rank Feature Gating output."""
        gating = FeatureGatingLayer(input_dim=320, gating_activation='sigmoid', low_rank=32)
        for _ in range(10):
            x = torch.randn(32, 320)
            out = gating(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")
    
    def test_feature_gating_gradient_flow(self):
        """Verify gradients flow through FeatureGatingLayer (full-rank)."""
        gating = FeatureGatingLayer(input_dim=80, gating_activation='sigmoid')
        x = torch.randn(4, 80, requires_grad=True)
        out = gating(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in gating.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
    
    def test_feature_gating_low_rank_gradient_flow(self):
        """Verify gradients flow through low-rank FeatureGatingLayer."""
        gating = FeatureGatingLayer(input_dim=80, gating_activation='sigmoid', low_rank=16)
        x = torch.randn(4, 80, requires_grad=True)
        out = gating(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in gating.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
    
    def test_feature_gating_activations(self):
        """Test Feature Gating with different activation functions."""
        for activation in ['sigmoid', 'tanh', 'relu', 'gelu', 'silu']:
            gating = FeatureGatingLayer(input_dim=80, gating_activation=activation)
            x = torch.randn(4, 80)
            out = gating(x)
            self.assertEqual(out.shape, x.shape, f"Failed for activation: {activation}")
    
    def test_feature_gating_low_rank_activations(self):
        """Test low-rank Feature Gating with different activation functions."""
        for activation in ['sigmoid', 'tanh', 'relu', 'gelu', 'silu']:
            gating = FeatureGatingLayer(input_dim=80, gating_activation=activation, low_rank=16)
            x = torch.randn(4, 80)
            out = gating(x)
            self.assertEqual(out.shape, x.shape, f"Failed for activation: {activation}")


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
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(hasattr(model, 'feature_gating'), "Model should have feature_gating layer")
    
    def test_model_with_neither_senet_nor_feature_gating(self):
        """Test model with both SENET and Feature Gating disabled."""
        config = make_test_config(use_senet=False, use_feature_gating=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_model_with_senet_only(self):
        """Test model with SENET enabled and Feature Gating disabled (default)."""
        config = make_test_config(use_senet=True, use_feature_gating=False)
        model = GatedDCNModel({'f1': 100}, ['f1'], config)
        x = torch.randint(0, 100, (4, 1))
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
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
        self.assertEqual(out.shape, (4, 1))
    
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
        self.assertEqual(out.shape, (4, 1))
    
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
        self.assertEqual(out.shape, (4, 1))
    
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


# =============================================================================
# Tests for train.py - FocalLoss
# =============================================================================
class TestFocalLoss(unittest.TestCase):
    """Tests for Focal Loss implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Import FocalLoss from train module."""
        from train import FocalLoss
        cls.FocalLoss = FocalLoss
    
    def test_focal_loss_forward_basic(self):
        """Test basic forward pass."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1)).float()
        loss = focal(logits, targets)
        
        self.assertEqual(loss.dim(), 0)  # Scalar output
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_focal_loss_gamma_zero_equals_bce(self):
        """Verify gamma=0 is equivalent to standard BCE."""
        focal = self.FocalLoss(gamma=0.0)
        bce = nn.BCEWithLogitsLoss()
        
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()
        
        focal_loss = focal(logits, targets)
        bce_loss = bce(logits, targets)
        
        self.assertAlmostEqual(focal_loss.item(), bce_loss.item(), places=5)
    
    def test_focal_loss_down_weights_easy_examples(self):
        """Verify focal loss down-weights confident predictions."""
        focal_high_gamma = self.FocalLoss(gamma=5.0)
        focal_low_gamma = self.FocalLoss(gamma=0.0)
        
        # Easy example: prediction and target are the same (high confidence)
        easy_logits = torch.tensor([[10.0]])  # Very confident positive
        easy_targets = torch.tensor([[1.0]])
        
        focal_high = focal_high_gamma(easy_logits, easy_targets)
        focal_low = focal_low_gamma(easy_logits, easy_targets)
        
        # High gamma should have lower loss for easy examples
        self.assertLess(focal_high.item(), focal_low.item())
    
    def test_focal_loss_with_alpha(self):
        """Test focal loss with class weights (alpha)."""
        focal = self.FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1)).float()
        loss = focal(logits, targets)
        
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_all_zeros_targets(self):
        """Test focal loss when all targets are 0."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.zeros(8, 1)
        loss = focal(logits, targets)
        
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_all_ones_targets(self):
        """Test focal loss when all targets are 1."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.ones(8, 1)
        loss = focal(logits, targets)
        
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_gradient_flow(self):
        """Verify gradients flow through focal loss."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(4, 1, requires_grad=True)
        targets = torch.randint(0, 2, (4, 1)).float()
        
        loss = focal(logits, targets)
        loss.backward()
        
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


# =============================================================================
# Tests for train.py - LRSchedulerWithWarmup
# =============================================================================
class TestLRSchedulerWithWarmup(unittest.TestCase):
    """Tests for learning rate scheduler with warmup."""
    
    @classmethod
    def setUpClass(cls):
        """Import LRSchedulerWithWarmup from train module."""
        from train import LRSchedulerWithWarmup
        cls.LRSchedulerWithWarmup = LRSchedulerWithWarmup
    
    def test_warmup_phase(self):
        """Test that LR increases linearly during warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=100, total_steps=1000
        )
        
        initial_lr = scheduler.get_lr()
        self.assertEqual(initial_lr, 0.001)  # Base LR
        
        # After a few warmup steps, LR should be increasing
        for _ in range(10):
            scheduler.step()
        
        warmup_lr = scheduler.get_lr()
        self.assertLess(warmup_lr, initial_lr)  # Still in warmup (below base)
        self.assertGreater(warmup_lr, 0)
    
    def test_lr_at_warmup_end(self):
        """Test that LR equals base_lr exactly at end of warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=100, total_steps=1000
        )
        
        # Run exactly to end of warmup
        for _ in range(100):
            scheduler.step()
        
        lr_at_warmup_end = scheduler.get_lr()
        # At step 100 (end of warmup), we start cosine decay, so LR should be close to base
        self.assertAlmostEqual(lr_at_warmup_end, 0.01, places=3)
    
    def test_cosine_decay_phase(self):
        """Test that LR decays after warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6
        )
        
        # Complete warmup
        for _ in range(10):
            scheduler.step()
        lr_after_warmup = scheduler.get_lr()
        
        # Continue into decay phase
        for _ in range(50):
            scheduler.step()
        lr_mid_decay = scheduler.get_lr()
        
        # LR should have decayed
        self.assertLess(lr_mid_decay, lr_after_warmup)
        self.assertGreater(lr_mid_decay, 1e-6)  # Above min_lr
    
    def test_lr_at_end(self):
        """Test that LR approaches min_lr at end of training."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        min_lr = 1e-5
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100, min_lr=min_lr
        )
        
        # Run to completion
        for _ in range(100):
            scheduler.step()
        
        final_lr = scheduler.get_lr()
        # LR should be close to min_lr at end
        self.assertAlmostEqual(final_lr, min_lr, places=5)
    
    def test_get_lr_returns_current(self):
        """Test that get_lr returns current optimizer LR."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100
        )
        
        for _ in range(5):
            scheduler.step()
        
        reported_lr = scheduler.get_lr()
        actual_lr = optimizer.param_groups[0]['lr']
        self.assertEqual(reported_lr, actual_lr)


# =============================================================================
# Tests for dataset.py - AvazuDataset
# =============================================================================
class TestAvazuDataset(unittest.TestCase):
    """Tests for AvazuDataset PyTorch dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Import AvazuDataset."""
        from dataset import AvazuDataset
        cls.AvazuDataset = AvazuDataset
    
    def test_dataset_with_labels(self):
        """Test dataset initialization with labels (training mode)."""
        X = np.random.randint(0, 100, (10, 5)).astype(np.int32)
        y = np.random.rand(10).astype(np.float32)
        
        dataset = self.AvazuDataset(X, y)
        
        self.assertEqual(len(dataset), 10)
        sample_x, sample_y = dataset[0]
        self.assertEqual(sample_x.shape, (5,))
        self.assertEqual(sample_y.shape, ())
    
    def test_dataset_without_labels(self):
        """Test dataset initialization without labels (inference mode)."""
        X = np.random.randint(0, 100, (10, 5)).astype(np.int32)
        
        dataset = self.AvazuDataset(X)
        
        self.assertEqual(len(dataset), 10)
        sample_x = dataset[0]
        assert isinstance(sample_x, torch.Tensor), "Inference mode should return a Tensor"
        self.assertEqual(sample_x.shape, (5,))
        self.assertIsInstance(sample_x, torch.Tensor)
    
    def test_dataset_tensor_types(self):
        """Verify correct tensor dtypes."""
        X = np.random.randint(0, 100, (5, 3)).astype(np.int32)
        y = np.random.rand(5).astype(np.float32)
        
        dataset = self.AvazuDataset(X, y)
        sample_x, sample_y = dataset[0]
        
        self.assertEqual(sample_x.dtype, torch.long)
        self.assertEqual(sample_y.dtype, torch.float32)
    
    def test_dataset_getitem_range(self):
        """Test that all indices are accessible."""
        X = np.random.randint(0, 100, (20, 4)).astype(np.int32)
        y = np.random.rand(20).astype(np.float32)
        
        dataset = self.AvazuDataset(X, y)
        
        for i in range(len(dataset)):
            sample = dataset[i]
            self.assertIsNotNone(sample)
    
    def test_dataset_works_with_dataloader(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        X = np.random.randint(0, 100, (32, 5)).astype(np.int32)
        y = np.random.rand(32).astype(np.float32)
        
        dataset = self.AvazuDataset(X, y)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for batch_x, batch_y in loader:
            self.assertEqual(batch_x.shape[0], 8)
            self.assertEqual(batch_x.shape[1], 5)
            self.assertEqual(batch_y.shape[0], 8)
            break
    
    def test_dataset_inference_with_dataloader(self):
        """Test that inference dataset works with DataLoader."""
        from torch.utils.data import DataLoader
        
        X = np.random.randint(0, 100, (16, 5)).astype(np.int32)
        
        dataset = self.AvazuDataset(X)
        loader = DataLoader(dataset, batch_size=4)
        
        for batch_x in loader:
            self.assertEqual(batch_x.shape[0], 4)
            self.assertEqual(batch_x.shape[1], 5)
            break


# =============================================================================
# Tests for config.py
# =============================================================================
class TestConfigExtended(unittest.TestCase):
    """Extended tests for config validation and seed_everything."""
    
    def test_seed_everything_reproducibility(self):
        """Test that seed_everything produces reproducible results."""
        from config import seed_everything
        
        seed_everything(42)
        random1 = np.random.rand(5)
        torch_random1 = torch.rand(5)
        
        seed_everything(42)
        random2 = np.random.rand(5)
        torch_random2 = torch.rand(5)
        
        np.testing.assert_array_equal(random1, random2)
        self.assertTrue(torch.equal(torch_random1, torch_random2))
    
    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        from config import seed_everything
        
        seed_everything(42)
        random1 = np.random.rand(5)
        
        seed_everything(123)
        random2 = np.random.rand(5)
        
        self.assertFalse(np.array_equal(random1, random2))
    
    def test_config_embedding_dim_positive(self):
        """Verify embedding_dim is positive."""
        self.assertGreater(CONFIG['embedding_dim'], 0)
    
    def test_config_batch_size_positive(self):
        """Verify batch_size is positive."""
        self.assertGreater(CONFIG['batch_size'], 0)
    
    def test_config_epochs_positive(self):
        """Verify epochs is positive."""
        self.assertGreater(CONFIG['epochs'], 0)
    
    def test_config_lr_positive(self):
        """Verify learning rate is positive."""
        self.assertGreater(CONFIG['lr'], 0)
    
    def test_config_validation_split_valid(self):
        """Verify validation_split is in valid range (0, 1)."""
        self.assertGreater(CONFIG['validation_split'], 0)
        self.assertLess(CONFIG['validation_split'], 1)
    
    def test_config_senet_and_gating_mutual_exclusivity(self):
        """Verify SENET and feature gating are mutually exclusive in production config."""
        # This test checks production config doesn't violate the constraint
        if CONFIG['use_senet'] and CONFIG['use_feature_gating']:
            self.fail("Production config has both use_senet and use_feature_gating enabled")
    
    def test_config_mlp_hidden_dims_not_empty(self):
        """Verify MLP has at least one hidden layer."""
        self.assertGreater(len(CONFIG['mlp_hidden_dims']), 0)
    
    def test_config_embedding_dim_rules_sorted(self):
        """Verify embedding_dim_rules are sorted ascending by cardinality."""
        rules = CONFIG['embedding_dim_rules']
        cardinalities = [r[0] for r in rules]
        self.assertEqual(cardinalities, sorted(cardinalities))


# =============================================================================
# Tests for data_processor.py - Time Feature Expressions
# =============================================================================
class TestDataProcessorTimeFeatures(unittest.TestCase):
    """Tests for data_processor time feature extraction."""
    
    def test_time_feature_expressions_output(self):
        """Test that time feature expressions produce correct output."""
        import polars as pl
        from data_processor import get_time_feature_expressions
        
        # Create test data with known hour values (YYMMDDHH format)
        test_data = pl.DataFrame({
            'hour': ['14102100', '14102223', '14110105']  # 2014-10-21 00:00, 2014-10-22 23:00, 2014-11-01 05:00
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # Verify extracted values (year removed as it has zero variance)
        self.assertEqual(result['month'].to_list(), [10, 10, 11])
        self.assertEqual(result['day_of_month'].to_list(), [21, 22, 1])
        self.assertEqual(result['hour_of_day'].to_list(), [0, 23, 5])
    
    def test_time_feature_day_of_week(self):
        """Test day_of_week calculation."""
        import polars as pl
        from data_processor import get_time_feature_expressions
        
        # 2014-10-21 was a Tuesday (weekday = 1 in Polars, 0=Monday)
        test_data = pl.DataFrame({
            'hour': ['14102100']
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # Tuesday = 2 (1-indexed in Polars dt.weekday())
        self.assertEqual(result['day_of_week'].to_list()[0], 2)
    
    def test_time_features_types(self):
        """Verify time features have correct types."""
        import polars as pl
        from data_processor import get_time_feature_expressions
        
        test_data = pl.DataFrame({
            'hour': ['14102100']
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # year removed as it has zero variance in the dataset
        self.assertEqual(result['month'].dtype, pl.UInt8)
        self.assertEqual(result['day_of_month'].dtype, pl.UInt8)
        self.assertEqual(result['hour_of_day'].dtype, pl.UInt8)
        self.assertEqual(result['day_of_week'].dtype, pl.UInt8)


class TestDataProcessorVocabulary(unittest.TestCase):
    """Tests for vocabulary building functions."""
    
    def test_build_vocabularies_basic(self):
        """Test basic vocabulary building."""
        import polars as pl
        from data_processor import build_vocabularies
        
        # Create test data
        test_data = pl.DataFrame({
            'cat1': ['a', 'b', 'a', 'c', 'a', 'b', 'a'],  # a=4, b=2, c=1
            'cat2': ['x', 'x', 'y', 'y', 'z', 'z', 'z'],  # x=2, y=2, z=3
        })
        
        vocab_sizes, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat1', 'cat2'], min_freq=2
        )
        
        # cat1: a and b pass min_freq=2, c doesn't -> size = 2 + 1 (UNK) = 3
        self.assertEqual(vocab_sizes['cat1'], 3)
        # cat2: all pass min_freq=2 -> size = 3 + 1 (UNK) = 4
        self.assertEqual(vocab_sizes['cat2'], 4)
        
        # Verify mappings exist
        self.assertIn('a', feat_maps['cat1'])
        self.assertIn('b', feat_maps['cat1'])
        self.assertNotIn('c', feat_maps['cat1'])  # Filtered by min_freq
    
    def test_build_vocabularies_min_freq_filtering(self):
        """Test that min_freq correctly filters low-frequency values."""
        import polars as pl
        from data_processor import build_vocabularies
        
        test_data = pl.DataFrame({
            'cat': ['a'] * 10 + ['b'] * 5 + ['c'] * 2 + ['d'] * 1
        })
        
        vocab_sizes, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat'], min_freq=5
        )
        
        # Only 'a' (10) and 'b' (5) should pass min_freq=5
        self.assertEqual(vocab_sizes['cat'], 3)  # a, b + UNK
        self.assertIn('a', feat_maps['cat'])
        self.assertIn('b', feat_maps['cat'])
        self.assertNotIn('c', feat_maps['cat'])
        self.assertNotIn('d', feat_maps['cat'])
    
    def test_build_vocabularies_mapping_starts_at_one(self):
        """Verify vocabulary indices start at 1 (0 reserved for UNK)."""
        import polars as pl
        from data_processor import build_vocabularies
        
        test_data = pl.DataFrame({
            'cat': ['x', 'y', 'z'] * 5
        })
        
        _, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat'], min_freq=1
        )
        
        # All indices should be >= 1
        for val, idx in feat_maps['cat'].items():
            self.assertGreaterEqual(idx, 1)
        
        # Index 0 should not be used (reserved for UNK)
        self.assertNotIn(0, feat_maps['cat'].values())


class TestDataProcessorMapping(unittest.TestCase):
    """Tests for mapping expression creation."""
    
    def test_create_mapping_expressions(self):
        """Test that mapping expressions correctly transform values."""
        import polars as pl
        from data_processor import create_mapping_expressions
        
        # Create a simple mapping
        feat_maps = {
            'cat1': {'a': 1, 'b': 2, 'c': 3}
        }
        
        test_data = pl.DataFrame({
            'cat1': ['a', 'b', 'c', 'unknown']  # 'unknown' should map to 0
        })
        
        mapping_exprs = create_mapping_expressions(feat_maps, ['cat1'])
        result = test_data.with_columns(mapping_exprs)
        
        expected = [1, 2, 3, 0]  # 'unknown' -> 0 (UNK)
        self.assertEqual(result['cat1'].to_list(), expected)
    
    def test_create_mapping_expressions_multiple_columns(self):
        """Test mapping with multiple categorical columns."""
        import polars as pl
        from data_processor import create_mapping_expressions
        
        feat_maps = {
            'cat1': {'x': 1, 'y': 2},
            'cat2': {'p': 1, 'q': 2, 'r': 3}
        }
        
        test_data = pl.DataFrame({
            'cat1': ['x', 'y', 'x'],
            'cat2': ['p', 'q', 'r']
        })
        
        mapping_exprs = create_mapping_expressions(feat_maps, ['cat1', 'cat2'])
        result = test_data.with_columns(mapping_exprs)
        
        self.assertEqual(result['cat1'].to_list(), [1, 2, 1])
        self.assertEqual(result['cat2'].to_list(), [1, 2, 3])


# =============================================================================
# Tests for data_processor.py - User Proxy Feature
# =============================================================================
class TestUserProxyFeature(unittest.TestCase):
    """Tests for user proxy feature (device_ip + device_model)."""
    
    def test_user_proxy_expression_creates_combined_id(self):
        """Test that user proxy correctly combines device_ip and device_model."""
        import polars as pl
        from data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '10.0.0.1', '192.168.1.1'],
            'device_model': ['iPhone12', 'Galaxy_S21', 'iPhone12']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        expected = ['192.168.1.1_iPhone12', '10.0.0.1_Galaxy_S21', '192.168.1.1_iPhone12']
        self.assertEqual(result['user_proxy'].to_list(), expected)
    
    def test_user_proxy_same_ip_different_model(self):
        """Test that same IP with different models creates different user proxies."""
        import polars as pl
        from data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '192.168.1.1'],
            'device_model': ['iPhone12', 'iPhone13']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertNotEqual(proxies[0], proxies[1])
        self.assertEqual(proxies[0], '192.168.1.1_iPhone12')
        self.assertEqual(proxies[1], '192.168.1.1_iPhone13')
    
    def test_user_proxy_different_ip_same_model(self):
        """Test that different IPs with same model creates different user proxies."""
        import polars as pl
        from data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'device_model': ['iPhone12', 'iPhone12']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertNotEqual(proxies[0], proxies[1])
    
    def test_user_proxy_with_empty_values(self):
        """Test user proxy handles empty/null values gracefully."""
        import polars as pl
        from data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '', 'null'],
            'device_model': ['iPhone12', 'Galaxy', '']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        # Should still produce valid strings (even if containing empty parts)
        proxies = result['user_proxy'].to_list()
        self.assertEqual(len(proxies), 3)
        self.assertEqual(proxies[0], '192.168.1.1_iPhone12')
        self.assertEqual(proxies[1], '_Galaxy')
        self.assertEqual(proxies[2], 'null_')
    
    def test_user_proxy_with_special_characters(self):
        """Test user proxy handles special characters in values."""
        import polars as pl
        from data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1'],
            'device_model': ['iPhone-12_Pro Max']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertEqual(proxies[0], '192.168.1.1_iPhone-12_Pro Max')


# =============================================================================
# Tests for data_processor.py - Interaction Features
# =============================================================================
class TestInteractionFeatures(unittest.TestCase):
    """Tests for interaction feature creation (device_id_x_app_id, device_ip_x_C14)."""
    
    def test_interaction_expressions_create_correct_columns(self):
        """Test that interaction expressions create the expected columns."""
        import polars as pl
        from data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002'],
            'app_id': ['app_A', 'app_B'],
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'C14': ['14001', '14002']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        # Check columns exist
        self.assertIn('device_id_x_app_id', result.columns)
        self.assertIn('device_ip_x_C14', result.columns)
    
    def test_interaction_device_id_app_id(self):
        """Test device_id x app_id interaction feature values."""
        import polars as pl
        from data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002', 'dev_001'],
            'app_id': ['app_A', 'app_B', 'app_B'],
            'device_ip': ['ip1', 'ip2', 'ip3'],
            'C14': ['c1', 'c2', 'c3']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        expected = ['dev_001_app_A', 'dev_002_app_B', 'dev_001_app_B']
        self.assertEqual(result['device_id_x_app_id'].to_list(), expected)
    
    def test_interaction_device_ip_c14(self):
        """Test device_ip x C14 interaction feature values."""
        import polars as pl
        from data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002'],
            'app_id': ['app_A', 'app_B'],
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'C14': ['14001', '14002']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        expected = ['192.168.1.1_14001', '10.0.0.1_14002']
        self.assertEqual(result['device_ip_x_C14'].to_list(), expected)
    
    def test_interaction_uniqueness(self):
        """Test that different input combinations produce different interaction values."""
        import polars as pl
        from data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_001', 'dev_002', 'dev_002'],
            'app_id': ['app_A', 'app_B', 'app_A', 'app_B'],
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip1'],
            'C14': ['c1', 'c1', 'c1', 'c1']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        interactions = result['device_id_x_app_id'].to_list()
        # All 4 combinations should be unique
        self.assertEqual(len(set(interactions)), 4)


# =============================================================================
# Tests for data_processor.py - Count Features
# =============================================================================
class TestCountFeatures(unittest.TestCase):
    """Tests for count/frequency feature computation."""
    
    def test_compute_count_features_basic(self):
        """Test basic count feature computation."""
        import polars as pl
        from data_processor import compute_count_features_from_train
        
        # Create train data with known frequencies
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip2', 'ip2', 'ip3']  # ip1=3, ip2=2, ip3=1
        }).lazy()
        
        # Test data with same and new values
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip4']  # ip4 is new (count=0)
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        train_result = lf_train.collect()
        test_result = lf_test.collect()
        
        # Verify train counts
        self.assertIn('device_ip_count', train_result.columns)
        train_counts = train_result['device_ip_count'].to_list()
        self.assertEqual(train_counts, [3, 3, 3, 2, 2, 1])
        
        # Verify test counts (based on train frequencies)
        test_counts = test_result['device_ip_count'].to_list()
        self.assertEqual(test_counts, [3, 2, 0])  # ip4 has count 0 (not in train)
    
    def test_compute_count_features_multiple_columns(self):
        """Test count features for multiple columns."""
        import polars as pl
        from data_processor import compute_count_features_from_train
        
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2'],
            'C14': ['c1', 'c2', 'c1']  # c1=2, c2=1
        }).lazy()
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip3'],
            'C14': ['c1', 'c2']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip', 'C14']
        )
        
        test_result = lf_test.collect()
        
        self.assertIn('device_ip_count', test_result.columns)
        self.assertIn('C14_count', test_result.columns)
        
        self.assertEqual(test_result['device_ip_count'].to_list(), [2, 0])
        self.assertEqual(test_result['C14_count'].to_list(), [2, 1])
    
    def test_compute_count_features_no_data_leakage(self):
        """Test that count features don't leak test data into training stats."""
        import polars as pl
        from data_processor import compute_count_features_from_train
        
        # Train has ip1 only
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1']
        }).lazy()
        
        # Test has ip2 only (not in train)
        test_data = pl.DataFrame({
            'device_ip': ['ip2', 'ip2', 'ip2']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        test_result = lf_test.collect()
        
        # ip2 should have count 0 (not in train), not 3 (from test)
        test_counts = test_result['device_ip_count'].to_list()
        self.assertEqual(test_counts, [0, 0, 0])
    
    def test_count_features_dtype(self):
        """Verify count features have correct data type (UInt32)."""
        import polars as pl
        from data_processor import compute_count_features_from_train
        
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2']
        }).lazy()
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        train_result = lf_train.collect()
        self.assertEqual(train_result['device_ip_count'].dtype, pl.UInt32)


# =============================================================================
# Tests for data_processor.py - Count Binning
# =============================================================================
class TestCountBinning(unittest.TestCase):
    """Tests for count feature binning."""
    
    def test_bin_count_features_basic(self):
        """Test basic count binning."""
        import polars as pl
        from data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 1, 3, 7, 25, 75, 250, 750, 2000]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        expected_bins = ['0', '1', '2-5', '6-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
        self.assertEqual(result['device_ip_count_bin'].to_list(), expected_bins)
    
    def test_bin_count_features_boundary_values(self):
        """Test binning at exact boundary values."""
        import polars as pl
        from data_processor import bin_count_features
        
        # Test exact boundary values
        test_data = pl.DataFrame({
            'device_ip_count': [0, 1, 5, 6, 10, 11, 50, 51, 100, 101, 500, 501, 1000, 1001]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_count_bin'].to_list()
        
        # Verify boundaries
        self.assertEqual(bins[0], '0')       # 0
        self.assertEqual(bins[1], '1')       # 1
        self.assertEqual(bins[2], '2-5')     # 5 (upper bound of 2-5)
        self.assertEqual(bins[3], '6-10')    # 6 (lower edge of 6-10)
        self.assertEqual(bins[4], '6-10')    # 10 (upper bound of 6-10)
        self.assertEqual(bins[5], '11-50')   # 11 (lower edge)
        self.assertEqual(bins[6], '11-50')   # 50 (upper bound)
        self.assertEqual(bins[7], '51-100')  # 51
        self.assertEqual(bins[8], '51-100')  # 100
        self.assertEqual(bins[9], '101-500') # 101
        self.assertEqual(bins[10], '101-500') # 500
        self.assertEqual(bins[11], '501-1000') # 501
        self.assertEqual(bins[12], '501-1000') # 1000
        self.assertEqual(bins[13], '1000+')    # 1001
    
    def test_bin_count_features_multiple_columns(self):
        """Test binning for multiple columns."""
        import polars as pl
        from data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 100, 5000],
            'C14_count': [1, 50, 1500]
        })
        
        bin_exprs = bin_count_features(['device_ip', 'C14'])
        result = test_data.with_columns(bin_exprs)
        
        self.assertIn('device_ip_count_bin', result.columns)
        self.assertIn('C14_count_bin', result.columns)
        
        self.assertEqual(result['device_ip_count_bin'].to_list(), ['0', '51-100', '1000+'])
        self.assertEqual(result['C14_count_bin'].to_list(), ['1', '11-50', '1000+'])
    
    def test_bin_count_features_all_same_bin(self):
        """Test when all values fall in the same bin."""
        import polars as pl
        from data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [3, 4, 5, 2, 3]  # All in '2-5' bin
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_count_bin'].to_list()
        self.assertTrue(all(b == '2-5' for b in bins))
    
    def test_bin_count_features_output_type(self):
        """Verify binned features are string type (for categorical encoding)."""
        import polars as pl
        from data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 100, 5000]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        # Binned columns should be strings (for later label encoding)
        self.assertEqual(result['device_ip_count_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Cumulative Count Features
# =============================================================================
class TestCumulativeCountFeatures(unittest.TestCase):
    """Tests for cumulative count feature computation."""
    
    def test_cumulative_count_basic(self):
        """Test basic cumulative count computation."""
        import polars as pl
        from data_processor import get_cumulative_count_expressions
        
        # Create test data with repeated values
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip1', 'ip1', 'ip2']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # ip1 appears at positions 0, 2, 3 -> cumcount should be 1, 2, 3
        # ip2 appears at positions 1, 4 -> cumcount should be 1, 2
        expected = [1, 1, 2, 3, 2]
        self.assertEqual(result['device_ip_cumcount'].to_list(), expected)
    
    def test_cumulative_count_multiple_columns(self):
        """Test cumulative count for multiple columns."""
        import polars as pl
        from data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2'],
            'user_proxy': ['u1', 'u2', 'u1']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip', 'user_proxy'])
        result = test_data.with_columns(cumcount_exprs)
        
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 2, 1])
        self.assertEqual(result['user_proxy_cumcount'].to_list(), [1, 1, 2])
    
    def test_cumulative_count_all_unique(self):
        """Test cumulative count when all values are unique."""
        import polars as pl
        from data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip3', 'ip4']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # All unique values should have cumcount of 1
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 1, 1, 1])
    
    def test_cumulative_count_all_same(self):
        """Test cumulative count when all values are the same."""
        import polars as pl
        from data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip1']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # All same values should have increasing cumcount
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 2, 3, 4])


class TestCumulativeCountBinning(unittest.TestCase):
    """Tests for cumulative count binning."""
    
    def test_bin_cumcount_basic(self):
        """Test basic cumulative count binning."""
        import polars as pl
        from data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 2, 5, 15, 75, 150]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        expected = ['first', '2-3', '4-10', '11-50', '51-100', '100+']
        self.assertEqual(result['device_ip_cumcount_bin'].to_list(), expected)
    
    def test_bin_cumcount_boundary_values(self):
        """Test binning at exact boundaries."""
        import polars as pl
        from data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 3, 4, 10, 11, 50, 51, 100, 101]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_cumcount_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 1
        self.assertEqual(bins[1], '2-3')     # 3
        self.assertEqual(bins[2], '4-10')    # 4
        self.assertEqual(bins[3], '4-10')    # 10
        self.assertEqual(bins[4], '11-50')   # 11
        self.assertEqual(bins[5], '11-50')   # 50
        self.assertEqual(bins[6], '51-100')  # 51
        self.assertEqual(bins[7], '51-100')  # 100
        self.assertEqual(bins[8], '100+')    # 101
    
    def test_bin_cumcount_output_type(self):
        """Verify binned features are string type."""
        import polars as pl
        from data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 50, 200]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        self.assertEqual(result['device_ip_cumcount_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Hourly Aggregated Features
# =============================================================================
class TestHourlyAggregatedFeatures(unittest.TestCase):
    """Tests for hourly aggregated feature computation."""
    
    def test_compute_hourly_features_basic(self):
        """Test basic hourly aggregated feature computation."""
        import polars as pl
        from data_processor import compute_hourly_aggregated_features
        
        # Create train data with known user-hour patterns
        train_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1', 'u2', 'u2'],
            'hour': ['14102100', '14102100', '14102101', '14102100', '14102100']
        }).lazy()
        
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u3'],
            'hour': ['14102100', '14102100', '14102100']
        }).lazy()
        
        lf_train, lf_test = compute_hourly_aggregated_features(train_data, test_data)
        
        train_result = lf_train.collect()
        test_result = lf_test.collect()
        
        # u1 in hour 14102100 appears 2 times in train
        # u2 in hour 14102100 appears 2 times in train
        # u1 in hour 14102101 appears 1 time in train
        self.assertIn('user_hourly_impressions', train_result.columns)
        
        # Test data: u1@14102100 -> 2, u2@14102100 -> 2, u3@14102100 -> 1 (unknown)
        test_impressions = test_result['user_hourly_impressions'].to_list()
        self.assertEqual(test_impressions[0], 2)  # u1@14102100
        self.assertEqual(test_impressions[1], 2)  # u2@14102100
        self.assertEqual(test_impressions[2], 1)  # u3@14102100 (not in train, defaults to 1)
    
    def test_hourly_features_no_leakage(self):
        """Test that hourly features don't leak test data."""
        import polars as pl
        from data_processor import compute_hourly_aggregated_features
        
        train_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1'],
            'hour': ['14102100', '14102100']
        }).lazy()
        
        # Test has u2 who's not in train
        test_data = pl.DataFrame({
            'user_proxy': ['u2', 'u2', 'u2'],
            'hour': ['14102100', '14102100', '14102100']
        }).lazy()
        
        _, lf_test = compute_hourly_aggregated_features(train_data, test_data)
        test_result = lf_test.collect()
        
        # u2 should have count 1 (default), not 3 (from test)
        self.assertTrue(all(c == 1 for c in test_result['user_hourly_impressions'].to_list()))


class TestHourlyImpressionsBinning(unittest.TestCase):
    """Tests for hourly impressions binning."""
    
    def test_bin_hourly_impressions_basic(self):
        """Test basic hourly impressions binning.
        
        EDA-optimized bins:
        - 'single' (1): Most common
        - '2' (2): Returning within hour
        - '3-4' (3-4): Up to P90
        - '5+': High-frequency users
        """
        import polars as pl
        from data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 2, 3, 4, 5]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        expected = ['single', '2', '3-4', '3-4', '5+']
        self.assertEqual(result['user_hourly_impressions_bin'].to_list(), expected)
    
    def test_bin_hourly_impressions_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 2, 3, 4, 5, 10]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_hourly_impressions_bin'].to_list()
        self.assertEqual(bins[0], 'single')  # 1
        self.assertEqual(bins[1], '2')       # 2
        self.assertEqual(bins[2], '3-4')     # 3
        self.assertEqual(bins[3], '3-4')     # 4
        self.assertEqual(bins[4], '5+')      # 5
        self.assertEqual(bins[5], '5+')      # 10
    
    def test_bin_hourly_impressions_output_type(self):
        """Verify binned feature is string type."""
        import polars as pl
        from data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 10, 100]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        self.assertEqual(result['user_hourly_impressions_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Time-Delta Features
# =============================================================================
class TestTimeDeltaFeatures(unittest.TestCase):
    """Tests for time-delta feature computation (hours since last click)."""
    
    def test_compute_time_delta_basic(self):
        """Test basic time delta computation."""
        import polars as pl
        from data_processor import compute_time_delta_features
        
        # Create test data with known time sequence for same user
        # user u1 clicks at hours 00, 01, 05 -> deltas should be 0, 1, 4
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1'],
            'hour': ['14102100', '14102101', '14102105']
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        
        self.assertIn('hours_since_last_click', result.columns)
        deltas = result['hours_since_last_click'].to_list()
        self.assertEqual(deltas[0], 0)  # First click, no previous
        self.assertEqual(deltas[1], 1)  # 1 hour after first
        self.assertEqual(deltas[2], 4)  # 4 hours after second
    
    def test_compute_time_delta_multiple_users(self):
        """Test time delta computation with multiple users."""
        import polars as pl
        from data_processor import compute_time_delta_features
        
        # Two users with different click patterns
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u1', 'u2'],
            'hour': ['14102100', '14102100', '14102102', '14102110']
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        deltas = result['hours_since_last_click'].to_list()
        
        # u1: first click (0), then 2 hours later
        # u2: first click (0), then 10 hours later
        self.assertEqual(deltas[0], 0)   # u1 first
        self.assertEqual(deltas[1], 0)   # u2 first
        self.assertEqual(deltas[2], 2)   # u1 second, 2 hours after first
        self.assertEqual(deltas[3], 10)  # u2 second, 10 hours after first
    
    def test_compute_time_delta_across_days(self):
        """Test time delta computation across different days."""
        import polars as pl
        from data_processor import compute_time_delta_features
        
        # User clicks on different days
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1'],
            'hour': ['14102100', '14102200']  # Oct 21 00:00 -> Oct 22 00:00 = 24 hours
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        deltas = result['hours_since_last_click'].to_list()
        
        self.assertEqual(deltas[0], 0)   # First click
        self.assertEqual(deltas[1], 24)  # 24 hours later


class TestTimeDeltaBinning(unittest.TestCase):
    """Tests for time delta binning."""
    
    def test_bin_time_delta_basic(self):
        """Test basic time delta binning.
        
        EDA-optimized bins:
        - 'first': First click (0 hours)
        - '1-4h': Short interval (<=4)
        - '5-19h': Medium interval (<=19)
        - '20-53h': Long interval (<=53)
        - '>53h': Re-engagement
        """
        import polars as pl
        from data_processor import bin_time_delta_features
        
        test_data = pl.DataFrame({
            'hours_since_last_click': [0, 3, 15, 50, 100]
        })
        
        bin_expr = bin_time_delta_features()
        result = test_data.with_columns(bin_expr)
        
        bins = result['hours_since_last_click_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 0
        self.assertEqual(bins[1], '1-4h')    # 3 (<=4)
        self.assertEqual(bins[2], '5-19h')   # 15 (<=19)
        self.assertEqual(bins[3], '20-53h')  # 50 (<=53)
        self.assertEqual(bins[4], '>53h')    # 100 (>53)
    
    def test_bin_time_delta_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from data_processor import bin_time_delta_features
        
        test_data = pl.DataFrame({
            'hours_since_last_click': [0, 4, 5, 19, 20, 53, 54]
        })
        
        bin_expr = bin_time_delta_features()
        result = test_data.with_columns(bin_expr)
        
        bins = result['hours_since_last_click_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 0
        self.assertEqual(bins[1], '1-4h')    # 4 (boundary)
        self.assertEqual(bins[2], '5-19h')   # 5 (boundary)
        self.assertEqual(bins[3], '5-19h')   # 19 (boundary)
        self.assertEqual(bins[4], '20-53h')  # 20 (boundary)
        self.assertEqual(bins[5], '20-53h')  # 53 (boundary)
        self.assertEqual(bins[6], '>53h')    # 54


# =============================================================================
# Tests for data_processor.py - Previous Click Count Features
# =============================================================================
class TestPreviousClickCount(unittest.TestCase):
    """Tests for previous click count feature computation."""
    
    def test_compute_prev_click_count_basic(self):
        """Test basic previous click count computation."""
        import polars as pl
        from data_processor import compute_previous_click_count
        
        # User makes 4 clicks -> prev counts should be 0, 1, 2, 3
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1', 'u1']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        
        self.assertIn('user_proxy_prev_clicks', result.columns)
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        self.assertEqual(prev_clicks, [0, 1, 2, 3])
    
    def test_compute_prev_click_count_multiple_users(self):
        """Test previous click count with multiple users."""
        import polars as pl
        from data_processor import compute_previous_click_count
        
        # Two users with different click counts
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u1', 'u2', 'u1']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        
        # u1: 0, 1, 2 (positions 0, 2, 4)
        # u2: 0, 1 (positions 1, 3)
        self.assertEqual(prev_clicks[0], 0)  # u1 first
        self.assertEqual(prev_clicks[1], 0)  # u2 first
        self.assertEqual(prev_clicks[2], 1)  # u1 second
        self.assertEqual(prev_clicks[3], 1)  # u2 second
        self.assertEqual(prev_clicks[4], 2)  # u1 third
    
    def test_prev_click_count_all_unique(self):
        """Test previous click count when all users are unique."""
        import polars as pl
        from data_processor import compute_previous_click_count
        
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u3', 'u4']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        
        # All first-time users should have 0 previous clicks
        self.assertEqual(prev_clicks, [0, 0, 0, 0])


class TestPreviousClicksBinning(unittest.TestCase):
    """Tests for previous clicks binning."""
    
    def test_bin_prev_clicks_basic(self):
        """Test basic previous clicks binning.
        
        EDA-optimized bins:
        - 'new' (0): First-time user
        - 'returning' (1-7): Up to P50
        - 'regular' (8-32): P50 to P75
        - 'heavy' (33-224): P75 to P90
        - 'power' (>224): Top 10% most active
        """
        import polars as pl
        from data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 3, 15, 75, 250]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_proxy_prev_clicks_bin'].to_list()
        self.assertEqual(bins[0], 'new')       # 0
        self.assertEqual(bins[1], 'returning') # 3 (<=7)
        self.assertEqual(bins[2], 'regular')   # 15 (<=32)
        self.assertEqual(bins[3], 'heavy')     # 75 (<=224)
        self.assertEqual(bins[4], 'power')     # 250 (>224)
    
    def test_bin_prev_clicks_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 7, 8, 32, 33, 224, 225]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_proxy_prev_clicks_bin'].to_list()
        self.assertEqual(bins[0], 'new')       # 0
        self.assertEqual(bins[1], 'returning') # 7 (boundary)
        self.assertEqual(bins[2], 'regular')   # 8 (boundary)
        self.assertEqual(bins[3], 'regular')   # 32 (boundary)
        self.assertEqual(bins[4], 'heavy')     # 33 (boundary)
        self.assertEqual(bins[5], 'heavy')     # 224 (boundary)
        self.assertEqual(bins[6], 'power')     # 225
    
    def test_bin_prev_clicks_output_type(self):
        """Verify binned feature is string type."""
        import polars as pl
        from data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 50, 200]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        self.assertEqual(result['user_proxy_prev_clicks_bin'].dtype, pl.String)





def list_tests():
    """Print all available tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    print("Available tests:")
    print("-" * 50)
    for test_group in suite:
        for test in iter(test_group):  # type: ignore[arg-type]
            print(f"  {test}")
    print("-" * 50)
    print("\nUsage examples:")
    print("  Run all:      python tests.py")
    print("  Run one:      python tests.py TestModelStructure.test_dcn_layers")
    print("  Run class:    python tests.py TestModelStructure")
    print("  List tests:   python tests.py --list")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_tests()
    else:
        # Run with verbosity by default for better output
        unittest.main(verbosity=2)

