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
from config import CONFIG, ConfigType
from model import GatedDCNModel, DCNv2, SENetLayer


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
        
        # Model Architecture
        'embedding_dim': 16,
        'use_dcn': True,
        'dcn_num_layers': 2,
        'dcn_use_layernorm': False,
        'dcn_low_rank': None,
        'use_senet': True,
        'senet_squeeze_funcs': ['mean'],
        'senet_reduction_ratio': 3,
        'senet_activation': 'sigmoid',
        'mlp_hidden_dims': [32, 16],
        'mlp_activation': 'relu',
        'use_batch_norm': True,
        
        # Training
        'lr': 1e-3,
        'embedding_lr': 1.0,
        'embedding_optimizer': 'adagrad',
        'epochs': 1,
        'lr_warmup_epoch_ratio': 0.1,
        'early_stopping_patience': 3,
        'use_tensorboard': False,
        'tensorboard_logdir': './runs',
        
        # Regularization
        'mlp_dropout': 0.1,
        'grad_clip': 1.0,
        'weight_decay': 1e-5,
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
        The exact structure depends on use_batch_norm config.
        """
        # Just verify MLP has more than 0 layers and ends with Linear(*, 1)
        self.assertGreater(len(self.model.mlp), 0, "MLP should have layers")
        
        # Check final layer outputs single value
        final_layer = self.model.mlp[-1]
        self.assertIsInstance(final_layer, nn.Linear, "Final layer should be Linear")
        self.assertEqual(final_layer.out_features, 1, "Final layer should output 1")
    
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
        config = make_test_config(use_batch_norm=False)
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
        config = make_test_config(use_dcn=False, use_senet=False, use_batch_norm=False)
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
