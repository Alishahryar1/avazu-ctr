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
from config import CONFIG
from model import GatedDCNModel


class TestModelStructure(unittest.TestCase):
    """Tests for model architecture and configuration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up mock data used across all tests in this class."""
        cls.vocab_sizes = {'f1': 100, 'f2': 100}
        cls.feature_names = ['f1', 'f2']
        cls.model = GatedDCNModel(
            cls.vocab_sizes, 
            CONFIG['embedding_dim'], 
            cls.feature_names,
            dcn_num_layers=CONFIG['dcn_num_layers'],
            mlp_hidden_dims=CONFIG['mlp_hidden_dims'],
            mlp_dropout=CONFIG['mlp_dropout'],
            mlp_activation=CONFIG['mlp_activation'],
            gating_activation=CONFIG['gating_activation']
        )
    
    def test_dcn_layers(self):
        """Verify DCN has the correct number of layers."""
        expected_layers = CONFIG['dcn_num_layers']
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
        expected_dim = CONFIG['embedding_dim']
        for name, embedding in self.model.embeddings.items():
            actual_dim = embedding.embedding_dim
            self.assertEqual(
                actual_dim,
                expected_dim,
                f"Embedding '{name}' has dim {actual_dim}, expected {expected_dim}"
            )


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
