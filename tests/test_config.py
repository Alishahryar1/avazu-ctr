"""
Test suite for configuration validation.

This module tests configuration validity, value types, ranges,
and the seed_everything function for reproducibility.
"""

import unittest
import numpy as np
import torch
from src.config.config import CONFIG


class TestConfig(unittest.TestCase):
    """Tests for configuration validity."""

    def test_required_keys_exist(self):
        """Verify all required config keys are present."""
        # Top-level keys
        top_level_keys = ['embedding_dim', 'lr', 'batch_size', 'epochs']
        for key in top_level_keys:
            self.assertIn(key, CONFIG, f"Missing required top-level config key: {key}")
        
        # Model-level keys (nested under 'model')
        self.assertIn('model', CONFIG, "Missing 'model' key in config")
        model_keys = ['dcn_num_layers', 'mlp_hidden_dims', 'mlp_dropout']
        model_config = CONFIG['model']
        for key in model_keys:
            self.assertIn(key, model_config, f"Missing required model config key: {key}")

    def test_config_value_types(self):
        """Verify config values have correct types."""
        model_config = CONFIG['model']
        self.assertIsInstance(CONFIG['embedding_dim'], int)
        self.assertIsInstance(model_config['dcn_num_layers'], int)
        self.assertIsInstance(model_config['mlp_hidden_dims'], (list, tuple))
        self.assertIsInstance(model_config['mlp_dropout'], float)

    def test_config_value_ranges(self):
        """Verify config values are in valid ranges."""
        model_config = CONFIG['model']
        self.assertGreater(CONFIG['embedding_dim'], 0, "embedding_dim must be positive")
        self.assertGreater(model_config['dcn_num_layers'], 0, "dcn_num_layers must be positive")
        self.assertGreaterEqual(model_config['mlp_dropout'], 0.0, "mlp_dropout must be >= 0")
        self.assertLess(model_config['mlp_dropout'], 1.0, "mlp_dropout must be < 1")
        if model_config['dcn_low_rank'] is not None:
            self.assertGreater(model_config['dcn_low_rank'], 0, "dcn_low_rank must be positive if set")


class TestConfigExtended(unittest.TestCase):
    """Extended tests for config validation and seed_everything."""

    def test_seed_everything_reproducibility(self):
        """Test that seed_everything produces reproducible results."""
        from src.config.config import seed_everything

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
        from src.config.config import seed_everything

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
        """Verify validation_split is in valid range [0, 1)."""
        self.assertGreaterEqual(CONFIG['validation_split'], 0)  # 0 means validation disabled
        self.assertLess(CONFIG['validation_split'], 1)

    def test_config_senet_and_gating_mutual_exclusivity(self):
        """Verify SENET and feature gating are mutually exclusive in production config."""
        model_config = CONFIG['model']
        # This test checks production config doesn't violate the constraint
        if model_config['use_senet'] and model_config['use_feature_gating']:
            self.fail("Production config has both use_senet and use_feature_gating enabled")

    def test_config_mlp_hidden_dims_not_empty(self):
        """Verify MLP has at least one hidden layer."""
        self.assertGreater(len(CONFIG['model']['mlp_hidden_dims']), 0)

    def test_config_feature_embeddings_valid(self):
        """Verify feature_embeddings has valid structure."""
        feature_embeddings = CONFIG['feature_embeddings']
        self.assertIsInstance(feature_embeddings, dict)
        for name, config in feature_embeddings.items():
            self.assertIn('type', config, f"feature {name} missing 'type'")
            self.assertIn(config['type'], ['standard', 'hash'], f"feature {name} has invalid type")
            self.assertIn('dim', config, f"feature {name} missing 'dim'")
            self.assertGreater(config['dim'], 0, f"feature {name} has invalid dim")


if __name__ == "__main__":
    unittest.main(verbosity=2)
