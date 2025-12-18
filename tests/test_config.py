"""
Test suite for configuration validation.

This module tests configuration validity, value types, ranges,
and the seed_everything function for reproducibility.
"""

import unittest
import numpy as np
import torch
from config import CONFIG


class TestConfig(unittest.TestCase):
    """Tests for configuration validity."""

    def test_required_keys_exist(self):
        """Verify all required config keys are present."""
        # Top-level keys
        top_level_keys = ['embedding_dim', 'batch_size', 'epochs', 'dense_optimizer', 'embedding_optimizer']
        for key in top_level_keys:
            self.assertIn(key, CONFIG, f"Missing required top-level config key: {key}")
        
        # Model key must exist
        self.assertIn('model', CONFIG, "Missing 'model' key in config")
        model_config = CONFIG['model']
        
        # Check for ensemble or single model structure
        if 'models' in model_config:
            # Ensemble config
            self.assertIn('ensemble_aggregation', model_config)
            self.assertIsInstance(model_config['models'], list)
            self.assertGreater(len(model_config['models']), 0, "Ensemble must have at least one model")
        else:
            # Single model config (GatedDCN or STEC)
            # Should have either 'use_dcn' (GatedDCN) or 'stec_num_layers' (STEC)
            has_model_keys = 'use_dcn' in model_config or 'stec_num_layers' in model_config
            self.assertTrue(has_model_keys, "Model config must have identifying keys")

    def test_config_value_types(self):
        """Verify top-level config values have correct types."""
        self.assertIsInstance(CONFIG['embedding_dim'], int)
        self.assertIsInstance(CONFIG['dense_optimizer'], dict)
        self.assertIsInstance(CONFIG['embedding_optimizer'], dict)
        self.assertIsInstance(CONFIG['batch_size'], int)
        self.assertIsInstance(CONFIG['epochs'], int)

    def test_config_value_ranges(self):
        """Verify config values are in valid ranges."""
        self.assertGreater(CONFIG['embedding_dim'], 0, "embedding_dim must be positive")
        # Check lr in optimizer configs
        dense_lr = CONFIG['dense_optimizer'].get('lr')
        if dense_lr is not None:
            self.assertGreater(dense_lr, 0, "dense optimizer lr must be positive")
        self.assertGreater(CONFIG['batch_size'], 0, "batch_size must be positive")
        self.assertGreater(CONFIG['epochs'], 0, "epochs must be positive")


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
        """Verify learning rates are positive."""
        dense_lr = CONFIG['dense_optimizer'].get('lr')
        if dense_lr is not None:
            self.assertGreater(dense_lr, 0)
        embed_lr = CONFIG['embedding_optimizer'].get('lr')
        if embed_lr is not None:
            self.assertGreater(embed_lr, 0)

    def test_config_validation_split_valid(self):
        """Verify validation_split is in valid range [0, 1)."""
        self.assertGreaterEqual(CONFIG['validation_split'], 0)  # 0 means validation disabled
        self.assertLess(CONFIG['validation_split'], 1)

    def test_config_model_structure(self):
        """Verify model config has valid structure."""
        model_config = CONFIG['model']
        
        # If ensemble, check aggregation method
        if 'models' in model_config:
            self.assertIn(model_config['ensemble_aggregation'], ['mean', 'median'])
        
    def test_config_senet_and_gating_mutual_exclusivity(self):
        """Verify SENET and feature gating are mutually exclusive in all GatedDCN configs."""
        def check_gated_dcn_config(cfg: dict):
            """Check single GatedDCN config for mutual exclusivity."""
            if 'use_senet' in cfg and 'use_feature_gating' in cfg:
                if cfg['use_senet'] and cfg['use_feature_gating']:
                    self.fail("Config has both use_senet and use_feature_gating enabled")
        
        def check_model_configs(cfg: dict):
            """Recursively check all model configs."""
            if 'models' in cfg:
                # Ensemble: check each sub-model
                for sub_cfg in cfg['models']:
                    check_model_configs(sub_cfg)
            elif 'use_dcn' in cfg:
                # GatedDCN config
                check_gated_dcn_config(cfg)
        
        check_model_configs(CONFIG['model'])

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



if __name__ == "__main__":
    unittest.main(verbosity=2)
