"""Tests for hyperparameter tuning script."""

import unittest
from copy import deepcopy
from typing import Any, cast

import optuna

from config import CONFIG


class TestCreateConfigFromTrial(unittest.TestCase):
    """Tests for create_config_from_trial function."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = cast(dict[str, Any], deepcopy(CONFIG))

    def test_config_not_mutated(self):
        """Test that base config is not mutated by trial sampling."""
        # Import here to avoid issues with module loading
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial

        original_lr = self.base_config["dense_optimizer"]["lr"]

        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()

        # Sample config
        new_config = create_config_from_trial(trial, self.base_config)

        # Original config should be unchanged
        self.assertEqual(
            self.base_config["dense_optimizer"]["lr"],
            original_lr,
            "Base config was mutated",
        )

        # New config should potentially have different values
        self.assertIsInstance(new_config, dict)

    def test_all_heads_configured(self):
        """Test that all 4 heads are configured in output."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial, NUM_HEADS

        study = optuna.create_study()
        trial = study.ask()

        config = create_config_from_trial(trial, self.base_config)

        self.assertEqual(len(config["model"]["heads"]), NUM_HEADS)

    def test_head_structure_valid(self):
        """Test that each head has all required keys."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial

        study = optuna.create_study()
        trial = study.ask()

        config = create_config_from_trial(trial, self.base_config)

        required_keys = [
            "hidden_dims",
            "activation",
            "dropout",
            "use_layer_norm",
            "use_skip_connections",
        ]

        for i, head in enumerate(config["model"]["heads"]):
            for key in required_keys:
                self.assertIn(key, head, f"Head {i} missing required key: {key}")

    def test_hyperparameter_ranges(self):
        """Test that sampled hyperparameters are within expected ranges."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial

        study = optuna.create_study()
        trial = study.ask()

        config = create_config_from_trial(trial, self.base_config)

        # Check learning rates are in range
        self.assertGreaterEqual(config["dense_optimizer"]["lr"], 1e-5)
        self.assertLessEqual(config["dense_optimizer"]["lr"], 1e-1)

        self.assertGreaterEqual(config["embedding_optimizer"]["lr"], 1e-3)
        self.assertLessEqual(config["embedding_optimizer"]["lr"], 1.0)

        # Check dropout in range
        self.assertGreaterEqual(config["model"]["backbone_config"]["mlp_dropout"], 0.0)
        self.assertLessEqual(config["model"]["backbone_config"]["mlp_dropout"], 0.5)

        # Check grad clip in range
        self.assertGreaterEqual(config["grad_clip"], 0.1)
        self.assertLessEqual(config["grad_clip"], 5.0)

        # Check DCN layers in range
        self.assertGreaterEqual(config["model"]["backbone_config"]["dcn_num_layers"], 2)
        self.assertLessEqual(config["model"]["backbone_config"]["dcn_num_layers"], 16)

    def test_aggregation_method_gated_sets_hidden_dim(self):
        """Test that gated aggregation sets gating_hidden_dim."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial

        # Run multiple trials to get both aggregation methods
        gated_found = False
        mean_found = False

        for _ in range(20):  # Should be enough to get both
            study = optuna.create_study()
            trial = study.ask()
            config = create_config_from_trial(trial, self.base_config)

            if config["model"]["aggregation_method"] == "gated":
                gated_found = True
                # gating_hidden_dim should be set (could be None or int)
                self.assertIn("gating_hidden_dim", config["model"])
            elif config["model"]["aggregation_method"] == "mean":
                mean_found = True
                # gating_hidden_dim should be None for mean
                self.assertIsNone(config["model"]["gating_hidden_dim"])

            if gated_found and mean_found:
                break

    def test_mlp_hidden_dims_structure(self):
        """Test that MLP hidden dims are properly structured."""
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from misc.tune_hyperparams import create_config_from_trial

        study = optuna.create_study()
        trial = study.ask()

        config = create_config_from_trial(trial, self.base_config)

        mlp_dims = config["model"]["backbone_config"]["mlp_hidden_dims"]

        # Should be a list
        self.assertIsInstance(mlp_dims, list)

        # Should have 1-6 layers
        self.assertGreaterEqual(len(mlp_dims), 1)
        self.assertLessEqual(len(mlp_dims), 6)

        # All dims should be the same (uniform width)
        if len(mlp_dims) > 1:
            self.assertEqual(len(set(mlp_dims)), 1)

        # Width should be in range
        self.assertGreaterEqual(mlp_dims[0], 128)
        self.assertLessEqual(mlp_dims[0], 4096)


class TestStudyCreation(unittest.TestCase):
    """Tests for Optuna study creation."""

    def test_study_creates_successfully(self):
        """Test that study can be created with correct settings."""
        study = optuna.create_study(
            study_name="test_study",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        self.assertEqual(study.study_name, "test_study")
        self.assertEqual(study.direction, optuna.study.StudyDirection.MINIMIZE)


if __name__ == "__main__":
    unittest.main()
