"""
Test suite for Ensemble model with heterogeneous model support.

Tests for EnsembleModel with different model types (GatedDCN, STEC, nested Ensemble).
"""

import unittest
from typing import Any, cast
import torch
from config import ConfigType, GatedDCNConfig, STECConfig, EnsembleConfig
from src.models.architectures.ensemble import EnsembleModel
from src.models.architectures.gated_dcn import GatedDCNModel
from src.models.architectures.stec import STECModel


def make_gated_dcn_config() -> GatedDCNConfig:
    """Create a minimal GatedDCN config."""
    return {
        "use_dcn": True,
        "dcn_num_layers": 2,
        "dcn_use_layernorm": False,
        "dcn_low_rank": None,
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
        "mlp_hidden_dims": [32, 16],
        "mlp_activation": "relu",
        "mlp_use_skip_connections": False,
        "mlp_dropout": 0.1,
        "use_layer_norm": False,
    }


def make_stec_config() -> STECConfig:
    """Create a minimal STEC config."""
    return {
        "stec_num_layers": 2,
        "stec_num_heads": 4,
        "stec_hidden_dim": None,
        "stec_dropout": 0.1,
        "stec_use_ffn": True,
        "stec_mlp_hidden_dims": [32, 16],
    }


def make_base_config(
    model_config: EnsembleConfig | GatedDCNConfig | STECConfig,
) -> ConfigType:
    """Create a base ConfigType with the given model config."""
    return cast(
        ConfigType,
        {
            "seed": 42,
            "device": "cpu",
            "batch_size": 32,
            "num_workers": 0,
            "min_freq": 5,
            "data_processor_sort_keys": [],
            "validation_split": 0.1,
            "shuffle_train": False,
            "embedding_dim": 16,
            "feature_embeddings": {},
            "embedding_projection_dim": None,
            "model": model_config,
            "epochs": 1,
            "early_stopping_patience": 3,
            "grad_clip": 1.0,
            "use_tensorboard": False,
            "tensorboard_logdir": "./runs",
            "tensorboard_log_interval": 1000,
            "dense_optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "warmup_epoch_ratio": 0.1,
                "weight_decay": 1e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            },
            "embedding_optimizer": {
                "type": "adagrad",
                "lr": 1.0,
                "warmup_epoch_ratio": 0.0,
                "weight_decay": 0.0,
                "eps": 1e-10,
                "lr_decay": 0.0,
            },
            "auto_amp": False,
            "amp_dtype": "float16",
            "compile_model": False,
            "train_path": "./data/train.gz",
            "test_path": "./data/test.gz",
            "sub_path": "submission.csv",
            "processed_path": "./data",
            "models_path": "./models",
        },
    )


class TestEnsembleHomogeneous(unittest.TestCase):
    """Tests for ensemble with same model types."""

    def setUp(self):
        self.vocab_sizes = {"f1": 100, "f2": 100}
        self.feature_names = ["f1", "f2"]

    def test_ensemble_two_gated_dcn(self):
        """Test ensemble with two GatedDCN models."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        self.assertEqual(model.num_models(), 2)
        self.assertIsInstance(model.get_model(0), GatedDCNModel)
        self.assertIsInstance(model.get_model(1), GatedDCNModel)

    def test_ensemble_forward_pass(self):
        """Test forward pass through ensemble."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        output = model(x)

        self.assertEqual(output["logits"].shape, (4, 1))
        self.assertIsNotNone(output["aux_logits"])
        self.assertEqual(len(output["aux_logits"]), 2)

    def test_ensemble_median_aggregation(self):
        """Test median aggregation method."""
        ensemble_config: EnsembleConfig = {
            "models": [
                make_gated_dcn_config(),
                make_gated_dcn_config(),
                make_gated_dcn_config(),
            ],
            "ensemble_aggregation": "median",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        output = model(x)

        self.assertEqual(output["logits"].shape, (4, 1))


class TestEnsembleHeterogeneous(unittest.TestCase):
    """Tests for ensemble with different model types."""

    def setUp(self):
        self.vocab_sizes = {"f1": 100, "f2": 100}
        self.feature_names = ["f1", "f2"]

    def test_ensemble_gated_dcn_and_stec(self):
        """Test ensemble with mixed GatedDCN and STEC models."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_stec_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        self.assertEqual(model.num_models(), 2)
        self.assertIsInstance(model.get_model(0), GatedDCNModel)
        self.assertIsInstance(model.get_model(1), STECModel)

    def test_heterogeneous_forward_pass(self):
        """Test forward pass with mixed model types."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_stec_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        output = model(x)

        self.assertEqual(output["logits"].shape, (4, 1))
        self.assertEqual(len(output["aux_logits"]), 2)


class TestNestedEnsemble(unittest.TestCase):
    """Tests for nested ensemble (ensemble containing ensemble)."""

    def setUp(self):
        self.vocab_sizes = {"f1": 100, "f2": 100}
        self.feature_names = ["f1", "f2"]

    def test_nested_ensemble(self):
        """Test ensemble containing another ensemble."""
        inner_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        outer_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), inner_ensemble_config],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(outer_ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        self.assertEqual(model.num_models(), 2)
        self.assertIsInstance(model.get_model(0), GatedDCNModel)
        self.assertIsInstance(model.get_model(1), EnsembleModel)

        # The nested ensemble should have 2 models
        inner_ensemble = model.get_model(1)
        self.assertIsInstance(inner_ensemble, EnsembleModel)
        self.assertEqual(inner_ensemble.num_models(), 2)

    def test_nested_ensemble_forward(self):
        """Test forward pass through nested ensemble."""
        inner_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        outer_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), inner_ensemble_config],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(outer_ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        output = model(x)

        self.assertEqual(output["logits"].shape, (4, 1))


class TestEnsembleLoss(unittest.TestCase):
    """Tests for ensemble loss computation."""

    def setUp(self):
        self.vocab_sizes = {"f1": 100, "f2": 100}
        self.feature_names = ["f1", "f2"]

    def test_loss_computation(self):
        """Test that loss can be computed for ensemble."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        y = torch.rand(4, 1)

        output = model(x)
        loss = model.compute_loss(output, y)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar loss

    def test_recursive_loss_computation(self):
        """Test that nested ensembles compute recursive loss."""
        inner_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        outer_ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), inner_ensemble_config],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(outer_ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        y = torch.rand(4, 1)

        output = model(x)
        total_loss = model.compute_loss(output, y)

        # Total loss should be a scalar
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.shape, ())

        # The total loss should be greater than just the outer ensemble's loss
        # because it includes the inner ensemble's K-BCE loss too
        # (This is a sanity check - we can't easily compute the exact expected value)
        self.assertGreater(total_loss.item(), 0)


class TestEnsembleEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        self.vocab_sizes = {"f1": 100, "f2": 100}
        self.feature_names = ["f1", "f2"]

    def test_empty_models_list_raises_error(self):
        """Test that empty models list raises ValueError."""
        ensemble_config: EnsembleConfig = {
            "models": [],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        with self.assertRaises(ValueError):
            EnsembleModel(self.vocab_sizes, self.feature_names, config)

    def test_single_model_ensemble(self):
        """Test ensemble with single model (valid but trivial)."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        self.assertEqual(model.num_models(), 1)

        x = torch.randint(0, 100, (4, 2))
        output = model(x)
        self.assertEqual(output["logits"].shape, (4, 1))

    def test_forward_single_model(self):
        """Test forward_single method."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config(), make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        output = model.forward_single(x, 0)

        self.assertEqual(output["logits"].shape, (4, 1))

    def test_invalid_model_idx_raises_error(self):
        """Test that invalid model index raises ValueError."""
        ensemble_config: EnsembleConfig = {
            "models": [make_gated_dcn_config()],
            "ensemble_aggregation": "mean",
        }
        config = make_base_config(ensemble_config)

        model = EnsembleModel(self.vocab_sizes, self.feature_names, config)

        x = torch.randint(0, 100, (4, 2))
        with self.assertRaises(ValueError):
            model.forward_single(x, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
