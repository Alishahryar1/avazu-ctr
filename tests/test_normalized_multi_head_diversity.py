"""Tests for NormalizedMultiHeadDiversityModel and normalized layers."""

import pytest
import torch
import torch.nn as nn
import math

from src.models.layers.normalized_layers import (
    l2_normalize,
    NormalizedEmbedding,
    NormalizedLinear,
    NormalizedMLP,
    NormalizedResidualMLP,
    WeightNormalizationCallback,
)
from src.models.architectures.normalized_multi_head_diversity import (
    NormalizedMultiHeadDiversityModel,
)


class TestL2Normalize:
    """Tests for l2_normalize function."""

    def test_normalize_vectors(self):
        """Test that vectors are normalized to unit norm."""
        x = torch.randn(32, 64)
        x_norm = l2_normalize(x, dim=-1)

        norms = torch.norm(x_norm, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_normalize_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        x = torch.randn(32, 64)
        x_norm = l2_normalize(x, dim=-1)

        # Dot product should be positive (same direction)
        dots = (x * x_norm).sum(dim=-1)
        assert (dots > 0).all()

    def test_normalize_batch_dimension(self):
        """Test normalization along batch dimension."""
        x = torch.randn(32, 64)
        x_norm = l2_normalize(x, dim=0)

        norms = torch.norm(x_norm, p=2, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNormalizedEmbedding:
    """Tests for NormalizedEmbedding layer."""

    def test_embedding_output_normalized(self):
        """Test that embedding outputs are unit norm."""
        emb = NormalizedEmbedding(num_embeddings=100, embedding_dim=32)
        indices = torch.randint(0, 100, (16,))

        output = emb(indices)
        norms = torch.norm(output, p=2, dim=-1)

        assert output.shape == (16, 32)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_embedding_weights_normalized(self):
        """Test that embedding weights are unit norm."""
        emb = NormalizedEmbedding(num_embeddings=100, embedding_dim=32)

        # Check initial weights
        norms = torch.norm(emb.weight.data, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_normalize_weights_after_update(self):
        """Test in-place weight normalization."""
        emb = NormalizedEmbedding(num_embeddings=100, embedding_dim=32)

        # Simulate optimizer update (disturb weights)
        with torch.no_grad():
            emb.weight.data += torch.randn_like(emb.weight.data) * 0.1

        # Weights should no longer be normalized
        norms = torch.norm(emb.weight.data, p=2, dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Normalize
        emb.normalize_weights_()

        # Now weights should be normalized
        norms = torch.norm(emb.weight.data, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_padding_idx_stays_zero(self):
        """Test that padding index stays zero after normalization."""
        emb = NormalizedEmbedding(num_embeddings=100, embedding_dim=32, padding_idx=0)

        # Check padding is zero
        assert torch.allclose(emb.weight.data[0], torch.zeros(32))

        # After normalization
        emb.normalize_weights_()
        assert torch.allclose(emb.weight.data[0], torch.zeros(32))


class TestNormalizedLinear:
    """Tests for NormalizedLinear layer."""

    def test_output_bounded(self):
        """Test that output values are bounded (cosine similarity)."""
        layer = NormalizedLinear(64, 32, scale_init=1.0, scale_factor=1.0)

        # Normalized input
        x = l2_normalize(torch.randn(16, 64), dim=-1)
        output = layer(x)

        # Without scaling, output should be bounded in [-1, 1]
        # With scale_init=scale_factor=1.0, actual_scale = 1.0
        assert output.shape == (16, 32)
        # Values should be reasonable (scaled cosine similarities)
        assert output.abs().max() < 10  # With reasonable scaling

    def test_weights_normalized(self):
        """Test that weights are normalized along input dimension."""
        layer = NormalizedLinear(64, 32)

        # During forward, weights are normalized
        x = torch.randn(16, 64)
        _ = layer(x)

        # Check weight norms
        norms = torch.norm(layer.weight.data, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_scaling_factor(self):
        """Test that scaling factor controls effective learning rate."""
        layer1 = NormalizedLinear(64, 32, scale_init=1.0, scale_factor=1.0)
        layer2 = NormalizedLinear(64, 32, scale_init=1.0, scale_factor=0.1)

        # Same weights
        with torch.no_grad():
            layer2.weight.data = layer1.weight.data.clone()
            layer2.scale.data = layer1.scale.data.clone() * 0.1

        x = l2_normalize(torch.randn(16, 64), dim=-1)

        out1 = layer1(x)
        out2 = layer2(x)

        # Outputs should be proportional to scale
        # layer2 has scale_factor=0.1, so actual_scale = scale * (1.0/0.1) = scale * 10
        # But layer2.scale = layer1.scale * 0.1, so actual_scale = layer1.scale * 0.1 * 10 = layer1.scale
        # So outputs should be the same
        assert torch.allclose(out1, out2, atol=1e-5)


class TestNormalizedMLP:
    """Tests for NormalizedMLP layer."""

    def test_output_normalized(self):
        """Test that MLP output is normalized when using LERP."""
        mlp = NormalizedMLP(input_dim=64, hidden_dim=256)
        h = l2_normalize(torch.randn(16, 64), dim=-1)

        h_out, h_M = mlp(h, use_lerp=True)

        # Output should be normalized
        norms = torch.norm(h_out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_lerp_interpolation(self):
        """Test that LERP update is between h and h_M."""
        mlp = NormalizedMLP(input_dim=64, hidden_dim=256, alpha_init=0.5)
        h = l2_normalize(torch.randn(16, 64), dim=-1)

        h_out, h_M = mlp(h, use_lerp=True)

        # h_out should be between h and h_M (on hypersphere)
        # Check that h_out is closer to weighted average
        h_norm = l2_normalize(h, dim=-1)

        # Dot product with h should decrease, with h_M should increase
        # compared to h being closer to itself
        dot_h = (h_out * h_norm).sum(dim=-1).mean()
        dot_hM = (h_out * h_M).sum(dim=-1).mean()

        # Both should be positive (same hemisphere)
        assert dot_h > 0
        assert dot_hM > 0

    def test_no_lerp_returns_block_output(self):
        """Test that use_lerp=False returns raw block output."""
        mlp = NormalizedMLP(input_dim=64, hidden_dim=256)
        h = l2_normalize(torch.randn(16, 64), dim=-1)

        h_out, h_M = mlp(h, use_lerp=False)

        # h_out should equal h_M
        assert torch.allclose(h_out, h_M, atol=1e-5)

    def test_glu_vs_non_glu(self):
        """Test GLU vs non-GLU variants."""
        mlp_glu = NormalizedMLP(input_dim=64, hidden_dim=256, use_glu=True)
        mlp_no_glu = NormalizedMLP(input_dim=64, hidden_dim=256, use_glu=False)

        h = l2_normalize(torch.randn(16, 64), dim=-1)

        out_glu, _ = mlp_glu(h, use_lerp=True)
        out_no_glu, _ = mlp_no_glu(h, use_lerp=True)

        # Both should produce valid normalized outputs
        assert torch.norm(out_glu, p=2, dim=-1).allclose(torch.ones(16), atol=1e-5)
        assert torch.norm(out_no_glu, p=2, dim=-1).allclose(torch.ones(16), atol=1e-5)


class TestNormalizedResidualMLP:
    """Tests for NormalizedResidualMLP layer."""

    def test_output_shape_preserved(self):
        """Test that output dimension equals input dimension."""
        mlp = NormalizedResidualMLP(
            input_dim=64, hidden_dims=[256, 256], alpha_init=0.1
        )
        x = torch.randn(16, 64)

        output = mlp(x)

        assert output.shape == (16, 64)

    def test_output_normalized(self):
        """Test that output is normalized."""
        mlp = NormalizedResidualMLP(
            input_dim=64, hidden_dims=[256, 256], alpha_init=0.1
        )
        x = torch.randn(16, 64)

        output = mlp(x)
        norms = torch.norm(output, p=2, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_multiple_layers(self):
        """Test with multiple layers."""
        mlp = NormalizedResidualMLP(
            input_dim=64, hidden_dims=[128, 256, 128], alpha_init=0.1
        )

        assert len(mlp.layers) == 3

        x = torch.randn(16, 64)
        output = mlp(x)

        assert output.shape == (16, 64)

    def test_dropout(self):
        """Test with dropout."""
        mlp = NormalizedResidualMLP(
            input_dim=64, hidden_dims=[256], alpha_init=0.1, dropout=0.1
        )

        mlp.train()
        x = torch.randn(16, 64)

        # Should work without errors
        output = mlp(x)
        assert output.shape == (16, 64)


class TestWeightNormalizationCallback:
    """Tests for WeightNormalizationCallback."""

    def test_normalizes_all_layers(self):
        """Test that callback normalizes all normalized layers."""
        model = nn.Sequential(
            NormalizedLinear(64, 32),
            NormalizedLinear(32, 16),
        )

        callback = WeightNormalizationCallback(model)

        # Disturb weights
        for module in model.modules():
            if isinstance(module, NormalizedLinear):
                with torch.no_grad():
                    module.weight.data += torch.randn_like(module.weight.data) * 0.1

        # Normalize
        callback()

        # Check all weights normalized
        for module in model.modules():
            if isinstance(module, NormalizedLinear):
                norms = torch.norm(module.weight.data, p=2, dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNormalizedMultiHeadDiversityModel:
    """Tests for NormalizedMultiHeadDiversityModel."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            "embedding_dim": 16,
            "model": {
                "backbone_type": "gated_dcn",
                "backbone_config": {
                    "use_dcn": True,
                    "dcn_num_layers": 2,
                    "dcn_use_layernorm": False,
                    "mlp_hidden_dims": [32],
                    "mlp_activation": "relu",
                    "mlp_dropout": 0.0,
                    "mlp_use_skip_connections": False,
                    "use_senet": False,
                    "use_feature_gating": False,
                },
                "heads": [
                    {
                        "hidden_dims": [16],
                        "activation": "relu",
                        "dropout": 0.0,
                        "use_layer_norm": False,
                        "use_skip_connections": False,
                    },
                    {
                        "hidden_dims": [16],
                        "activation": "relu",
                        "dropout": 0.0,
                        "use_layer_norm": False,
                        "use_skip_connections": False,
                    },
                ],
                "diversity_weight": 0.1,
                "feature_bagging_ratio": 1.0,
                "aggregation_method": "mean",
                "gating_hidden_dim": None,
                # nGPT parameters
                "use_normalized_embeddings": True,
                "use_normalized_weights": True,
                "alpha_init": 0.05,
                "alpha_scale": None,
                "su_init": 1.0,
                "sv_init": 1.0,
                "use_lerp_updates": True,
                "normalize_before_head": True,
            },
            "feature_embeddings": {},
        }

    @pytest.fixture
    def vocab_sizes(self):
        """Sample vocabulary sizes."""
        return {"feat1": 100, "feat2": 50, "feat3": 200}

    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ["feat1", "feat2", "feat3"]

    def test_model_creation(self, sample_config, vocab_sizes, feature_names):
        """Test that model can be created."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        assert model is not None
        assert len(model.heads) == 2
        assert model.model_name() == "normalized_multi_head_diversity"

    def test_forward_pass(self, sample_config, vocab_sizes, feature_names):
        """Test forward pass produces correct output structure."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        # Create batch
        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )

        output = model(x)

        assert "logits" in output
        assert "aux_logits" in output
        assert output["logits"].shape == (batch_size, 1)
        assert output["aux_logits"].shape == (2, batch_size, 1)  # 2 heads

    def test_loss_computation(self, sample_config, vocab_sizes, feature_names):
        """Test loss computation."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )
        y = torch.rand(batch_size, 1)

        output = model(x)
        loss = model.compute_loss(output, y)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_predictions(self, sample_config, vocab_sizes, feature_names):
        """Test get_predictions returns probabilities."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )

        preds = model.get_predictions(x)

        assert preds.shape == (batch_size, 1)
        assert (preds >= 0).all()
        assert (preds <= 1).all()

    def test_weight_normalization(self, sample_config, vocab_sizes, feature_names):
        """Test weight normalization callback."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        # Simulate training step
        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )
        y = torch.rand(batch_size, 1)

        output = model(x)
        loss = model.compute_loss(output, y)
        loss.backward()

        # Simulate optimizer step (disturb weights)
        for param in model.parameters():
            if param.grad is not None:
                with torch.no_grad():
                    param.data -= 0.01 * param.grad

        # Normalize weights
        model.normalize_weights()

        # Check embeddings are normalized
        for name, emb in model.embeddings.items():
            if isinstance(emb, NormalizedEmbedding):
                norms = torch.norm(emb.weight.data, p=2, dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_feature_bagging(self, sample_config, vocab_sizes, feature_names):
        """Test feature bagging."""
        sample_config["model"]["feature_bagging_ratio"] = 0.5
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        # Check masks are registered
        assert hasattr(model, "head_mask_0")
        assert hasattr(model, "head_mask_1")

        # Masks should have approximately 50% features
        mask0 = model.head_mask_0
        mask1 = model.head_mask_1
        assert mask0.shape == (3,)
        assert mask1.shape == (3,)

    def test_gated_aggregation(self, sample_config, vocab_sizes, feature_names):
        """Test gated aggregation."""
        sample_config["model"]["aggregation_method"] = "gated"
        sample_config["model"]["gating_hidden_dim"] = 8

        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )

        output = model(x)
        assert output["logits"].shape == (batch_size, 1)

    def test_without_normalized_weights(
        self, sample_config, vocab_sizes, feature_names
    ):
        """Test model with normalized weights disabled."""
        sample_config["model"]["use_normalized_weights"] = False

        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )

        output = model(x)
        assert output["logits"].shape == (batch_size, 1)

    def test_without_lerp_updates(self, sample_config, vocab_sizes, feature_names):
        """Test model with LERP updates disabled."""
        sample_config["model"]["use_lerp_updates"] = False

        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )

        output = model(x)
        assert output["logits"].shape == (batch_size, 1)

    def test_gradient_flow(self, sample_config, vocab_sizes, feature_names):
        """Test that gradients flow through the model."""
        model = NormalizedMultiHeadDiversityModel(
            vocab_sizes=vocab_sizes,
            feature_names=feature_names,
            config=sample_config,
        )

        batch_size = 32
        x = torch.stack(
            [torch.randint(0, vocab_sizes[f], (batch_size,)) for f in feature_names],
            dim=1,
        )
        y = torch.rand(batch_size, 1)

        output = model(x)
        loss = model.compute_loss(output, y)
        loss.backward()

        # Check that gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
