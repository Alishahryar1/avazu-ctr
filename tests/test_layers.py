"""
Test suite for neural network layers.

This module tests SENetLayer (Squeeze-and-Excitation) and
FeatureGatingLayer implementations.
"""

import unittest
import torch
from src.models.layers import SENetLayer, FeatureGatingLayer


class TestSENetLayer(unittest.TestCase):
    """Tests for SENetLayer (Squeeze-and-Excitation network)."""

    def test_senet_forward_single_squeeze(self):
        """Test SENET forward pass with single squeeze function."""
        num_fields = 5
        embed_dim = 16
        senet = SENetLayer(num_fields=num_fields, feature_dims=embed_dim, squeeze_funcs=['mean'])
        # Create list of embeddings
        embeddings = [torch.randn(4, embed_dim) for _ in range(num_fields)]
        out = senet(embeddings)
        self.assertEqual(len(out), num_fields)
        for i, emb in enumerate(out):
            self.assertEqual(emb.shape, (4, embed_dim))

    def test_senet_forward_multiple_squeeze(self):
        """Test SENET forward pass with multiple squeeze functions (mean + max)."""
        num_fields = 5
        embed_dim = 16
        senet = SENetLayer(num_fields=num_fields, feature_dims=embed_dim, squeeze_funcs=['mean', 'max'])
        embeddings = [torch.randn(4, embed_dim) for _ in range(num_fields)]
        out = senet(embeddings)
        self.assertEqual(len(out), num_fields)
        for emb in out:
            self.assertEqual(emb.shape, (4, embed_dim))

    def test_senet_numerical_stability(self):
        """Verify no NaN/Inf in SENET output."""
        senet = SENetLayer(num_fields=10, feature_dims=32, squeeze_funcs=['mean', 'max'])
        for _ in range(10):
            embeddings = [torch.randn(32, 32) for _ in range(10)]
            out = senet(embeddings)
            for emb in out:
                self.assertFalse(torch.isnan(emb).any(), "NaN in output")
                self.assertFalse(torch.isinf(emb).any(), "Inf in output")

    def test_senet_gradient_flow(self):
        """Verify gradients flow through SENetLayer."""
        senet = SENetLayer(num_fields=5, feature_dims=16, squeeze_funcs=['mean', 'max'])
        embeddings = [torch.randn(4, 16, requires_grad=True) for _ in range(5)]
        out = senet(embeddings)
        loss = torch.stack([emb.sum() for emb in out]).sum()
        loss.backward()

        for name, param in senet.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")

    def test_senet_invalid_squeeze_func(self):
        """Verify SENET raises error for invalid squeeze function during forward."""
        # SENetLayer doesn't validate squeeze funcs in __init__, 
        # it raises NotImplementedError during forward if unknown func is used
        senet = SENetLayer(num_fields=5, feature_dims=16, squeeze_funcs=['invalid'])
        embeddings = [torch.randn(4, 16) for _ in range(5)]
        with self.assertRaises(NotImplementedError):
            senet(embeddings)

    def test_senet_activations(self):
        """Test SENET with different activation functions."""
        for activation in ['sigmoid', 'tanh', 'relu', 'softmax']:
            senet = SENetLayer(num_fields=5, feature_dims=16, excitation_activation=activation)
            embeddings = [torch.randn(4, 16) for _ in range(5)]
            out = senet(embeddings)
            self.assertEqual(len(out), 5, f"Failed for activation: {activation}")

    def test_senet_variable_dims_forward(self):
        """Test SENET with variable embedding dimensions per field."""
        feature_dims = [8, 16, 32, 16, 8]  # Variable dimensions
        senet = SENetLayer(num_fields=5, feature_dims=feature_dims, squeeze_funcs=['mean', 'max'])
        embeddings = [torch.randn(4, dim) for dim in feature_dims]
        out = senet(embeddings)
        self.assertEqual(len(out), 5)
        for i, emb in enumerate(out):
            self.assertEqual(emb.shape, (4, feature_dims[i]))

    def test_senet_variable_dims_gradient_flow(self):
        """Verify gradients flow through SENet with variable dimensions."""
        feature_dims = [8, 16, 32]
        senet = SENetLayer(num_fields=3, feature_dims=feature_dims, squeeze_funcs=['mean'])
        embeddings = [torch.randn(4, dim, requires_grad=True) for dim in feature_dims]
        out = senet(embeddings)
        loss = torch.stack([emb.sum() for emb in out]).sum()
        loss.backward()

        for name, param in senet.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")

    # === SENet+ Tests ===

    def test_senet_grouped_squeeze(self):
        """Test SENet+ grouped squeeze with num_groups=2."""
        senet = SENetLayer(num_fields=5, feature_dims=16, num_groups=2)
        embeddings = [torch.randn(4, 16) for _ in range(5)]
        out = senet(embeddings)
        self.assertEqual(len(out), 5)
        for emb in out:
            self.assertEqual(emb.shape, (4, 16))

    def test_senet_grouped_squeeze_multiple_groups(self):
        """Test SENet+ with num_groups=4."""
        senet = SENetLayer(num_fields=3, feature_dims=32, num_groups=4, squeeze_funcs=['mean', 'max'])
        embeddings = [torch.randn(8, 32) for _ in range(3)]
        out = senet(embeddings)
        self.assertEqual(len(out), 3)
        for emb in out:
            self.assertEqual(emb.shape, (8, 32))

    def test_senet_groups_with_variable_dims(self):
        """Test SENet+ grouped squeeze with variable dimensions."""
        feature_dims = [8, 16, 32]  # All divisible by 2
        senet = SENetLayer(num_fields=3, feature_dims=feature_dims, num_groups=2)
        embeddings = [torch.randn(4, dim) for dim in feature_dims]
        out = senet(embeddings)
        for i, emb in enumerate(out):
            self.assertEqual(emb.shape, (4, feature_dims[i]))

    def test_senet_element_reweight_mode(self):
        """Test SENet+ with reweight_mode='element'."""
        senet = SENetLayer(num_fields=5, feature_dims=16, reweight_mode='element')
        embeddings = [torch.randn(4, 16) for _ in range(5)]
        out = senet(embeddings)
        self.assertEqual(len(out), 5)
        for emb in out:
            self.assertEqual(emb.shape, (4, 16))

    def test_senet_element_reweight_variable_dims(self):
        """Test element reweight mode with variable dimensions."""
        feature_dims = [8, 16, 24]
        senet = SENetLayer(num_fields=3, feature_dims=feature_dims, reweight_mode='element')
        embeddings = [torch.randn(4, dim) for dim in feature_dims]
        out = senet(embeddings)
        for i, emb in enumerate(out):
            self.assertEqual(emb.shape, (4, feature_dims[i]))

    def test_senet_fuse(self):
        """Test SENet+ fuse (residual connection)."""
        senet = SENetLayer(num_fields=3, feature_dims=16, use_fuse=True)
        embeddings = [torch.randn(4, 16) for _ in range(3)]
        out = senet(embeddings)
        self.assertEqual(len(out), 3)
        for emb in out:
            self.assertEqual(emb.shape, (4, 16))

    def test_senet_layer_norm(self):
        """Test SENet+ with layer normalization."""
        # Use variable dims to trigger layer_norms (ModuleList)
        feature_dims = [8, 16, 32]
        senet = SENetLayer(num_fields=3, feature_dims=feature_dims, use_layer_norm=True)
        self.assertIsNotNone(senet.layer_norms)
        self.assertEqual(len(senet.layer_norms), 3)
        embeddings = [torch.randn(4, dim) for dim in feature_dims]
        out = senet(embeddings)
        self.assertEqual(len(out), 3)

    def test_senet_full_senetplus(self):
        """Test SENet+ with all features enabled."""
        senet = SENetLayer(
            num_fields=4,
            feature_dims=16,
            squeeze_funcs=['mean', 'max'],
            num_groups=2,
            reweight_mode='element',
            use_fuse=True,
            use_layer_norm=True
        )
        embeddings = [torch.randn(8, 16) for _ in range(4)]
        out = senet(embeddings)
        self.assertEqual(len(out), 4)
        for emb in out:
            self.assertEqual(emb.shape, (8, 16))
            self.assertFalse(torch.isnan(emb).any())

    def test_senet_backward_compatibility(self):
        """Test that default params preserve original behavior."""
        # Original behavior: no grouping, feature reweight, no fuse, no layer norm
        senet = SENetLayer(num_fields=5, feature_dims=16, squeeze_funcs=['mean'])
        self.assertEqual(senet.num_groups, 1)
        self.assertEqual(senet.reweight_mode, 'feature')
        self.assertFalse(senet.use_fuse)
        self.assertFalse(senet.use_layer_norm)
        # layer_norms attribute only exists when use_layer_norm=True and not uniform dims
        self.assertFalse(hasattr(senet, 'layer_norms'))

    def test_senet_invalid_groups(self):
        """Verify SENet raises error when num_groups doesn't divide embed_dim."""
        with self.assertRaises(AssertionError):
            SENetLayer(num_fields=5, feature_dims=15, num_groups=4)  # 15 not divisible by 4

    def test_senet_invalid_reweight_mode(self):
        """Verify SENet uses element mode for unrecognized reweight_mode."""
        # Implementation doesn't validate reweight_mode - unrecognized values
        # fall through to element mode (weights applied directly)
        senet = SENetLayer(num_fields=5, feature_dims=16, reweight_mode='invalid')  # type: ignore[arg-type]
        embeddings = [torch.randn(4, 16) for _ in range(5)]
        # Should work without error, using element-level reweighting
        out = senet(embeddings)
        self.assertEqual(len(out), 5)

    def test_senet_senetplus_gradient_flow(self):
        """Verify gradients flow through full SENet+ configuration."""
        senet = SENetLayer(
            num_fields=3,
            feature_dims=16,
            num_groups=2,
            reweight_mode='element',
            use_fuse=True,
            use_layer_norm=True
        )
        embeddings = [torch.randn(4, 16, requires_grad=True) for _ in range(3)]
        out = senet(embeddings)
        loss = torch.stack([emb.sum() for emb in out]).sum()
        loss.backward()

        for name, param in senet.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
