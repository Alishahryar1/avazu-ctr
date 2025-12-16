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


if __name__ == "__main__":
    unittest.main(verbosity=2)
