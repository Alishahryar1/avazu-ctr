"""
Test suite for FCNv2 neural network layers and model.

This module tests MultiHeadFeatureEmbedding, Exponential2LinearCrossNetwork,
Linear2ExponentialCrossNetwork, FCNv2Model, and TriBCELoss.
"""

import unittest
import torch
import torch.nn as nn
from src.models.layers.multihead_embedding import MultiHeadFeatureEmbedding
from src.models.layers.exp2lin_cross_network import Exponential2LinearCrossNetwork
from src.models.layers.lin2exp_cross_network import Linear2ExponentialCrossNetwork
from src.models.architectures.fcnv2 import FCNv2Model
from src.training.losses import TriBCELoss


class TestMultiHeadFeatureEmbedding(unittest.TestCase):
    """Tests for MultiHeadFeatureEmbedding layer."""

    def test_forward_2d_input(self):
        """Test forward pass with 2D input [B, D]."""
        num_heads = 4
        batch_size = 8
        total_dim = 64  # Must be divisible by num_heads
        
        layer = MultiHeadFeatureEmbedding(num_heads=num_heads)
        x = torch.randn(batch_size, total_dim)
        out = layer(x)
        
        # Output should be [B, H, D/H]
        expected_shape = (batch_size, num_heads, total_dim // num_heads)
        self.assertEqual(out.shape, expected_shape)

    def test_gradient_flow(self):
        """Verify gradients flow through the layer."""
        layer = MultiHeadFeatureEmbedding(num_heads=2)
        x = torch.randn(4, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_numerical_stability(self):
        """Verify no NaN/Inf in output."""
        layer = MultiHeadFeatureEmbedding(num_heads=4)
        for _ in range(10):
            x = torch.randn(16, 128)
            out = layer(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")


class TestExponential2LinearCrossNetwork(unittest.TestCase):
    """Tests for Exponential2LinearCrossNetwork (E2LCN)."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        input_dim = 64
        num_heads = 2
        batch_size = 8
        
        layer = Exponential2LinearCrossNetwork(
            input_dim=input_dim,
            exp_num_layers=2,
            lin_num_layers=2,
            num_heads=num_heads
        )
        x = torch.randn(batch_size, num_heads, input_dim)
        out = layer(x)
        
        # Output should be [B, H, 1]
        expected_shape = (batch_size, num_heads, 1)
        self.assertEqual(out.shape, expected_shape)

    def test_with_batch_norm(self):
        """Test with batch normalization enabled."""
        layer = Exponential2LinearCrossNetwork(
            input_dim=32,
            exp_num_layers=2,
            lin_num_layers=2,
            batch_norm=True,
            num_heads=4
        )
        x = torch.randn(8, 4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (8, 4, 1))

    def test_with_layer_norm(self):
        """Test with layer normalization enabled."""
        layer = Exponential2LinearCrossNetwork(
            input_dim=32,
            exp_num_layers=2,
            lin_num_layers=2,
            layer_norm=True,
            batch_norm=False,
            num_heads=4
        )
        x = torch.randn(8, 4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (8, 4, 1))

    def test_gradient_flow(self):
        """Verify gradients flow through all layers."""
        layer = Exponential2LinearCrossNetwork(
            input_dim=32,
            exp_num_layers=2,
            lin_num_layers=2,
            num_heads=2
        )
        x = torch.randn(4, 2, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")

    def test_numerical_stability(self):
        """Verify no NaN/Inf in output."""
        layer = Exponential2LinearCrossNetwork(
            input_dim=64,
            exp_num_layers=3,
            lin_num_layers=3,
            num_heads=2
        )
        for _ in range(10):
            x = torch.randn(16, 2, 64)
            out = layer(x)
            self.assertFalse(torch.isnan(out).any(), "NaN in output")
            self.assertFalse(torch.isinf(out).any(), "Inf in output")


class TestLinear2ExponentialCrossNetwork(unittest.TestCase):
    """Tests for Linear2ExponentialCrossNetwork (L2ECN)."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        input_dim = 64
        num_heads = 2
        batch_size = 8
        
        layer = Linear2ExponentialCrossNetwork(
            input_dim=input_dim,
            exp_num_layers=2,
            lin_num_layers=2,
            num_heads=num_heads
        )
        x = torch.randn(batch_size, num_heads, input_dim)
        out = layer(x)
        
        # Output should be [B, H, 1]
        expected_shape = (batch_size, num_heads, 1)
        self.assertEqual(out.shape, expected_shape)

    def test_with_batch_norm(self):
        """Test with batch normalization enabled."""
        layer = Linear2ExponentialCrossNetwork(
            input_dim=32,
            exp_num_layers=2,
            lin_num_layers=2,
            batch_norm=True,
            num_heads=4
        )
        x = torch.randn(8, 4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (8, 4, 1))

    def test_gradient_flow(self):
        """Verify gradients flow through all layers."""
        layer = Linear2ExponentialCrossNetwork(
            input_dim=32,
            exp_num_layers=2,
            lin_num_layers=2,
            num_heads=2
        )
        x = torch.randn(4, 2, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")


class TestTriBCELoss(unittest.TestCase):
    """Tests for TriBCELoss."""

    def test_forward(self):
        """Test loss computation."""
        loss_fn = TriBCELoss()
        
        y_pred = torch.randn(16, 1)
        y_d = torch.randn(16, 1)
        y_s = torch.randn(16, 1)
        y_true = torch.randint(0, 2, (16, 1)).float()
        
        loss = loss_fn(y_pred, y_d, y_s, y_true)
        
        self.assertEqual(loss.dim(), 0)  # Should be scalar
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_gradient_flow(self):
        """Verify gradients flow through the loss."""
        loss_fn = TriBCELoss()
        
        y_pred = torch.randn(16, 1, requires_grad=True)
        y_d = torch.randn(16, 1, requires_grad=True)
        y_s = torch.randn(16, 1, requires_grad=True)
        y_true = torch.randint(0, 2, (16, 1)).float()
        
        loss = loss_fn(y_pred, y_d, y_s, y_true)
        loss.backward()
        
        self.assertIsNotNone(y_pred.grad)
        self.assertIsNotNone(y_d.grad)
        self.assertIsNotNone(y_s.grad)

    def test_weighting_mechanism(self):
        """Test that worse branches get higher weights."""
        loss_fn = TriBCELoss()
        
        y_true = torch.ones(16, 1)
        y_pred = torch.ones(16, 1) * 5  # Good prediction (high logit for 1)
        y_d = torch.zeros(16, 1)  # Worse prediction
        y_s = torch.ones(16, 1) * 5  # Good prediction
        
        loss = loss_fn(y_pred, y_d, y_s, y_true)
        
        # Loss should be finite
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss > 0)


class TestFCNv2Model(unittest.TestCase):
    """Tests for FCNv2Model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.vocab_sizes = {'f1': 100, 'f2': 200, 'f3': 50}
        cls.feature_names = ['f1', 'f2', 'f3']
        cls.config = {
            'seed': 42,
            'embedding_dim': 16,
            'fcnv2_num_heads': 2,
            'fcnv2_exp_num_layers': 2,
            'fcnv2_lin_num_layers': 2,
            'fcnv2_batch_norm': True,
            'fcnv2_layer_norm': False,
            'fcnv2_dropout': 0.1,
            'use_variable_embeddings': False,
            'feature_embedding_overrides': {},
        }

    def test_forward_pass(self):
        """Test model forward pass."""
        model = FCNv2Model(self.vocab_sizes, self.feature_names, self.config)
        x = torch.randint(0, 50, (8, 3))
        
        output = model(x)
        
        self.assertIn('y_pred', output)
        self.assertIn('y_d', output)
        self.assertIn('y_s', output)
        self.assertEqual(output['y_pred'].shape, (8, 1))
        self.assertEqual(output['y_d'].shape, (8, 1))
        self.assertEqual(output['y_s'].shape, (8, 1))

    def test_get_logits(self):
        """Test get_logits convenience method."""
        model = FCNv2Model(self.vocab_sizes, self.feature_names, self.config)
        x = torch.randint(0, 50, (8, 3))
        
        logits = model.get_logits(x)
        
        self.assertEqual(logits.shape, (8, 1))

    def test_gradient_flow(self):
        """Verify gradients flow through the model."""
        model = FCNv2Model(self.vocab_sizes, self.feature_names, self.config)
        x = torch.randint(0, 50, (8, 3))
        
        output = model(x)
        loss = output['y_pred'].sum()
        loss.backward()
        
        # Check that embeddings receive gradients
        for feat in self.feature_names:
            grad = model.embeddings[feat].weight.grad
            self.assertIsNotNone(grad, f"No gradient for embedding {feat}")

    def test_numerical_stability(self):
        """Verify no NaN/Inf in output."""
        model = FCNv2Model(self.vocab_sizes, self.feature_names, self.config)
        
        for _ in range(10):
            x = torch.randint(0, 50, (16, 3))
            output = model(x)
            
            for key in ['y_pred', 'y_d', 'y_s']:
                self.assertFalse(torch.isnan(output[key]).any(), f"NaN in {key}")
                self.assertFalse(torch.isinf(output[key]).any(), f"Inf in {key}")

    def test_different_num_heads(self):
        """Test model with different number of heads."""
        for num_heads in [1, 2, 4]:
            config = self.config.copy()
            config['fcnv2_num_heads'] = num_heads
            
            model = FCNv2Model(self.vocab_sizes, self.feature_names, config)
            x = torch.randint(0, 50, (4, 3))
            output = model(x)
            
            self.assertEqual(output['y_pred'].shape, (4, 1))

    def test_with_layer_norm(self):
        """Test model with layer normalization."""
        config = self.config.copy()
        config['fcnv2_batch_norm'] = False
        config['fcnv2_layer_norm'] = True
        
        model = FCNv2Model(self.vocab_sizes, self.feature_names, config)
        x = torch.randint(0, 50, (4, 3))
        output = model(x)
        
        self.assertEqual(output['y_pred'].shape, (4, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
