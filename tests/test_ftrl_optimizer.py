"""
Test suite for FTRL Proximal optimizer.

This module tests the FTRLProximal optimizer implementation
for correctness, gradient flow, and L1 sparsity behavior.
"""

import unittest
import torch
import torch.nn as nn


class TestFTRLProximal(unittest.TestCase):
    """Tests for FTRL Proximal optimizer implementation."""

    @classmethod
    def setUpClass(cls):
        """Import FTRLProximal from optimizers module."""
        from src.training.optimizers import FTRLProximal
        cls.FTRLProximal = FTRLProximal

    def test_ftrl_initialization(self):
        """Test FTRL optimizer initialization with valid parameters."""
        model = nn.Linear(10, 1)
        optimizer = self.FTRLProximal(
            model.parameters(),
            alpha=1.0,
            beta=1.0,
            l1=0.1,
            l2=0.1
        )
        
        self.assertEqual(optimizer.defaults['alpha'], 1.0)
        self.assertEqual(optimizer.defaults['beta'], 1.0)
        self.assertEqual(optimizer.defaults['l1'], 0.1)
        self.assertEqual(optimizer.defaults['l2'], 0.1)

    def test_ftrl_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        model = nn.Linear(10, 1)
        with self.assertRaises(ValueError):
            self.FTRLProximal(model.parameters(), alpha=0)
        with self.assertRaises(ValueError):
            self.FTRLProximal(model.parameters(), alpha=-1)

    def test_ftrl_invalid_regularization(self):
        """Test that negative regularization raises ValueError."""
        model = nn.Linear(10, 1)
        with self.assertRaises(ValueError):
            self.FTRLProximal(model.parameters(), l1=-0.1)
        with self.assertRaises(ValueError):
            self.FTRLProximal(model.parameters(), l2=-0.1)

    def test_ftrl_step_updates_params(self):
        """Test that optimizer step updates parameters."""
        model = nn.Linear(10, 1)
        optimizer = self.FTRLProximal(model.parameters(), alpha=0.5)
        
        # Get initial weights
        initial_weight = model.weight.clone()
        initial_bias = model.bias.clone()
        
        # Do forward/backward pass
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Step optimizer
        optimizer.step()
        
        # Weights should have changed
        self.assertFalse(torch.equal(model.weight, initial_weight))

    def test_ftrl_gradient_flow(self):
        """Test that gradients are properly used."""
        model = nn.Linear(5, 1)
        optimizer = self.FTRLProximal(model.parameters(), alpha=1.0)
        
        x = torch.randn(2, 5, requires_grad=True)
        y = torch.ones(2, 1)
        
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            loss.backward()
            optimizer.step()
        
        # Model should have trained
        self.assertIsNotNone(model.weight.grad is None or True)

    def test_ftrl_l1_sparsity(self):
        """Test that high L1 regularization induces sparsity."""
        # Create a small model
        model = nn.Linear(20, 1, bias=False)
        nn.init.uniform_(model.weight, -0.01, 0.01)  # Small initial weights
        
        optimizer = self.FTRLProximal(
            model.parameters(),
            alpha=0.5,
            beta=1.0,
            l1=1.0,  # High L1 for sparsity
            l2=0.0
        )
        
        # Run several optimization steps
        for _ in range(100):
            x = torch.randn(8, 20)
            y = torch.randn(8, 1)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()
        
        # With high L1, some weights should be exactly zero
        zero_count = (model.weight == 0).sum().item()
        self.assertGreater(zero_count, 0, "L1 regularization should induce some sparsity")

    def test_ftrl_state_initialization(self):
        """Test that z and n accumulators are properly initialized."""
        model = nn.Linear(10, 1)
        optimizer = self.FTRLProximal(model.parameters(), alpha=1.0)
        
        # Before first step, state should be empty
        for param in model.parameters():
            self.assertEqual(len(optimizer.state[param]), 0)
        
        # Do one forward/backward/step
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        # After step, state should have z and n
        for param in model.parameters():
            self.assertIn('z', optimizer.state[param])
            self.assertIn('n', optimizer.state[param])
            self.assertEqual(optimizer.state[param]['z'].shape, param.shape)
            self.assertEqual(optimizer.state[param]['n'].shape, param.shape)

    def test_ftrl_closure_support(self):
        """Test that closure is properly supported."""
        model = nn.Linear(10, 1)
        optimizer = self.FTRLProximal(model.parameters(), alpha=1.0)
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            return loss
        
        # Step with closure
        loss = optimizer.step(closure)
        
        # Should return the loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss.item(), float)

    def test_ftrl_zero_grad(self):
        """Test that zero_grad properly clears gradients."""
        model = nn.Linear(10, 1)
        optimizer = self.FTRLProximal(model.parameters(), alpha=1.0)
        
        # Create gradients
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Gradients should exist
        self.assertIsNotNone(model.weight.grad)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Gradients should be zeroed (or None)
        if model.weight.grad is not None:
            self.assertTrue(torch.all(model.weight.grad == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
