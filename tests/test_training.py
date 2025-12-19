"""
Test suite for training components.

This module tests the FocalLoss implementation and
LRSchedulerWithWarmup for learning rate scheduling.
"""

import unittest
import torch
import torch.nn as nn


class TestFocalLoss(unittest.TestCase):
    """Tests for Focal Loss implementation."""

    @classmethod
    def setUpClass(cls):
        """Import FocalLoss from losses module."""
        from src.training.losses import FocalLoss

        cls.FocalLoss = FocalLoss

    def test_focal_loss_forward_basic(self):
        """Test basic forward pass."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1)).float()
        loss = focal(logits, targets)

        self.assertEqual(loss.dim(), 0)  # Scalar output
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_focal_loss_gamma_zero_equals_bce(self):
        """Verify gamma=0 is equivalent to standard BCE."""
        focal = self.FocalLoss(gamma=0.0)
        bce = nn.BCEWithLogitsLoss()

        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()

        focal_loss = focal(logits, targets)
        bce_loss = bce(logits, targets)

        self.assertAlmostEqual(focal_loss.item(), bce_loss.item(), places=5)

    def test_focal_loss_down_weights_easy_examples(self):
        """Verify focal loss down-weights confident predictions."""
        focal_high_gamma = self.FocalLoss(gamma=5.0)
        focal_low_gamma = self.FocalLoss(gamma=0.0)

        # Easy example: prediction and target are the same (high confidence)
        easy_logits = torch.tensor([[10.0]])  # Very confident positive
        easy_targets = torch.tensor([[1.0]])

        focal_high = focal_high_gamma(easy_logits, easy_targets)
        focal_low = focal_low_gamma(easy_logits, easy_targets)

        # High gamma should have lower loss for easy examples
        self.assertLess(focal_high.item(), focal_low.item())

    def test_focal_loss_with_alpha(self):
        """Test focal loss with class weights (alpha)."""
        focal = self.FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1)).float()
        loss = focal(logits, targets)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_focal_loss_all_zeros_targets(self):
        """Test focal loss when all targets are 0."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.zeros(8, 1)
        loss = focal(logits, targets)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_focal_loss_all_ones_targets(self):
        """Test focal loss when all targets are 1."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.ones(8, 1)
        loss = focal(logits, targets)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_focal_loss_gradient_flow(self):
        """Verify gradients flow through focal loss."""
        focal = self.FocalLoss(gamma=2.0)
        logits = torch.randn(4, 1, requires_grad=True)
        targets = torch.randint(0, 2, (4, 1)).float()

        loss = focal(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


class TestLRSchedulerWithWarmup(unittest.TestCase):
    """Tests for learning rate scheduler with warmup."""

    @classmethod
    def setUpClass(cls):
        """Import LRSchedulerWithWarmup from schedulers module."""
        from src.training.schedulers import LRSchedulerWithWarmup

        cls.LRSchedulerWithWarmup = LRSchedulerWithWarmup

    def test_warmup_phase(self):
        """Test that LR increases linearly during warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=100, total_steps=1000
        )

        initial_lr = scheduler.get_lr()
        self.assertEqual(initial_lr, 0.001)  # Base LR

        # After a few warmup steps, LR should be increasing
        for _ in range(10):
            scheduler.step()

        warmup_lr = scheduler.get_lr()
        self.assertLess(warmup_lr, initial_lr)  # Still in warmup (below base)
        self.assertGreater(warmup_lr, 0)

    def test_lr_at_warmup_end(self):
        """Test that LR equals base_lr exactly at end of warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=100, total_steps=1000
        )

        # Run exactly to end of warmup
        for _ in range(100):
            scheduler.step()

        lr_at_warmup_end = scheduler.get_lr()
        # At step 100 (end of warmup), we start cosine decay, so LR should be close to base
        self.assertAlmostEqual(lr_at_warmup_end, 0.01, places=3)

    def test_cosine_decay_phase(self):
        """Test that LR decays after warmup."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100, min_lr=1e-6
        )

        # Complete warmup
        for _ in range(10):
            scheduler.step()
        lr_after_warmup = scheduler.get_lr()

        # Continue into decay phase
        for _ in range(50):
            scheduler.step()
        lr_mid_decay = scheduler.get_lr()

        # LR should have decayed
        self.assertLess(lr_mid_decay, lr_after_warmup)
        self.assertGreater(lr_mid_decay, 1e-6)  # Above min_lr

    def test_lr_at_end(self):
        """Test that LR approaches min_lr at end of training."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        min_lr = 1e-5
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100, min_lr=min_lr
        )

        # Run to completion
        for _ in range(100):
            scheduler.step()

        final_lr = scheduler.get_lr()
        # LR should be close to min_lr at end
        self.assertAlmostEqual(final_lr, min_lr, places=5)

    def test_get_lr_returns_current(self):
        """Test that get_lr returns current optimizer LR."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = self.LRSchedulerWithWarmup(
            optimizer, warmup_steps=10, total_steps=100
        )

        for _ in range(5):
            scheduler.step()

        reported_lr = scheduler.get_lr()
        actual_lr = optimizer.param_groups[0]["lr"]
        self.assertEqual(reported_lr, actual_lr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
