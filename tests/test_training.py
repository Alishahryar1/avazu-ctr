"""
Test suite for training components.

This module tests the LRSchedulerWithWarmup for learning rate scheduling.
"""

import unittest
import torch
import torch.nn as nn


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
