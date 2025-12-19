"""Learning rate schedulers for training."""

import numpy as np


class LRSchedulerWithWarmup:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
