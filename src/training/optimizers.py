"""Custom optimizers for CTR prediction models."""
import math
import torch
from torch.optim import Optimizer


class FTRLProximal(Optimizer):
    """
    FTRL-Proximal optimizer implementation in PyTorch.
    
    Follow The Regularized Leader - Proximal is an online learning algorithm
    that combines L1 and L2 regularization with per-coordinate learning rates.
    Particularly effective for sparse, high-dimensional CTR prediction tasks.
    
    Reference:
        "Ad Click Prediction: a View from the Trenches" - H. B. McMahan et al.
    
    Args:
        params: Iterable of parameters to optimize
        alpha: Learning rate proportionality constant (default: 1.0)
        beta: Learning rate parameter for smoothing (default: 1.0)
        l1: L1 regularization coefficient - higher values increase sparsity (default: 0.0)
        l2: L2 regularization coefficient (default: 0.0)
    
    The per-coordinate learning rate is: eta_i = alpha / (beta + sqrt(sum of g_i^2))
    
    Weight update formula:
        w_i = -sign(z_i) * max(0, |z_i| - l1) / (l2 + (beta + sqrt(n_i)) / alpha)
        where z_i and n_i are per-coordinate accumulators
    """
    
    def __init__(self, params, alpha=1.0, beta=1.0, l1=0.0, l2=0.0):
        if alpha <= 0:
            raise ValueError(f"Invalid alpha: {alpha}. Must be positive.")
        if beta < 0:
            raise ValueError(f"Invalid beta: {beta}. Must be non-negative.")
        if l1 < 0:
            raise ValueError(f"Invalid l1: {l1}. Must be non-negative.")
        if l2 < 0:
            raise ValueError(f"Invalid l2: {l2}. Must be non-negative.")
        
        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        
        Returns:
            Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            l1 = group['l1']
            l2 = group['l2']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    # z accumulator (sum of gradient adjustments)
                    state['z'] = torch.zeros_like(p)
                    # n accumulator (sum of squared gradients)
                    state['n'] = torch.zeros_like(p)
                
                z = state['z']
                n = state['n']
                
                # Compute sigma: (sqrt(n_new) - sqrt(n_old)) / alpha
                n_old = n.clone()
                n.add_(grad * grad)
                sigma = (torch.sqrt(n) - torch.sqrt(n_old)) / alpha
                
                # Update z accumulator: z += g - sigma * w
                z.add_(grad - sigma * p.data)
                
                # Compute new weights with L1 soft-thresholding
                # w_i = 0 if |z_i| <= l1
                # w_i = -sign(z_i) * (|z_i| - l1) / (l2 + (beta + sqrt(n_i)) / alpha) otherwise
                
                abs_z = torch.abs(z)
                sign_z = torch.sign(z)
                
                # Soft-thresholding for L1 regularization
                threshold_mask = abs_z > l1
                
                # Compute denominator: l2 + (beta + sqrt(n)) / alpha
                denom = l2 + (beta + torch.sqrt(n)) / alpha
                
                # Compute new weights
                new_weights = torch.where(
                    threshold_mask,
                    -sign_z * (abs_z - l1) / denom,
                    torch.zeros_like(p)
                )
                
                # Update parameters in-place
                p.data.copy_(new_weights)
        
        return loss
