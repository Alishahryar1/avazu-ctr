"""Collapse- and cancellation-resistant normalization and losses.

Adapted from Barron et al., https://jonbarron.info/softmaxfix/.
Finite floating inputs are required. Computation and results use FP32 (FP64
for FP64 inputs); gradient storage and exponent underflow still limit tails.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def _class_dimension(logits: torch.Tensor, dim: int) -> int:
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not -logits.ndim <= dim < logits.ndim:
        raise ValueError("class dimension is out of range")
    dim %= logits.ndim
    if logits.shape[dim] == 0:
        raise ValueError("class dimension must be nonempty")
    return dim


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize without subtracting a rounded dominant probability in backward."""
    dim = _class_dimension(logits, dim)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
        if values.shape[dim] == 1:
            return values * 0.0
        index = values.argmax(dim=dim, keepdim=True)
        maximum = values.gather(dim, index)
        # A constant at the selected maximum makes its gradient the negative
        # sum of the other logit-difference gradients, including beyond the cliff.
        shifted = (values - maximum).scatter(dim, index, 0.0)
        others = shifted.scatter(dim, index, -torch.inf)
        # Finite logits can overflow to -inf when subtracting opposite extremes.
        # Avoid the undefined all--inf logsumexp backward, even in an unused branch.
        saturated = torch.isneginf(others).all(dim=dim, keepdim=True)
        safe_others = torch.where(saturated, 0.0, others)
        normalizer = functional.softplus(torch.logsumexp(safe_others, dim=dim, keepdim=True))
        normalizer = torch.where(saturated, 0.0, normalizer)
        return shifted - normalizer


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exponentiate corrected log probabilities; forward may still round to one."""
    return log_softmax(logits, dim).exp()


def binary_cross_entropy_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean unweighted BCE for same-shape logits and constant targets in [0, 1]."""
    if logits.shape != targets.shape or logits.device != targets.device:
        raise ValueError("logits and targets must have the same shape and device")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating point")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        values = logits.to(dtype)
        labels = targets.detach().to(dtype)
        return (
            labels * functional.softplus(-values) + (1 - labels) * functional.softplus(values)
        ).mean()


def soft_target_cross_entropy(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Mean CE with detached teacher logits and corrected first-order student gradient.

    Teacher probabilities cannot replace logits here. Higher-order derivatives
    are not supported. A bounded local branch avoids overflow in unused expm1
    calculations; distant distributions use projected probability differences.
    """
    if (
        student_logits.shape != teacher_logits.shape
        or student_logits.device != teacher_logits.device
    ):
        raise ValueError("student and teacher logits must have the same shape and device")
    dim = _class_dimension(student_logits, dim)
    _class_dimension(teacher_logits, dim)
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        dtype = (
            torch.float64
            if torch.float64 in (student_logits.dtype, teacher_logits.dtype)
            else torch.float32
        )
        student = student_logits.to(dtype)
        with torch.no_grad():
            teacher = teacher_logits.to(dtype)
            log_q = log_softmax(student, dim)
            q, p = log_q.exp(), softmax(teacher, dim)
            delta = student - teacher
            centered = delta - (p * delta).sum(dim=dim, keepdim=True)
            local = centered.abs().amax(dim=dim, keepdim=True) <= 1
            safe = torch.where(local, centered, 0.0)
            correction = torch.log1p((p * safe.expm1()).sum(dim=dim, keepdim=True).clamp_min(0))
            difference = torch.where(local, p * (safe - correction).expm1(), q - p)
            gradient = difference - q * difference.sum(dim=dim, keepdim=True)
            loss = -(p * log_q).sum(dim=dim)
        surrogate = (gradient * (student - student.detach())).sum(dim=dim)
        return (loss + surrogate).mean()
