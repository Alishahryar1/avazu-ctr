from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional

from avazu_ctr.numerics import (
    binary_cross_entropy_with_logits,
    log_softmax,
    soft_target_cross_entropy,
    softmax,
)


@pytest.mark.parametrize(("margin", "dtype"), [(18.0, torch.float32), (40.0, torch.float64)])
@pytest.mark.parametrize("logarithmic", [False, True])
def test_dominant_gradient_survives_precision_cliff(margin, dtype, logarithmic) -> None:
    logits = torch.tensor([margin, 0.0, 0.0], dtype=dtype, requires_grad=True)
    operation = log_softmax if logarithmic else softmax
    result = operation(logits, dim=-1)
    result[0].backward()
    tail = math.exp(-margin)
    denominator = 1 + 2 * tail
    dominant = 2 * tail / (denominator if logarithmic else denominator**2)
    expected = torch.tensor([dominant, -dominant / 2, -dominant / 2], dtype=dtype)
    torch.testing.assert_close(logits.grad, expected, rtol=3e-6, atol=0)
    if logarithmic:
        assert result[0].item() == pytest.approx(-math.log1p(2 * tail), rel=3e-6, abs=0)


def test_binary_positive_tail_survives_precision_cliff() -> None:
    logits = torch.tensor([18.0], requires_grad=True)
    binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert logits.grad is not None
    assert logits.grad.item() == pytest.approx(
        -math.exp(-18) / (1 + math.exp(-18)), rel=1e-6, abs=0
    )


@pytest.mark.parametrize("equal", [True, False])
def test_teacher_near_equality_uses_logit_differences(equal: bool) -> None:
    teacher = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    student = teacher.detach().clone()
    if not equal:
        student += torch.tensor([-1e-7, 1e-7, 0.0])
    student.requires_grad_()
    loss = soft_target_cross_entropy(student, teacher)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None
    if equal:
        assert torch.equal(student.grad, torch.zeros_like(student))
    else:
        expected = student.detach().double().softmax(-1) - teacher.detach().double().softmax(-1)
        torch.testing.assert_close(student.grad.double(), expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("operation", [softmax, log_softmax])
@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_normalization_matches_values_and_vjp_with_ties_and_strides(operation, dim) -> None:
    values = torch.tensor([[2.0, 2.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float64).T
    values.requires_grad_()
    upstream = torch.arange(6, dtype=torch.float64).reshape(3, 2) / 3
    native = torch.softmax if operation is softmax else torch.log_softmax
    actual = operation(values, dim)
    expected = native(values, dim)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(operation(values + 16, dim), actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, values, upstream)[0],
        torch.autograd.grad(expected, values, upstream)[0],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_normalization_computation_dtype_and_singleton(dtype) -> None:
    values = torch.ones(2, 1, dtype=dtype, requires_grad=True)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    logs = log_softmax(values)
    probabilities = softmax(values)
    assert logs.dtype == probabilities.dtype == expected_dtype
    assert torch.equal(logs, torch.zeros_like(logs))
    assert torch.equal(probabilities, torch.ones_like(probabilities))
    (logs.sum() + probabilities.sum()).backward()
    assert values.grad is not None
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_softmax_high_margin_arbitrary_upstream() -> None:
    values = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
    upstream = torch.tensor([2.0, -3.0, 5.0])
    (softmax(values) * upstream).sum().backward()
    assert values.grad is not None
    # Pairwise Jacobian products avoid 1-p at the dominant coordinate.
    p = values.detach().double().softmax(-1)
    expected = (p[:, None] * p[None, :] * (upstream[:, None] - upstream[None, :])).sum(1)
    torch.testing.assert_close(values.grad.double(), expected, rtol=4e-6, atol=0)
    assert abs(values.grad.sum().item()) < 1e-13


def test_binary_mean_fractional_values_gradients_and_gradcheck() -> None:
    values = torch.tensor([[-2.0, 0.3], [1.7, 4.0]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0.0, 0.3], [0.7, 1.0]], dtype=torch.float64, requires_grad=True)
    actual = binary_cross_entropy_with_logits(values, labels)
    expected = functional.binary_cross_entropy_with_logits(values, labels.detach())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, values, retain_graph=True)[0],
        torch.autograd.grad(expected, values)[0],
    )
    actual.backward()
    assert labels.grad is None
    assert torch.autograd.gradcheck(
        lambda x: binary_cross_entropy_with_logits(x, labels), (values,)
    )


@pytest.mark.parametrize("label", [0.0, 1.0])
@pytest.mark.parametrize("logit", [-40.0, 40.0])
def test_binary_correct_and_incorrect_confident_predictions(label, logit) -> None:
    values = torch.tensor([logit], requires_grad=True)
    loss = binary_cross_entropy_with_logits(values, torch.tensor([label]))
    loss.backward()
    assert values.grad is not None
    expected = -1 / (1 + math.exp(logit)) if label else 1 / (1 + math.exp(-logit))
    assert values.grad.item() == pytest.approx(expected, rel=1e-6, abs=0)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("separation", [0.0, 0.9999, 1.0001, 20.0, 10000.0])
def test_teacher_local_far_transition_and_extreme_separation(separation) -> None:
    teacher = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    student = torch.tensor(
        [[separation, -separation, 0.0]] * 2, dtype=torch.float64, requires_grad=True
    )
    loss = soft_target_cross_entropy(student, teacher)
    loss.backward()
    assert student.grad is not None
    expected = (student.detach().softmax(-1) - 1 / 3) / 2
    torch.testing.assert_close(student.grad, expected, rtol=1e-10, atol=1e-15)
    torch.testing.assert_close(
        loss, -(teacher.softmax(-1) * student.log_softmax(-1)).sum(-1).mean()
    )
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert student.grad.sum(-1).abs().max() < 1e-15


def test_teacher_tiny_probability_and_nonlast_dimension() -> None:
    teacher = torch.tensor([[0.0, -80.0, -90.0], [2.0, 1.0, 0.0]]).T.contiguous()
    student = teacher.clone()
    student[1] += torch.tensor([0.125, 0.0001])
    student.requires_grad_()
    loss = soft_target_cross_entropy(student, teacher, dim=0)
    loss.backward()
    assert student.grad is not None
    p = teacher.double().softmax(0)
    q = student.detach().double().softmax(0)
    # The dominant tiny correction is the negative sum of the tail differences.
    expected = q - p
    expected[0] = -expected[1:].sum(0)
    torch.testing.assert_close(student.grad.double(), expected / 2, rtol=2e-4, atol=0)
    torch.testing.assert_close(soft_target_cross_entropy(student + 8, teacher + 8, 0), loss)


def test_teacher_entropy_equality_mixed_dtype_and_singleton() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    student = teacher.float().requires_grad_()
    loss = soft_target_cross_entropy(student, teacher)
    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, -(teacher.softmax(-1) * teacher.log_softmax(-1)).sum())
    loss.backward()
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student))
    single = torch.tensor([[3.0], [-2.0]], requires_grad=True)
    soft_target_cross_entropy(single, torch.zeros_like(single)).backward()
    assert single.grad is not None
    assert torch.equal(single.grad, torch.zeros_like(single))


def test_teacher_gradient_is_shift_invariant_and_passes_first_order_gradcheck() -> None:
    teacher = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    student = torch.tensor([2.125, 0.75, 0.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: soft_target_cross_entropy(x, teacher), (student,))
    original = torch.autograd.grad(soft_target_cross_entropy(student, teacher), student)[0]
    shifted = torch.autograd.grad(soft_target_cross_entropy(student + 16, teacher - 8), student)[0]
    torch.testing.assert_close(shifted, original, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("operation", [softmax, log_softmax])
def test_normalization_rejects_invalid_metadata(operation) -> None:
    with pytest.raises(TypeError, match="floating"):
        operation(torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="dimension"):
        operation(torch.ones(2, 3), 2)
    with pytest.raises(ValueError, match="nonempty"):
        operation(torch.empty(2, 0))


@pytest.mark.parametrize("operation", [binary_cross_entropy_with_logits, soft_target_cross_entropy])
def test_losses_reject_invalid_metadata(operation) -> None:
    with pytest.raises(ValueError, match="shape"):
        operation(torch.ones(2, 3), torch.ones(3))
    with pytest.raises(TypeError, match="floating"):
        operation(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.int64))
