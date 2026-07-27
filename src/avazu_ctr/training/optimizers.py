"""Explicit optimizer plans and successful-step schedulers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

import torch
from torch import nn
from torch.optim import Optimizer

from avazu_ctr.config.schema import (
    OptimizerConfig,
    OptimizerKind,
    OptimizerPlanConfig,
    SchedulerKind,
)
from avazu_ctr.models.base import CTRModel


class FTRLProximal(Optimizer):
    def __init__(
        self,
        params: Any,
        *,
        alpha: float,
        beta: float,
        l1: float,
        l2: float,
    ) -> None:
        if alpha <= 0 or beta < 0 or l1 < 0 or l2 < 0:
            raise ValueError("invalid FTRL hyperparameters")
        super().__init__(
            params,
            {"alpha": alpha, "beta": beta, "l1": l1, "l2": l2},
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        with torch.no_grad():
            for group in self.param_groups:
                alpha = group["alpha"]
                beta = group["beta"]
                l1 = group["l1"]
                l2 = group["l2"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    state = self.state[parameter]
                    if not state:
                        state["z"] = torch.zeros_like(parameter)
                        state["n"] = torch.zeros_like(parameter)
                    z = state["z"]
                    n = state["n"]
                    new_n = n + gradient.square()
                    sigma = (new_n.sqrt() - n.sqrt()) / alpha
                    z.add_(gradient - sigma * parameter)
                    denominator = (beta + new_n.sqrt()) / alpha + l2
                    parameter.copy_(
                        torch.where(
                            z.abs() <= l1,
                            torch.zeros_like(parameter),
                            -(z - z.sign() * l1) / denominator,
                        )
                    )
                    n.copy_(new_n)
        return loss


def _make_optimizer(parameters: list[nn.Parameter], config: OptimizerConfig) -> Optimizer:
    if not parameters:
        raise ValueError("optimizer received no parameters")
    if config.kind is OptimizerKind.ADAMW:
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    if config.kind is OptimizerKind.ADAGRAD:
        return torch.optim.Adagrad(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return FTRLProximal(
        parameters,
        alpha=config.learning_rate,
        beta=config.ftrl_beta,
        l1=config.l1,
        l2=config.l2,
    )


def _make_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler = config.scheduler
    if scheduler.kind is SchedulerKind.NONE:
        return None
    warmup_steps = int(total_steps * scheduler.warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, scheduler.minimum_lr_ratio)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return scheduler.minimum_lr_ratio + (1.0 - scheduler.minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@dataclass(slots=True)
class OptimizerBundle:
    optimizers: tuple[Optimizer, ...]
    schedulers: tuple[torch.optim.lr_scheduler.LRScheduler | None, ...]

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step_schedulers(self) -> None:
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()

    def learning_rates(self) -> dict[str, float]:
        return {
            f"learning_rate_{index}": float(optimizer.param_groups[0]["lr"])
            for index, optimizer in enumerate(self.optimizers)
        }


def build_optimizer_plan(
    model: CTRModel,
    config: OptimizerPlanConfig,
    *,
    total_steps: int,
) -> OptimizerBundle:
    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.embeddings is None:
        optimizer = _make_optimizer(all_parameters, config.dense)
        return OptimizerBundle(
            optimizers=(optimizer,),
            schedulers=(_make_scheduler(optimizer, config.dense, total_steps),),
        )

    embedding_parameters = [
        parameter for parameter in model.embedding_parameters() if parameter.requires_grad
    ]
    embedding_ids = {id(parameter) for parameter in embedding_parameters}
    dense_parameters = [
        parameter for parameter in all_parameters if id(parameter) not in embedding_ids
    ]
    if not embedding_parameters or not dense_parameters:
        raise ValueError("split optimizer plan requires embedding and dense parameters")
    embedding_optimizer = _make_optimizer(embedding_parameters, config.embeddings)
    dense_optimizer = _make_optimizer(dense_parameters, config.dense)
    return OptimizerBundle(
        optimizers=(embedding_optimizer, dense_optimizer),
        schedulers=(
            _make_scheduler(embedding_optimizer, config.embeddings, total_steps),
            _make_scheduler(dense_optimizer, config.dense, total_steps),
        ),
    )
