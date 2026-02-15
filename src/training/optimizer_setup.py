"""Optimizer and scheduler setup from config."""

from typing import Any, cast

import torch
import torch.optim as optim

from src.training.schedulers import LRSchedulerWithWarmup


def create_optimizer(
    params: list[torch.nn.Parameter], opt_cfg: dict[str, Any]
) -> optim.Optimizer:
    """Create optimizer based on config type."""
    opt_type = opt_cfg.get("type", "adamw")
    if opt_type == "ftrl":
        from src.training.optimizers import FTRLProximal

        return FTRLProximal(
            params,
            alpha=opt_cfg.get("alpha", 0.1),
            beta=opt_cfg.get("beta", 1.0),
            l1=opt_cfg.get("l1", 0.0),
            l2=opt_cfg.get("l2", 0.0),
        )
    if opt_type == "adagrad":
        return optim.Adagrad(
            params,
            lr=opt_cfg.get("lr", 1e-2),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    return optim.AdamW(
        params,
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )


def setup_optimizers(
    model: torch.nn.Module,
    dense_opt_cfg: dict[str, Any],
    embed_opt_cfg: dict[str, Any],
) -> tuple[
    optim.Optimizer | None,
    optim.Optimizer | None,
    optim.Optimizer | None,
    bool,
]:
    """
    Setup optimizers for model. Returns (optimizer, embedding_optimizer, other_optimizer, use_single_ftrl).

    When use_single_ftrl: optimizer is set, embedding_optimizer and other_optimizer are None.
    Otherwise: optimizer is None, embedding_optimizer and other_optimizer are set.
    """
    use_single_ftrl = dense_opt_cfg == embed_opt_cfg

    if use_single_ftrl:
        optimizer = create_optimizer(list(model.parameters()), dense_opt_cfg)
        return optimizer, None, None, use_single_ftrl

    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    embedding_optimizer = create_optimizer(embedding_params, embed_opt_cfg)
    other_optimizer = create_optimizer(other_params, dense_opt_cfg)
    return None, embedding_optimizer, other_optimizer, use_single_ftrl


def setup_scheduler(
    other_optimizer: optim.Optimizer | None,
    dense_opt_cfg: dict[str, Any],
    steps_per_epoch: int,
    total_epochs: int,
    use_single_ftrl: bool,
    dense_opt_type: str,
) -> LRSchedulerWithWarmup | None:
    """Create LR scheduler if applicable."""
    scheduler_cfg = cast(dict[str, Any], dense_opt_cfg.get("scheduler", {}))
    decay_type = str(scheduler_cfg.get("decay_type", "none"))
    warmup_ratio = float(scheduler_cfg.get("warmup_epoch_ratio", 0.0))
    min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(steps_per_epoch * warmup_ratio)

    use_scheduler = (
        not use_single_ftrl
        and other_optimizer is not None
        and dense_opt_type != "ftrl"
        and decay_type != "none"
    )

    if use_scheduler and other_optimizer is not None:
        return LRSchedulerWithWarmup(
            other_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
            decay_type=decay_type,
        )
    return None
