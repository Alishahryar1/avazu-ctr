"""Expanding temporal split construction."""

from __future__ import annotations

from dataclasses import dataclass

from avazu_ctr.config.schema import TemporalSplitConfig


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    final_holdout: bool = False

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end <= self.valid_start < self.valid_end:
            raise ValueError(f"invalid temporal window: {self}")


def build_temporal_windows(
    available_hours: list[int], config: TemporalSplitConfig
) -> tuple[TemporalWindow, ...]:
    hours = sorted(set(available_hours))
    if not hours:
        raise ValueError("cannot split an empty dataset")
    expected = list(range(hours[0], hours[-1] + 1))
    if hours != expected:
        missing = sorted(set(expected).difference(hours))
        raise ValueError(f"raw training data has missing hours: {missing[:10]}")

    final_valid_start = hours[-1] - config.holdout_hours + 1
    windows: list[TemporalWindow] = []
    for fold in range(config.walk_forward_folds):
        distance = config.walk_forward_folds - fold
        valid_end = final_valid_start - (distance - 1) * config.fold_hours
        valid_start = valid_end - config.fold_hours
        train_end = valid_start
        if train_end - hours[0] < config.minimum_train_hours:
            raise ValueError("not enough history for requested walk-forward folds")
        windows.append(
            TemporalWindow(
                name=f"walk_forward_{fold}",
                train_start=hours[0],
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

    windows.append(
        TemporalWindow(
            name="final_holdout",
            train_start=hours[0],
            train_end=final_valid_start,
            valid_start=final_valid_start,
            valid_end=hours[-1] + 1,
            final_holdout=True,
        )
    )
    return tuple(windows)
