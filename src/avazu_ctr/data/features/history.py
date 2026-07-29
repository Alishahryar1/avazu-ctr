"""Ordered causal history as a bounded Arrow stream for lazy Polars plans."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pyarrow as pa

from avazu_ctr.config.schema import ExperimentConfig

_DEFAULT_CAPACITY = 1 << 10
_MAX_LOAD_FACTOR = 0.7
_UINT32_MAX = np.iinfo(np.uint32).max
_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


def _power_of_two_at_least(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


@dataclass(slots=True)
class _HistoryTable:
    """Open-addressed identity state stored entirely in fixed-width arrays."""

    capacity: int = _DEFAULT_CAPACITY
    track_within_hour: bool = False
    track_clicks: bool = False
    click_pattern_bits: int = 0
    size: int = field(init=False, default=0)
    keys: np.ndarray = field(init=False)
    totals: np.ndarray = field(init=False)
    last_hours: np.ndarray = field(init=False)
    last_hour_counts: np.ndarray | None = field(init=False)
    label_totals: np.ndarray | None = field(init=False)
    positives: np.ndarray | None = field(init=False)
    last_label_hours: np.ndarray | None = field(init=False)
    last_click_hours: np.ndarray | None = field(init=False)
    impressions_since_click: np.ndarray | None = field(init=False)
    click_patterns: np.ndarray | None = field(init=False)
    click_pattern_lengths: np.ndarray | None = field(init=False)
    pending_keys: list[np.ndarray] = field(init=False, default_factory=list)
    pending_counts: list[np.ndarray] = field(init=False, default_factory=list)
    pending_positives: list[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.capacity & (self.capacity - 1):
            raise ValueError("history table capacity must be a positive power of two")
        if not 0 <= self.click_pattern_bits <= 8:
            raise ValueError("click pattern width must be between zero and eight")
        if self.click_pattern_bits and not self.track_clicks:
            raise ValueError("click patterns require click history")
        self.keys = np.empty(self.capacity, dtype=np.uint64)
        self.totals = np.zeros(self.capacity, dtype=np.uint32)
        self.last_hours = np.zeros(self.capacity, dtype=np.int32)
        self.last_hour_counts = (
            np.zeros(self.capacity, dtype=np.uint32) if self.track_within_hour else None
        )
        self.label_totals = np.zeros(self.capacity, dtype=np.uint32) if self.track_clicks else None
        self.positives = np.zeros(self.capacity, dtype=np.uint32) if self.track_clicks else None
        self.last_label_hours = (
            np.zeros(self.capacity, dtype=np.int32) if self.track_clicks else None
        )
        self.last_click_hours = (
            np.zeros(self.capacity, dtype=np.int32) if self.track_clicks else None
        )
        self.impressions_since_click = (
            np.zeros(self.capacity, dtype=np.uint32) if self.track_clicks else None
        )
        self.click_patterns = (
            np.zeros(self.capacity, dtype=np.uint8) if self.click_pattern_bits else None
        )
        self.click_pattern_lengths = (
            np.zeros(self.capacity, dtype=np.uint8) if self.click_pattern_bits else None
        )

    @property
    def nbytes(self) -> int:
        optional = (
            self.last_hour_counts,
            self.label_totals,
            self.positives,
            self.last_label_hours,
            self.last_click_hours,
            self.impressions_since_click,
            self.click_patterns,
            self.click_pattern_lengths,
        )
        return (
            self.keys.nbytes
            + self.totals.nbytes
            + self.last_hours.nbytes
            + sum(value.nbytes for value in optional if value is not None)
        )

    def slots_for(self, unique_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return slots and identify newly inserted keys."""

        if unique_keys.dtype != np.uint64 or unique_keys.ndim != 1:
            raise TypeError("history keys must be a one-dimensional uint64 array")
        self._ensure_capacity(self.size + unique_keys.size)
        return self._resolve(unique_keys)

    def _ensure_capacity(self, maximum_size: int) -> None:
        required = self.capacity
        while maximum_size > int(required * _MAX_LOAD_FACTOR):
            required *= 2
        if required != self.capacity:
            self._resize(required)

    def _resize(self, capacity: int) -> None:
        occupied = self.totals != 0
        old_keys = self.keys[occupied]
        old_totals = self.totals[occupied]
        old_last_hours = self.last_hours[occupied]
        old_last_hour_counts = (
            self.last_hour_counts[occupied] if self.last_hour_counts is not None else None
        )
        old_label_totals = self.label_totals[occupied] if self.label_totals is not None else None
        old_positives = self.positives[occupied] if self.positives is not None else None
        old_last_label_hours = (
            self.last_label_hours[occupied] if self.last_label_hours is not None else None
        )
        old_last_click_hours = (
            self.last_click_hours[occupied] if self.last_click_hours is not None else None
        )
        old_impressions_since_click = (
            self.impressions_since_click[occupied]
            if self.impressions_since_click is not None
            else None
        )
        old_click_patterns = (
            self.click_patterns[occupied] if self.click_patterns is not None else None
        )
        old_click_pattern_lengths = (
            self.click_pattern_lengths[occupied] if self.click_pattern_lengths is not None else None
        )

        self.capacity = capacity
        self.size = 0
        self.keys = np.empty(capacity, dtype=np.uint64)
        self.totals = np.zeros(capacity, dtype=np.uint32)
        self.last_hours = np.zeros(capacity, dtype=np.int32)
        self.last_hour_counts = (
            np.zeros(capacity, dtype=np.uint32) if self.track_within_hour else None
        )
        self.label_totals = np.zeros(capacity, dtype=np.uint32) if self.track_clicks else None
        self.positives = np.zeros(capacity, dtype=np.uint32) if self.track_clicks else None
        self.last_label_hours = np.zeros(capacity, dtype=np.int32) if self.track_clicks else None
        self.last_click_hours = np.zeros(capacity, dtype=np.int32) if self.track_clicks else None
        self.impressions_since_click = (
            np.zeros(capacity, dtype=np.uint32) if self.track_clicks else None
        )
        self.click_patterns = (
            np.zeros(capacity, dtype=np.uint8) if self.click_pattern_bits else None
        )
        self.click_pattern_lengths = (
            np.zeros(capacity, dtype=np.uint8) if self.click_pattern_bits else None
        )
        slots, inserted = self._resolve(old_keys)
        if not np.all(inserted):
            raise RuntimeError("history table rehash did not insert every key")
        self.totals[slots] = old_totals
        self.last_hours[slots] = old_last_hours
        if self.last_hour_counts is not None and old_last_hour_counts is not None:
            self.last_hour_counts[slots] = old_last_hour_counts
        for current, previous in (
            (self.label_totals, old_label_totals),
            (self.positives, old_positives),
            (self.last_label_hours, old_last_label_hours),
            (self.last_click_hours, old_last_click_hours),
            (self.impressions_since_click, old_impressions_since_click),
            (self.click_patterns, old_click_patterns),
            (self.click_pattern_lengths, old_click_pattern_lengths),
        ):
            if current is not None and previous is not None:
                current[slots] = previous

    def _resolve(self, unique_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.arange(unique_keys.size, dtype=np.int64)
        candidates = np.bitwise_and(unique_keys, self.capacity - 1).astype(
            np.int64,
            copy=False,
        )
        resolved_slots = np.empty(unique_keys.size, dtype=np.int64)
        inserted = np.zeros(unique_keys.size, dtype=np.bool_)

        while positions.size:
            occupied = self.totals[candidates] != 0
            matching = occupied & (self.keys[candidates] == unique_keys[positions])
            resolved_slots[positions[matching]] = candidates[matching]

            winners = np.zeros(positions.size, dtype=np.bool_)
            empty_indices = np.flatnonzero(~occupied)
            if empty_indices.size:
                empty_slots = candidates[empty_indices]
                _, first = np.unique(empty_slots, return_index=True)
                winner_indices = empty_indices[first]
                winners[winner_indices] = True
                winner_positions = positions[winner_indices]
                winner_slots = candidates[winner_indices]
                self.keys[winner_slots] = unique_keys[winner_positions]
                # A temporary nonzero count makes winners visible to colliding
                # keys during this lookup. The caller immediately writes the
                # true count after reading the returned insertion mask.
                self.totals[winner_slots] = 1
                self.size += winner_slots.size
                inserted[winner_positions] = True
                resolved_slots[winner_positions] = winner_slots

            resolved = matching | winners
            positions = positions[~resolved]
            candidates = (candidates[~resolved] + 1) & (self.capacity - 1)

        return resolved_slots, inserted

    def queue_labels(
        self,
        keys: np.ndarray,
        counts: np.ndarray,
        positives: np.ndarray,
    ) -> None:
        if not self.track_clicks:
            raise RuntimeError("cannot queue labels for label-free history")
        self.pending_keys.append(keys)
        self.pending_counts.append(counts.astype(np.uint32, copy=False))
        self.pending_positives.append(positives.astype(np.uint32, copy=False))

    def flush_labels(self, hour: int) -> None:
        if not self.pending_keys:
            return
        if (
            self.label_totals is None
            or self.positives is None
            or self.last_label_hours is None
            or self.last_click_hours is None
            or self.impressions_since_click is None
        ):
            raise RuntimeError("click history state is incomplete")

        pending_keys = np.concatenate(self.pending_keys)
        pending_counts = np.concatenate(self.pending_counts)
        pending_positives = np.concatenate(self.pending_positives)
        keys, inverse = np.unique(pending_keys, return_inverse=True)
        counts = np.bincount(
            inverse,
            weights=pending_counts,
            minlength=keys.size,
        ).astype(np.uint64)
        positives = np.bincount(
            inverse,
            weights=pending_positives,
            minlength=keys.size,
        ).astype(np.uint64)
        slots, inserted = self.slots_for(keys)
        if np.any(inserted):
            raise RuntimeError("label history contains an identity without an impression")

        previous_label_totals = self.label_totals[slots].astype(np.uint64)
        next_label_totals = previous_label_totals + counts
        next_positives = self.positives[slots].astype(np.uint64) + positives
        if np.any(next_label_totals > _UINT32_MAX) or np.any(next_positives > _UINT32_MAX):
            raise OverflowError("causal click history count exceeds uint32 capacity")
        clicked = positives > 0
        next_since_click = np.where(
            clicked,
            0,
            self.impressions_since_click[slots].astype(np.uint64) + counts,
        )
        if np.any(next_since_click > _UINT32_MAX):
            raise OverflowError("causal impressions-since-click count exceeds uint32 capacity")

        self.label_totals[slots] = next_label_totals.astype(np.uint32)
        self.positives[slots] = next_positives.astype(np.uint32)
        previous_label_hours = self.last_label_hours[slots].astype(np.int64)
        self.last_label_hours[slots] = hour
        self.last_click_hours[slots[clicked]] = hour
        self.impressions_since_click[slots] = next_since_click.astype(np.uint32)
        if self.click_patterns is not None and self.click_pattern_lengths is not None:
            mask = (1 << self.click_pattern_bits) - 1
            gaps = np.where(
                previous_label_totals > 0,
                hour - previous_label_hours,
                1,
            )
            if np.any(gaps <= 0):
                raise ValueError("causal click-pattern hours are not strictly increasing")
            shifts = np.minimum(gaps, self.click_pattern_bits).astype(np.uint16)
            self.click_patterns[slots] = (
                (self.click_patterns[slots].astype(np.uint16) << shifts) | clicked.astype(np.uint16)
            ).astype(np.uint8) & mask
            self.click_pattern_lengths[slots] = np.minimum(
                self.click_pattern_lengths[slots].astype(np.uint16) + gaps,
                self.click_pattern_bits,
            ).astype(np.uint8)

        self.pending_keys.clear()
        self.pending_counts.clear()
        self.pending_positives.clear()


@dataclass(slots=True)
class HistoryState:
    """Fixed-width identity tables carried across ordered source partitions."""

    initial_capacity: int = _DEFAULT_CAPACITY
    global_prior: float = 0.5
    tables: dict[str, _HistoryTable] = field(default_factory=dict)
    last_hour: int | None = None
    pending_hour: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.global_prior < 1.0:
            raise ValueError("history global prior must be strictly between zero and one")

    @classmethod
    def for_expected_rows(cls, rows: int, *, global_prior: float = 0.5) -> HistoryState:
        """Pre-size for a conservative quarter-row distinct-identity estimate."""

        if rows < 0:
            raise ValueError("expected rows cannot be negative")
        if not 0.0 < global_prior < 1.0:
            raise ValueError("history global prior must be strictly between zero and one")
        estimated_identities = max(_DEFAULT_CAPACITY, rows // 4)
        capacity = _power_of_two_at_least(int(estimated_identities / _MAX_LOAD_FACTOR) + 1)
        return cls(initial_capacity=capacity, global_prior=global_prior)

    @property
    def nbytes(self) -> int:
        return sum(table.nbytes for table in self.tables.values())

    def table(
        self,
        feature: str,
        *,
        within_hour: bool,
        clicks: bool,
        click_pattern_bits: int,
    ) -> _HistoryTable:
        table = self.tables.get(feature)
        if table is None:
            table = _HistoryTable(
                capacity=self.initial_capacity,
                track_within_hour=within_hour,
                track_clicks=clicks,
                click_pattern_bits=click_pattern_bits,
            )
            self.tables[feature] = table
        elif (
            table.track_within_hour is not within_hour
            or table.track_clicks is not clicks
            or table.click_pattern_bits != click_pattern_bits
        ):
            raise ValueError(f"history feature {feature!r} changed state semantics")
        return table

    def advance_to(self, hour: int) -> None:
        if self.pending_hour is None:
            self.pending_hour = hour
            return
        if hour < self.pending_hour:
            raise ValueError("causal history partitions are not temporally ordered")
        if hour == self.pending_hour:
            return
        for table in self.tables.values():
            table.flush_labels(self.pending_hour)
        self.pending_hour = hour


def _validate_order(batch: pl.DataFrame, state: HistoryState) -> tuple[np.ndarray, np.ndarray]:
    hours = batch["_timestamp_hour"].to_numpy().astype(np.int64, copy=False)
    rows = batch["_row_index"].to_numpy().astype(np.int64, copy=False)
    if np.any(hours[1:] < hours[:-1]) or np.any((hours[1:] == hours[:-1]) & (rows[1:] < rows[:-1])):
        raise ValueError("causal history input is not ordered by timestamp and row index")
    if state.last_hour is not None and int(hours[0]) < state.last_hour:
        raise ValueError("causal history partitions are not temporally ordered")
    if int(hours.min()) < _INT32_MIN or int(hours.max()) > _INT32_MAX:
        raise OverflowError("timestamp hour exceeds the causal history state range")
    return hours, rows


def _feature_history(
    hashes: np.ndarray,
    hours: np.ndarray,
    table: _HistoryTable,
    *,
    within_hour: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    unique_keys, inverse, counts = np.unique(
        hashes,
        return_inverse=True,
        return_counts=True,
    )
    slots, inserted = table.slots_for(unique_keys)

    previous_totals = table.totals[slots].astype(np.int64)
    previous_hours = table.last_hours[slots].astype(np.int64)
    previous_totals[inserted] = 0
    previous_hours[inserted] = 0
    previous_hour_counts: np.ndarray | None = None
    if table.last_hour_counts is not None:
        previous_hour_counts = table.last_hour_counts[slots].astype(np.int64)
        previous_hour_counts[inserted] = 0

    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    group_starts = np.empty(order.size, dtype=np.bool_)
    group_starts[0] = True
    group_starts[1:] = sorted_groups[1:] != sorted_groups[:-1]
    group_start_positions = np.flatnonzero(group_starts)

    prior_in_batch_sorted = np.arange(order.size, dtype=np.int64) - np.repeat(
        group_start_positions,
        counts,
    )
    prior_in_batch = np.empty(order.size, dtype=np.int64)
    prior_in_batch[order] = prior_in_batch_sorted
    prior = previous_totals[inverse] + prior_in_batch

    has_batch_previous = prior_in_batch > 0
    previous_positions = np.empty(order.size, dtype=np.int64)
    non_start_positions = np.flatnonzero(~group_starts)
    previous_positions[order[non_start_positions]] = order[non_start_positions - 1]
    previous_event_hours = previous_hours[inverse].copy()
    previous_event_hours[has_batch_previous] = hours[previous_positions[has_batch_previous]]
    elapsed = np.where(prior > 0, hours - previous_event_hours, 0)
    if np.any(elapsed < 0):
        raise ValueError("causal history identities are not temporally ordered")

    last_sorted_positions = group_start_positions + counts - 1
    last_rows = order[last_sorted_positions]
    next_totals = previous_totals + counts
    next_last_hours = hours[last_rows]
    prior_hour: np.ndarray | None = None
    next_last_hour_counts: np.ndarray | None = None
    if within_hour:
        if previous_hour_counts is None or table.last_hour_counts is None:
            raise RuntimeError("within-hour history requires within-hour state")
        sorted_hours = hours[order]
        hour_group_starts = group_starts.copy()
        hour_group_starts[1:] |= sorted_hours[1:] != sorted_hours[:-1]
        hour_start_positions = np.flatnonzero(hour_group_starts)
        hour_group_lengths = np.diff(np.append(hour_start_positions, order.size))
        prior_hour_sorted = np.arange(order.size, dtype=np.int64) - np.repeat(
            hour_start_positions,
            hour_group_lengths,
        )
        prior_hour = np.empty(order.size, dtype=np.int64)
        prior_hour[order] = prior_hour_sorted
        prior_hour += np.where(
            (previous_totals[inverse] > 0) & (hours == previous_hours[inverse]),
            previous_hour_counts[inverse],
            0,
        )
        next_last_hour_counts = prior_hour[last_rows] + 1
        if np.any(next_last_hour_counts > _UINT32_MAX):
            raise OverflowError("causal history count exceeds uint32 capacity")

    if np.any(next_totals > _UINT32_MAX):
        raise OverflowError("causal history count exceeds uint32 capacity")
    table.totals[slots] = next_totals.astype(np.uint32)
    table.last_hours[slots] = next_last_hours.astype(np.int32)
    if table.last_hour_counts is not None and next_last_hour_counts is not None:
        table.last_hour_counts[slots] = next_last_hour_counts.astype(np.uint32)

    return prior, elapsed, prior_hour, unique_keys, inverse, counts, slots


def _click_history(
    hours: np.ndarray,
    inverse: np.ndarray,
    slots: np.ndarray,
    table: _HistoryTable,
    *,
    global_prior: float,
    smoothing: float,
    probability_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if (
        table.label_totals is None
        or table.positives is None
        or table.last_click_hours is None
        or table.impressions_since_click is None
    ):
        raise RuntimeError("click history state is incomplete")
    label_totals = table.label_totals[slots].astype(np.int64)[inverse]
    positives = table.positives[slots].astype(np.int64)[inverse]
    negatives = label_totals - positives
    posterior = (positives + smoothing * global_prior) / (label_totals + smoothing)
    bounded_posterior = np.clip(posterior, probability_clip, 1.0 - probability_clip)
    bounded_prior = min(max(global_prior, probability_clip), 1.0 - probability_clip)
    logit_lift = np.where(
        label_totals > 0,
        np.log(bounded_posterior / (1.0 - bounded_posterior))
        - math.log(bounded_prior / (1.0 - bounded_prior)),
        0.0,
    )
    last_click_hours = table.last_click_hours[slots].astype(np.int64)[inverse]
    elapsed_click = np.where(positives > 0, hours - last_click_hours, 0)
    if np.any(elapsed_click < 0):
        raise ValueError("causal click history identities are not temporally ordered")
    since_click = table.impressions_since_click[slots].astype(np.int64)[inverse]
    pattern: np.ndarray | None = None
    if table.click_patterns is not None and table.click_pattern_lengths is not None:
        pattern = (
            (table.click_pattern_lengths[slots].astype(np.int64) << table.click_pattern_bits)
            | table.click_patterns[slots].astype(np.int64)
        )[inverse]
    return positives, negatives, logit_lift, elapsed_click, since_click, pattern


def add_causal_history(
    batch: pl.DataFrame,
    config: ExperimentConfig,
    state: HistoryState,
    *,
    use_labels: bool = False,
) -> pl.DataFrame:
    """Add exact prior-event features to one bounded, ordered batch."""

    if batch.is_empty():
        return batch
    hours, _ = _validate_order(batch, state)
    boundaries = np.append(np.flatnonzero(hours[1:] != hours[:-1]) + 1, hours.size)
    starts = np.append(0, boundaries[:-1])
    outputs: list[pl.DataFrame] = []
    for start, end in zip(starts, boundaries, strict=True):
        segment = batch.slice(int(start), int(end - start))
        segment_hours = hours[start:end]
        state.advance_to(int(segment_hours[0]))
        expressions: list[pl.Series] = []
        for feature in config.data.features.history:
            hashes = (
                segment.select(
                    pl.col(feature.key)
                    .cast(pl.String)
                    .hash(
                        seed=config.training.seed,
                        seed_1=config.training.seed + 1,
                        seed_2=config.training.seed + 2,
                        seed_3=config.training.seed + 3,
                    )
                    .alias(feature.key)
                )[feature.key]
                .to_numpy()
                .astype(np.uint64, copy=False)
            )
            table = state.table(
                feature.key,
                within_hour=feature.within_hour,
                clicks=feature.clicks,
                click_pattern_bits=feature.click_pattern_bits,
            )
            (
                prior,
                elapsed,
                prior_hour,
                unique_keys,
                inverse,
                counts,
                slots,
            ) = _feature_history(
                hashes,
                segment_hours,
                table,
                within_hour=feature.within_hour,
            )
            expressions.extend(
                (
                    pl.Series(
                        f"{feature.key}_prior_impressions_log1p",
                        np.log1p(prior).astype(np.float32),
                    ),
                    pl.Series(
                        f"{feature.key}_hours_since_previous_impression_log1p",
                        np.log1p(elapsed).astype(np.float32),
                    ),
                )
            )
            if prior_hour is not None:
                expressions.append(
                    pl.Series(
                        f"{feature.key}_prior_hour_impressions_log1p",
                        np.log1p(prior_hour).astype(np.float32),
                    )
                )
            if feature.clicks:
                (
                    positives,
                    negatives,
                    logit_lift,
                    elapsed_click,
                    since_click,
                    pattern,
                ) = _click_history(
                    segment_hours,
                    inverse,
                    slots,
                    table,
                    global_prior=state.global_prior,
                    smoothing=config.data.features.target_encoding.smoothing,
                    probability_clip=config.data.features.target_encoding.probability_clip,
                )
                expressions.extend(
                    (
                        pl.Series(
                            f"{feature.key}_prior_clicks_log1p",
                            np.log1p(positives).astype(np.float32),
                        ),
                        pl.Series(
                            f"{feature.key}_prior_nonclicks_log1p",
                            np.log1p(negatives).astype(np.float32),
                        ),
                        pl.Series(
                            f"{feature.key}_prior_ctr_logit_lift",
                            logit_lift.astype(np.float32),
                        ),
                        pl.Series(
                            f"{feature.key}_hours_since_last_click_log1p",
                            np.log1p(elapsed_click).astype(np.float32),
                        ),
                        pl.Series(
                            f"{feature.key}_impressions_since_last_click_log1p",
                            np.log1p(since_click).astype(np.float32),
                        ),
                    )
                )
                if pattern is not None:
                    expressions.append(
                        pl.Series(
                            f"{feature.key}_recent_click_pattern",
                            pattern,
                            dtype=pl.Int64,
                        )
                    )
                if use_labels:
                    if "click" not in segment.columns:
                        raise ValueError("labelled click history requires a click column")
                    labels = segment["click"].to_numpy().astype(np.int64, copy=False)
                    if np.any((labels != 0) & (labels != 1)):
                        raise ValueError("click history labels must be binary")
                    positive_counts = np.bincount(
                        inverse,
                        weights=labels,
                        minlength=unique_keys.size,
                    )
                    table.queue_labels(unique_keys, counts, positive_counts)
        outputs.append(segment.with_columns(expressions))
    state.last_hour = int(hours[-1])
    return pl.concat(outputs, how="vertical")


def scan_with_causal_history(
    frame: pl.LazyFrame,
    config: ExperimentConfig,
    state: HistoryState,
    *,
    chunk_size: int,
    use_labels: bool,
) -> pl.LazyFrame:
    """Expose ordered history batches as a single-use lazy Arrow stream."""

    schema = dict(frame.collect_schema())
    for feature in config.data.features.history:
        schema[f"{feature.key}_prior_impressions_log1p"] = pl.Float32()
        schema[f"{feature.key}_hours_since_previous_impression_log1p"] = pl.Float32()
        if feature.within_hour:
            schema[f"{feature.key}_prior_hour_impressions_log1p"] = pl.Float32()
        if feature.clicks:
            schema[f"{feature.key}_prior_clicks_log1p"] = pl.Float32()
            schema[f"{feature.key}_prior_nonclicks_log1p"] = pl.Float32()
            schema[f"{feature.key}_prior_ctr_logit_lift"] = pl.Float32()
            schema[f"{feature.key}_hours_since_last_click_log1p"] = pl.Float32()
            schema[f"{feature.key}_impressions_since_last_click_log1p"] = pl.Float32()
        if feature.click_pattern_bits:
            schema[f"{feature.key}_recent_click_pattern"] = pl.Int64()
    arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema

    def record_batches() -> Iterator[pa.RecordBatch]:
        batches = frame.collect_batches(
            chunk_size=chunk_size,
            maintain_order=True,
            engine="streaming",
        )
        for batch in batches:
            if batch.is_empty():
                continue
            table = add_causal_history(
                batch,
                config,
                state,
                use_labels=use_labels,
            ).to_arrow()
            yield from table.to_batches(max_chunksize=chunk_size)

    reader = pa.RecordBatchReader.from_batches(arrow_schema, record_batches())
    return pl.scan_arrow_c_stream(reader)
