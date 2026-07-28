"""Ordered causal history as a bounded Arrow stream for lazy Polars plans."""

from __future__ import annotations

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
    size: int = field(init=False, default=0)
    keys: np.ndarray = field(init=False)
    totals: np.ndarray = field(init=False)
    last_hours: np.ndarray = field(init=False)
    last_hour_counts: np.ndarray | None = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.capacity & (self.capacity - 1):
            raise ValueError("history table capacity must be a positive power of two")
        self.keys = np.empty(self.capacity, dtype=np.uint64)
        self.totals = np.zeros(self.capacity, dtype=np.uint32)
        self.last_hours = np.zeros(self.capacity, dtype=np.int32)
        self.last_hour_counts = (
            np.zeros(self.capacity, dtype=np.uint32) if self.track_within_hour else None
        )

    @property
    def nbytes(self) -> int:
        within_hour_bytes = self.last_hour_counts.nbytes if self.last_hour_counts is not None else 0
        return self.keys.nbytes + self.totals.nbytes + self.last_hours.nbytes + within_hour_bytes

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

        self.capacity = capacity
        self.size = 0
        self.keys = np.empty(capacity, dtype=np.uint64)
        self.totals = np.zeros(capacity, dtype=np.uint32)
        self.last_hours = np.zeros(capacity, dtype=np.int32)
        self.last_hour_counts = (
            np.zeros(capacity, dtype=np.uint32) if self.track_within_hour else None
        )
        slots, inserted = self._resolve(old_keys)
        if not np.all(inserted):
            raise RuntimeError("history table rehash did not insert every key")
        self.totals[slots] = old_totals
        self.last_hours[slots] = old_last_hours
        if self.last_hour_counts is not None and old_last_hour_counts is not None:
            self.last_hour_counts[slots] = old_last_hour_counts

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


@dataclass(slots=True)
class HistoryState:
    """Fixed-width identity tables carried across ordered source partitions."""

    initial_capacity: int = _DEFAULT_CAPACITY
    tables: dict[str, _HistoryTable] = field(default_factory=dict)
    last_hour: int | None = None

    @classmethod
    def for_expected_rows(cls, rows: int) -> HistoryState:
        """Pre-size for a conservative quarter-row distinct-identity estimate."""

        if rows < 0:
            raise ValueError("expected rows cannot be negative")
        estimated_identities = max(_DEFAULT_CAPACITY, rows // 4)
        capacity = _power_of_two_at_least(int(estimated_identities / _MAX_LOAD_FACTOR) + 1)
        return cls(initial_capacity=capacity)

    @property
    def nbytes(self) -> int:
        return sum(table.nbytes for table in self.tables.values())

    def table(self, feature: str, *, within_hour: bool) -> _HistoryTable:
        table = self.tables.get(feature)
        if table is None:
            table = _HistoryTable(
                capacity=self.initial_capacity,
                track_within_hour=within_hour,
            )
            self.tables[feature] = table
        elif table.track_within_hour is not within_hour:
            raise ValueError(f"history feature {feature!r} changed within-hour semantics")
        return table


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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

    return prior, elapsed, prior_hour


def add_causal_history(
    batch: pl.DataFrame,
    config: ExperimentConfig,
    state: HistoryState,
) -> pl.DataFrame:
    """Add exact prior-event features to one bounded, ordered batch."""

    if batch.is_empty():
        return batch
    hours, _ = _validate_order(batch, state)
    expressions: list[pl.Series] = []
    for feature in config.data.features.history:
        hashes = (
            batch.select(
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
        prior, elapsed, prior_hour = _feature_history(
            hashes,
            hours,
            state.table(feature.key, within_hour=feature.within_hour),
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
    state.last_hour = int(hours[-1])
    return batch.with_columns(expressions)


def scan_with_causal_history(
    frame: pl.LazyFrame,
    config: ExperimentConfig,
    state: HistoryState,
    *,
    chunk_size: int,
) -> pl.LazyFrame:
    """Expose ordered history batches as a single-use lazy Arrow stream."""

    schema = dict(frame.collect_schema())
    for feature in config.data.features.history:
        schema[f"{feature.key}_prior_impressions_log1p"] = pl.Float32()
        schema[f"{feature.key}_hours_since_previous_impression_log1p"] = pl.Float32()
        if feature.within_hour:
            schema[f"{feature.key}_prior_hour_impressions_log1p"] = pl.Float32()
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
            table = add_causal_history(batch, config, state).to_arrow()
            yield from table.to_batches(max_chunksize=chunk_size)

    reader = pa.RecordBatchReader.from_batches(arrow_schema, record_batches())
    return pl.scan_arrow_c_stream(reader)
