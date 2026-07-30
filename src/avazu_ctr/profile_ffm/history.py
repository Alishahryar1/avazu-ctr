"""Completed-hour click history for profile FFM fields."""

from __future__ import annotations

from dataclasses import dataclass

from avazu_ctr.profile_ffm.hashing import hash_token


@dataclass
class CausalHistory:
    history: str = ""
    buffered_labels: str = ""
    previous_hour: str = ""

    def advance(
        self,
        *,
        hour: str,
        label: str,
        update_labels: bool,
        completed_events: int,
    ) -> str:
        if len(hour) != 8 or not hour.isdigit():
            raise ValueError("hour must use Avazu YYMMDDHH format")
        if completed_events <= 0:
            raise ValueError("completed_events must be positive")
        if self.previous_hour and hour < self.previous_hour:
            raise ValueError("history rows must be ordered by nondecreasing hour")
        if self.previous_hour != hour:
            self.history = (self.history + self.buffered_labels)[-completed_events:]
            self.buffered_labels = ""
            self.previous_hour = hour
        visible = self.history
        if update_labels:
            if label not in {"0", "1"}:
                raise ValueError("training labels must be 0 or 1")
            self.buffered_labels += label
        return visible


def history_tokens(
    user_count: int,
    history: str,
    *,
    count_threshold: int,
) -> tuple[str, str]:
    if user_count <= 0:
        raise ValueError("user_count must be positive")
    if user_count > count_threshold:
        history_token = f"user_click_history-{user_count}"
    else:
        history_token = f"user_click_history2-{user_count}-{history}"
    return history_token, f"user_count-{user_count}"


def history_hashes(
    user_count: int,
    history: str,
    *,
    count_threshold: int,
    bins: int,
) -> tuple[int, int]:
    history_token, count_token = history_tokens(
        user_count,
        history,
        count_threshold=count_threshold,
    )
    return hash_token(history_token, bins=bins), hash_token(count_token, bins=bins)
