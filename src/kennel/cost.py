"""Cost tracking per model per shift.

Every ShiftRecord carries a cost. This module aggregates those
costs so you can answer: "How much did each model cost me today?"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from kennel.model import ShiftRecord


class CostTracker:
    """Aggregates shift costs across the kennel.

    Records each completed shift and provides summary views by
    model alias, by day, or total.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record(some_shift_record)
        >>> tracker.summary()
        {'scout': 0.42, 'retriever': 1.13}
    """

    def __init__(self) -> None:
        self._records: list[ShiftRecord] = []
        self._by_alias: dict[str, float] = defaultdict(float)
        self._by_day: dict[str, float] = defaultdict(float)

    def record(self, shift: ShiftRecord) -> None:
        """Record a completed shift's cost."""
        self._records.append(shift)
        self._by_alias[shift.alias] += shift.cost_usd
        day = self._day_key(shift.ended_at)
        self._by_day[day] += shift.cost_usd

    def summary(self) -> dict[str, float]:
        """Return total cost per model alias."""
        return dict(self._by_alias)

    def daily_summary(self) -> dict[str, float]:
        """Return total cost per day (YYYY-MM-DD)."""
        return dict(self._by_day)

    def total(self) -> float:
        """Return total cost across all models."""
        return sum(self._by_alias.values())

    def for_alias(self, alias: str) -> float:
        """Return total cost for a specific model alias."""
        return self._by_alias.get(alias, 0.0)

    def records(self) -> list[ShiftRecord]:
        """Return all recorded shifts."""
        return list(self._records)

    def reset(self) -> None:
        """Clear all tracked costs."""
        self._records.clear()
        self._by_alias.clear()
        self._by_day.clear()

    @staticmethod
    def _day_key(timestamp: float) -> str:
        """Convert a Unix timestamp to a YYYY-MM-DD string."""
        import datetime

        dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
