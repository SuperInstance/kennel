"""Shift rotation logic — prevents overgrazing and context poisoning.

In the Working Animal Architecture, overgrazing = a single model
accumulating too much context, leading to degraded output quality.
The rotation module enforces rest periods and shift limits.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from kennel.model import ModelInstance, ModelState, DutyRoster, ShiftRecord


class RotationPolicy(ABC):
    """Abstract base for rotation policies.

    A rotation policy decides:
    1. Whether a kenneled model has rested enough to re-deploy.
    2. Whether a fielded model must be pulled off shift.
    3. Which model to pick next from the pool.
    """

    @abstractmethod
    def is_rested(
        self, instance: ModelInstance, min_rest_hours: float
    ) -> bool:
        """Return True if the model has rested long enough."""
        ...

    @abstractmethod
    def should_kennel(
        self, instance: ModelInstance, roster: DutyRoster
    ) -> Optional[str]:
        """Return a reason string if the model should be kenneled, else None."""
        ...

    @abstractmethod
    def pick_next(
        self,
        instances: list[ModelInstance],
    ) -> Optional[ModelInstance]:
        """Pick the next model to deploy from available instances."""
        ...

    # -- Shared helpers used by concrete policies --

    def start_shift(
        self,
        instance: ModelInstance,
        roster: DutyRoster,
    ) -> None:
        """Transition a model onto active duty."""
        now = time.time()
        instance.state = ModelState.FIELDLED
        instance.shift_started_at = now
        instance.shift_cost_usd = 0.0
        if instance.deployed_at is None:
            instance.deployed_at = now

    def end_shift(
        self,
        instance: ModelInstance,
        roster: DutyRoster,
        reason: str = "manual",
    ) -> ShiftRecord:
        """End the current shift and return a ShiftRecord."""
        now = time.time()
        assert instance.shift_started_at is not None, (
            "Cannot end shift — no shift in progress."
        )
        duration = (now - instance.shift_started_at) / 3600.0
        record = ShiftRecord(
            alias=instance.alias,
            model=instance.name,
            started_at=instance.shift_started_at,
            ended_at=now,
            duration_hours=round(duration, 4),
            input_tokens=instance.total_input_tokens,
            output_tokens=instance.total_output_tokens,
            cost_usd=instance.shift_cost_usd,
            ended_reason=reason,
        )
        instance.state = ModelState.KENNELED
        instance.last_kenneled_at = now
        instance.shift_started_at = None
        return record


class RoundRobinPolicy(RotationPolicy):
    """Simple round-robin rotation.

    Picks models in the order they were last kenneled (oldest rest first).
    Enforces minimum rest periods and maximum shift lengths.
    """

    def is_rested(
        self, instance: ModelInstance, min_rest_hours: float
    ) -> bool:
        if instance.state is ModelState.POOLED:
            return True
        if instance.last_kenneled_at is None:
            return True
        rested_hours = (time.time() - instance.last_kenneled_at) / 3600.0
        return rested_hours >= min_rest_hours

    def should_kennel(
        self, instance: ModelInstance, roster: DutyRoster
    ) -> Optional[str]:
        if instance.state is not ModelState.FIELDLED:
            return None
        if instance.shift_started_at is None:
            return None

        elapsed = (time.time() - instance.shift_started_at) / 3600.0
        if elapsed >= roster.max_shift_hours:
            return "max_shift"
        if instance.shift_cost_usd >= roster.cost_ceiling_usd:
            return "cost_ceiling"
        return None

    def pick_next(
        self, instances: list[ModelInstance]
    ) -> Optional[ModelInstance]:
        available = [
            i for i in instances if i.state in (ModelState.POOLED, ModelState.KENNELED)
        ]
        if not available:
            return None
        # Oldest rest first (longest since last kenel)
        available.sort(key=lambda i: i.last_kenneled_at or 0)
        return available[0]


class WeightedPolicy(RotationPolicy):
    """Weighted rotation based on model priority scores.

    Assigns a weight to each model and picks proportionally.
    Useful when some models are more capable but more expensive —
    you want to use them more often without burning them out.
    """

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}

    def set_weight(self, alias: str, weight: float) -> None:
        """Set the rotation weight for a model (higher = picked more often)."""
        self._weights[alias] = weight

    def is_rested(
        self, instance: ModelInstance, min_rest_hours: float
    ) -> bool:
        if instance.state is ModelState.POOLED:
            return True
        if instance.last_kenneled_at is None:
            return True
        rested_hours = (time.time() - instance.last_kenneled_at) / 3600.0
        return rested_hours >= min_rest_hours

    def should_kennel(
        self, instance: ModelInstance, roster: DutyRoster
    ) -> Optional[str]:
        if instance.state is not ModelState.FIELDLED:
            return None
        if instance.shift_started_at is None:
            return None
        elapsed = (time.time() - instance.shift_started_at) / 3600.0
        if elapsed >= roster.max_shift_hours:
            return "max_shift"
        if instance.shift_cost_usd >= roster.cost_ceiling_usd:
            return "cost_ceiling"
        return None

    def pick_next(
        self, instances: list[ModelInstance]
    ) -> Optional[ModelInstance]:
        import random

        available = [
            i for i in instances if i.state in (ModelState.POOLED, ModelState.KENNELED)
        ]
        if not available:
            return None
        weights = [self._weights.get(i.alias, 1.0) for i in available]
        return random.choices(available, weights=weights, k=1)[0]
