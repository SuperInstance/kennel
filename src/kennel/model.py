"""Data classes for model instances, duty rosters, and shift records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelState(Enum):
    """Lifecycle states for a model instance.

    POOLED   → available but not yet deployed
    FIELDLED → actively working / serving requests
    KENNELED → resting between shifts
    RETIRED  → archived, no longer in active rotation
    """

    POOLED = "pooled"
    FIELDLED = "fieldled"
    KENNELED = "kenneled"
    RETIRED = "retired"


@dataclass
class ModelInstance:
    """A single model instance tracked by the kennel.

    Think of this as a working dog's record card: name, state,
    and a history of shifts.

    Attributes:
        name: Model identifier (e.g. "gpt-4o", "claude-sonnet-4").
        alias: Unique working name within this kennel (e.g. "scout").
        provider: Hosting provider.
        state: Current lifecycle state.
        deployed_at: Timestamp of first deployment.
        shift_started_at: When the current (or most recent) shift began.
        last_kenneled_at: When the model was last sent to rest.
        retired_at: Timestamp of retirement, if applicable.
        total_input_tokens: Cumulative input tokens across all shifts.
        total_output_tokens: Cumulative output tokens across all shifts.
        shift_cost_usd: Cost accumulated in the current shift.
        metadata: Free-form user attributes.
    """

    name: str
    alias: str
    provider: str = "unknown"
    state: ModelState = ModelState.POOLED
    deployed_at: Optional[float] = None
    shift_started_at: Optional[float] = None
    last_kenneled_at: Optional[float] = None
    retired_at: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    shift_cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deployed_at is None and self.state is ModelState.POOLED:
            # Not yet deployed — leave deployed_at as None
            pass

    @property
    def is_active(self) -> bool:
        """True if the model is fieldled (on active duty)."""
        return self.state is ModelState.FIELDLED

    @property
    def is_available(self) -> bool:
        """True if the model can be deployed (pooled or kenneled)."""
        return self.state in (ModelState.POOLED, ModelState.KENNELED)


@dataclass
class DutyRoster:
    """Shift schedule and limits for a model instance.

    The duty roster is the kennel master's rulebook for each dog:
    how long it can work, how long it must rest, and when to pull
    it off duty for cost reasons.

    Attributes:
        alias: The model alias this roster governs.
        max_shift_hours: Maximum consecutive hours on duty.
        min_rest_hours: Minimum hours of rest before re-deployment.
        cost_ceiling_usd: Per-shift cost limit before auto-kennel.
    """

    alias: str
    max_shift_hours: float = 8.0
    min_rest_hours: float = 2.0
    cost_ceiling_usd: float = 50.0


@dataclass
class ShiftRecord:
    """Permanent record of a completed shift.

    Every time a model goes on duty and comes off, a ShiftRecord
    is created and stored. This is the kennel's logbook.

    Attributes:
        alias: Model alias that worked this shift.
        model: Model name (e.g. "gpt-4o").
        started_at: Unix timestamp when shift began.
        ended_at: Unix timestamp when shift ended.
        duration_hours: Length of the shift in hours.
        input_tokens: Tokens consumed during this shift.
        output_tokens: Tokens generated during this shift.
        cost_usd: Dollar cost of this shift.
        ended_reason: Why the shift ended ("manual", "max_shift",
            "cost_ceiling", "retired").
    """

    alias: str
    model: str
    started_at: float
    ended_at: float
    duration_hours: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    ended_reason: str = "manual"
