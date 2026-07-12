"""
Kennel: model lifecycle management for Working Animal Architecture.

Manages AI model instances like a kennel manages working dogs:
- fielded (actively working)
- kenneled (resting between shifts)
- retired (archived)

Provides duty rosters for cost-managed shift rotation.
"""

from __future__ import annotations

import time
from typing import Optional

from kennel.model import (
    ModelInstance,
    ModelState,
    DutyRoster,
    ShiftRecord,
)
from kennel.rotation import RotationPolicy, RoundRobinPolicy
from kennel.cost import CostTracker

__version__ = "0.1.0"
__all__ = [
    "Kennel",
    "ModelInstance",
    "ModelState",
    "DutyRoster",
    "ShiftRecord",
]

# Default timing constants
_DEFAULT_MAX_SHIFT_HOURS = 8.0
_DEFAULT_MIN_REST_HOURS = 2.0
_DEFAULT_COST_CEILING_USD = 50.0


class Kennel:
    """Main API for managing model lifecycle.

    Think of this as the kennel master's office. You deploy models to
    active duty, kennel them for rest, retire them when done, and
    inspect the duty roster at any time.

    Example:
        >>> k = Kennel()
        >>> scout = k.deploy(name="gpt-4o", alias="scout")
        >>> k.roster()
        [{'alias': 'scout', 'model': 'gpt-4o', 'state': <ModelState.FIELDLED: 'fieldled'>, ...}]
        >>> k.retire("scout")
    """

    def __init__(
        self,
        rotation_policy: Optional[RotationPolicy] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self._models: dict[str, ModelInstance] = {}
        self._rosters: dict[str, DutyRoster] = {}
        self._shifts: list[ShiftRecord] = []
        self._rotation = rotation_policy or RoundRobinPolicy()
        self._costs = cost_tracker or CostTracker()

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def deploy(
        self,
        name: str,
        alias: str,
        provider: str = "unknown",
        max_shift_hours: float = _DEFAULT_MAX_SHIFT_HOURS,
        min_rest_hours: float = _DEFAULT_MIN_REST_HOURS,
        cost_ceiling_usd: float = _DEFAULT_COST_CEILING_USD,
        metadata: Optional[dict] = None,
    ) -> ModelInstance:
        """Deploy a model to active duty (FIELDLED state).

        If the model is POOLED, it enters fielded service immediately.
        If KENNELED, it must have completed its minimum rest period.

        Args:
            name: Model identifier (e.g. "gpt-4o").
            alias: Unique working name for this instance (e.g. "scout").
            provider: Provider name (openai, anthropic, etc.).
            max_shift_hours: Maximum hours on active duty before auto-kennel.
            min_rest_hours: Minimum hours of rest before re-deployment.
            cost_ceiling_usd: Auto-kennel when shift cost exceeds this.
            metadata: Free-form dict for user-defined attributes.

        Returns:
            The deployed ModelInstance.

        Raises:
            ValueError: If alias already exists and is retired, or if
                a kenneled model hasn't rested long enough.
        """
        if alias in self._models:
            existing = self._models[alias]
            if existing.state is ModelState.RETIRED:
                raise ValueError(
                    f"Alias '{alias}' is retired. Choose a new alias."
                )
            if existing.state is ModelState.FIELDLED:
                raise ValueError(
                    f"Alias '{alias}' is already fieldled."
                )
            # Kennelled — check rest period
            if not self._rotation.is_rested(
                existing, min_rest_hours
            ):
                raise ValueError(
                    f"Alias '{alias}' needs more rest before re-deployment."
                )
            instance = existing
        else:
            instance = ModelInstance(
                name=name,
                alias=alias,
                provider=provider,
                state=ModelState.POOLED,
                metadata=metadata or {},
            )
            self._models[alias] = instance
            self._rosters[alias] = DutyRoster(
                alias=alias,
                max_shift_hours=max_shift_hours,
                min_rest_hours=min_rest_hours,
                cost_ceiling_usd=cost_ceiling_usd,
            )

        # Start shift
        self._rotation.start_shift(instance, self._rosters[alias])
        return instance

    def kennel(self, alias: str) -> ModelInstance:
        """Pull a fielded model back to the kennel for rest.

        Records the completed shift and transitions to KENNELED state.

        Args:
            alias: The model's working name.

        Returns:
            The kenneled ModelInstance.

        Raises:
            KeyError: If alias not found.
            ValueError: If model is not currently fieldled.
        """
        instance = self._require(alias)
        if instance.state is not ModelState.FIELDLED:
            raise ValueError(
                f"Cannot kennel '{alias}' — current state is {instance.state.value}."
            )
        roster = self._rosters[alias]
        record = self._rotation.end_shift(instance, roster)
        self._shifts.append(record)
        self._costs.record(record)
        return instance

    def retire(self, alias: str) -> ModelInstance:
        """Retire a model permanently (RETIRED state).

        The model is archived. Historical shift and cost records are kept.

        Args:
            alias: The model's working name.

        Returns:
            The retired ModelInstance.
        """
        instance = self._require(alias)
        # If fieldled, close out the current shift first
        if instance.state is ModelState.FIELDLED:
            roster = self._rosters[alias]
            record = self._rotation.end_shift(instance, roster)
            self._shifts.append(record)
            self._costs.record(record)
        instance.state = ModelState.RETIRED
        instance.retired_at = time.time()
        return instance

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def roster(self) -> list[dict]:
        """Return the current duty roster.

        Shows every non-retired model, its state, and shift info.
        """
        result = []
        now = time.time()
        for alias, instance in self._models.items():
            if instance.state is ModelState.RETIRED:
                continue
            roster_cfg = self._rosters.get(alias)
            shift_hours = 0.0
            if instance.state is ModelState.FIELDLED and instance.shift_started_at:
                shift_hours = (now - instance.shift_started_at) / 3600.0
            result.append({
                "alias": alias,
                "model": instance.name,
                "provider": instance.provider,
                "state": instance.state,
                "shift_hours": round(shift_hours, 2),
                "max_shift_hours": roster_cfg.max_shift_hours if roster_cfg else None,
                "deployed_at": instance.deployed_at,
                "shift_started_at": instance.shift_started_at,
            })
        return result

    def status(self) -> dict:
        """Full kennel status report.

        Returns dict with counts, roster, and cost summary.
        """
        counts = {state.value: 0 for state in ModelState}
        for inst in self._models.values():
            counts[inst.state.value] += 1

        return {
            "total_models": len(self._models),
            "by_state": counts,
            "roster": self.roster(),
            "cost_summary": self._costs.summary(),
            "total_shifts": len(self._shifts),
        }

    def cost_history(self, alias: str) -> list[ShiftRecord]:
        """Return all shift records for a given model alias."""
        return [r for r in self._shifts if r.alias == alias]

    def cost_summary(self) -> dict[str, float]:
        """Aggregate cost per model alias."""
        return self._costs.summary()

    # ------------------------------------------------------------------
    # Record-keeping
    # ------------------------------------------------------------------

    def record_usage(
        self,
        alias: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record token usage and cost for a fielded model.

        Call this during a shift to track running costs. If the
        accumulated shift cost exceeds the roster's cost ceiling,
        the model is automatically kenneled.
        """
        instance = self._require(alias)
        if instance.state is not ModelState.FIELDLED:
            raise ValueError(
                f"'{alias}' is not fieldled — cannot record usage."
            )
        instance.total_input_tokens += input_tokens
        instance.total_output_tokens += output_tokens
        instance.shift_cost_usd += cost_usd

        roster = self._rosters[alias]
        if instance.shift_cost_usd >= roster.cost_ceiling_usd:
            self.kennel(alias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, alias: str) -> ModelInstance:
        if alias not in self._models:
            raise KeyError(f"Unknown alias '{alias}'.")
        return self._models[alias]
