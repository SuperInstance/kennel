"""Tests for the kennel package."""

import pytest
import time

from kennel import Kennel, ModelState
from kennel.model import ModelInstance, DutyRoster, ShiftRecord
from kennel.rotation import RoundRobinPolicy
from kennel.cost import CostTracker


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def kennel():
    return Kennel()


# ── Model state tests ────────────────────────────────────────────

class TestModelState:
    def test_states_exist(self):
        assert ModelState.POOLED
        assert ModelState.FIELDLED
        assert ModelState.KENNELED
        assert ModelState.RETIRED

    def test_state_values(self):
        assert ModelState.POOLED.value == "pooled"
        assert ModelState.FIELDLED.value == "fieldled"
        assert ModelState.KENNELED.value == "kenneled"
        assert ModelState.RETIRED.value == "retired"


class TestModelInstance:
    def test_default_state_is_pooled(self):
        inst = ModelInstance(name="gpt-4o", alias="scout")
        assert inst.state is ModelState.POOLED
        assert inst.is_available is True
        assert inst.is_active is False

    def test_metadata_defaults_to_empty(self):
        inst = ModelInstance(name="gpt-4o", alias="scout")
        assert inst.metadata == {}

    def test_custom_metadata(self):
        inst = ModelInstance(
            name="gpt-4o", alias="scout", metadata={"team": "research"}
        )
        assert inst.metadata["team"] == "research"


# ── Kennel lifecycle tests ───────────────────────────────────────

class TestKennelDeploy:
    def test_deploy_new_model(self, kennel):
        inst = kennel.deploy(name="gpt-4o", alias="scout")
        assert inst.state is ModelState.FIELDLED
        assert inst.name == "gpt-4o"
        assert inst.alias == "scout"
        assert inst.deployed_at is not None

    def test_deploy_creates_roster(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout", max_shift_hours=4)
        roster = kennel.roster()
        assert len(roster) == 1
        assert roster[0]["alias"] == "scout"
        assert roster[0]["state"] is ModelState.FIELDLED

    def test_deploy_duplicate_alias_fieldled_fails(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        with pytest.raises(ValueError, match="already fieldled"):
            kennel.deploy(name="gpt-4o", alias="scout")

    def test_deploy_retired_alias_fails(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.retire("scout")
        with pytest.raises(ValueError, match="retired"):
            kennel.deploy(name="gpt-4o", alias="scout")


class TestKennelKennel:
    def test_kennel_fieldled_model(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        inst = kennel.kennel("scout")
        assert inst.state is ModelState.KENNELED
        assert inst.last_kenneled_at is not None

    def test_kennel_unknown_alias(self, kennel):
        with pytest.raises(KeyError):
            kennel.kennel("ghost")

    def test_kennel_non_fieldled_fails(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.kennel("scout")
        with pytest.raises(ValueError):
            kennel.kennel("scout")


class TestKennelRetire:
    def test_retire_fieldled_model(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        inst = kennel.retire("scout")
        assert inst.state is ModelState.RETIRED
        assert inst.retired_at is not None

    def test_retire_kenneled_model(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.kennel("scout")
        inst = kennel.retire("scout")
        assert inst.state is ModelState.RETIRED

    def test_retire_unknown_alias(self, kennel):
        with pytest.raises(KeyError):
            kennel.retire("ghost")

    def test_retired_excluded_from_roster(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.retire("scout")
        roster = kennel.roster()
        assert len(roster) == 0


# ── Roster / status tests ────────────────────────────────────────

class TestRosterStatus:
    def test_empty_roster(self, kennel):
        assert kennel.roster() == []

    def test_status_shape(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        status = kennel.status()
        assert "total_models" in status
        assert "by_state" in status
        assert "roster" in status
        assert "cost_summary" in status
        assert status["total_models"] == 1
        assert status["by_state"]["fieldled"] == 1

    def test_multiple_models_on_roster(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.deploy(name="claude-sonnet-4", alias="retriever")
        roster = kennel.roster()
        aliases = {r["alias"] for r in roster}
        assert aliases == {"scout", "retriever"}


# ── Cost tracking tests ──────────────────────────────────────────

class TestCostTracking:
    def test_record_usage(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.record_usage("scout", input_tokens=1000, output_tokens=500, cost_usd=0.05)
        inst = kennel._models["scout"]
        assert inst.total_input_tokens == 1000
        assert inst.total_output_tokens == 500
        assert inst.shift_cost_usd == pytest.approx(0.05)

    def test_cost_ceiling_auto_kennel(self, kennel):
        kennel.deploy(
            name="gpt-4o", alias="scout", cost_ceiling_usd=0.10
        )
        kennel.record_usage("scout", cost_usd=0.12)
        inst = kennel._models["scout"]
        assert inst.state is ModelState.KENNELED

    def test_cost_history(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.record_usage("scout", input_tokens=100, cost_usd=0.01)
        kennel.kennel("scout")
        history = kennel.cost_history("scout")
        assert len(history) == 1
        assert history[0].cost_usd == pytest.approx(0.01)
        assert history[0].input_tokens == 100

    def test_cost_summary(self, kennel):
        kennel.deploy(name="gpt-4o", alias="scout")
        kennel.record_usage("scout", cost_usd=0.03)
        kennel.kennel("scout")
        kennel.deploy(name="claude-sonnet-4", alias="retriever")
        kennel.record_usage("retriever", cost_usd=0.07)
        kennel.kennel("retriever")
        summary = kennel.cost_summary()
        assert summary["scout"] == pytest.approx(0.03)
        assert summary["retriever"] == pytest.approx(0.07)


# ── Rotation tests ──────────────────────────────────────────────

class TestRotation:
    def test_round_robin_picks_oldest_rest(self):
        policy = RoundRobinPolicy()
        a = ModelInstance(name="a", alias="a", state=ModelState.KENNELED, last_kenneled_at=100)
        b = ModelInstance(name="b", alias="b", state=ModelState.KENNELED, last_kenneled_at=200)
        picked = policy.pick_next([a, b])
        assert picked is a  # a rested first (older timestamp)

    def test_round_robin_skips_fieldled(self):
        policy = RoundRobinPolicy()
        a = ModelInstance(name="a", alias="a", state=ModelState.FIELDLED)
        b = ModelInstance(name="b", alias="b", state=ModelState.KENNELED, last_kenneled_at=100)
        picked = policy.pick_next([a, b])
        assert picked is b

    def test_round_robin_returns_none_if_all_busy(self):
        policy = RoundRobinPolicy()
        a = ModelInstance(name="a", alias="a", state=ModelState.FIELDLED)
        b = ModelInstance(name="b", alias="b", state=ModelState.RETIRED)
        assert policy.pick_next([a, b]) is None

    def test_should_kennel_max_shift(self):
        policy = RoundRobinPolicy()
        inst = ModelInstance(name="a", alias="a", state=ModelState.FIELDLED, shift_started_at=time.time() - 36000)
        roster = DutyRoster(alias="a", max_shift_hours=2.0)
        reason = policy.should_kennel(inst, roster)
        assert reason == "max_shift"

    def test_should_kennel_cost_ceiling(self):
        policy = RoundRobinPolicy()
        inst = ModelInstance(
            name="a", alias="a", state=ModelState.FIELDLED,
            shift_started_at=time.time(), shift_cost_usd=100.0,
        )
        roster = DutyRoster(alias="a", cost_ceiling_usd=50.0)
        reason = policy.should_kennel(inst, roster)
        assert reason == "cost_ceiling"

    def test_should_kennel_returns_none_when_fine(self):
        policy = RoundRobinPolicy()
        inst = ModelInstance(
            name="a", alias="a", state=ModelState.FIELDLED,
            shift_started_at=time.time(), shift_cost_usd=1.0,
        )
        roster = DutyRoster(alias="a", max_shift_hours=8.0, cost_ceiling_usd=50.0)
        assert policy.should_kennel(inst, roster) is None


# ── Shift record tests ──────────────────────────────────────────

class TestShiftRecord:
    def test_shift_record_fields(self):
        record = ShiftRecord(
            alias="scout",
            model="gpt-4o",
            started_at=1000,
            ended_at=2000,
            duration_hours=0.278,
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.03,
            ended_reason="manual",
        )
        assert record.alias == "scout"
        assert record.duration_hours == pytest.approx(0.278)
        assert record.ended_reason == "manual"


# ── CostTracker unit tests ──────────────────────────────────────

class TestCostTracker:
    def test_empty_summary(self):
        tracker = CostTracker()
        assert tracker.summary() == {}
        assert tracker.total() == 0.0

    def test_record_and_summary(self):
        tracker = CostTracker()
        r = ShiftRecord(
            alias="scout", model="gpt-4o",
            started_at=1000, ended_at=2000, duration_hours=1.0,
            cost_usd=0.50,
        )
        tracker.record(r)
        assert tracker.summary() == {"scout": 0.50}
        assert tracker.total() == pytest.approx(0.50)

    def test_multiple_records_same_alias(self):
        tracker = CostTracker()
        for cost in (0.10, 0.20, 0.05):
            tracker.record(ShiftRecord(
                alias="scout", model="gpt-4o",
                started_at=0, ended_at=1, duration_hours=0.0,
                cost_usd=cost,
            ))
        assert tracker.for_alias("scout") == pytest.approx(0.35)

    def test_reset(self):
        tracker = CostTracker()
        tracker.record(ShiftRecord(
            alias="a", model="m", started_at=0, ended_at=1,
            duration_hours=0, cost_usd=1.0,
        ))
        tracker.reset()
        assert tracker.summary() == {}
        assert tracker.total() == 0.0
