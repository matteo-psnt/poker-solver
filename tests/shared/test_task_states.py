"""The one classification that has cost money, pinned.

Not a filing convention: on 2026-08-04 four abandoned attempts were credited
with 455 of the 718 node-hours the cost screen reported, growing by four hours
per elapsed hour, because an open interval was run to `now` for anything that
was not terminal. Queue time is not node time, and the two sets in this module
are what say so.

The rest of the vocabulary is checked here only where Batch's word means the
opposite of what it reads as -- those are the ones a reasonable person renames
by accident.
"""

from __future__ import annotations

from src.shared import task_history, task_states
from src.shared.task_states import Outcome, Phase


class TestQueuedTimeIsNotNodeTime:
    def test_a_queued_task_does_not_occupy_a_node(self):
        assert Phase.QUEUED not in task_states.OCCUPIES_A_NODE

    def test_a_queued_task_is_still_in_flight(self):
        """It is worth LISTING -- it just cannot be charged for."""
        assert Phase.QUEUED in task_states.IN_FLIGHT

    def test_the_live_causes_on_the_share_derive_from_that_decision(self):
        """`task_history.LIVE_CAUSES` is the same decision, not a second list."""
        assert task_history.LIVE_CAUSES == task_states.LIVE_CAUSES
        assert "active" not in task_history.LIVE_CAUSES


class TestBatchsWordsMeanTheOppositeOfWhatTheyRead:
    def test_active_is_queued(self):
        assert task_states.phase_of("BatchTaskState.ACTIVE") is Phase.QUEUED

    def test_completed_says_stopped_not_succeeded(self):
        assert task_states.phase_of("BatchTaskState.COMPLETED") is Phase.FINISHED
        assert task_states.outcome_of(1) is Outcome.FAILED

    def test_an_already_shortened_state_still_classifies(self):
        """Records written by an earlier reader carry the bare word."""
        assert task_states.phase_of("running") is Phase.RUNNING

    def test_an_unfamiliar_state_is_unknown_rather_than_a_guess(self):
        assert task_states.phase_of("BatchTaskState.SOMETHING_NEW") is Phase.UNKNOWN
        assert task_states.phase_of(None) is Phase.UNKNOWN


class TestTheTwoDeathsThatMustStayApart:
    """A wrong terminal cause suppresses reconciliation permanently, so a hang
    recorded as an OOM-kill loses the observer half of the join forever."""

    def test_the_wall_clock_guard_is_not_a_crash(self):
        assert task_states.outcome_of(124) is Outcome.TIMED_OUT
        assert "hang" in (task_states.exit_meaning(124) or "")

    def test_the_oom_killer_is_a_failure(self):
        assert task_states.outcome_of(137) is Outcome.FAILED
        assert "OOM" in (task_states.exit_meaning(137) or "")

    def test_a_cancellation_is_not_a_failure(self):
        assert task_states.outcome_of(-9) is Outcome.CANCELLED

    def test_an_unfamiliar_code_is_left_as_a_number(self):
        assert task_states.exit_meaning(42) is None
