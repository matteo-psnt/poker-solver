"""What compaction is allowed to verify against.

The bundle lands, the share is downloaded FRESH, and the joined task log is
compared -- because only the share can say whether it still answers the same.
The question is what "the same" means, and the first answer was too broad: a
whole-list comparison could not pass while a single task was running, and on
this pool something is running nearly always.

It failed exactly that way. One live task advanced its cause during the three
minutes the download, bundle and re-download took, so 10,718 files stayed loose
over a change compaction had nothing to do with.
"""

from __future__ import annotations

from src.interfaces.commands import compact_legs
from src.shared import task_history


def _row(task_id: str, attempt: int, cause: str = "completed") -> task_history.TaskRow:
    return task_history.TaskRow(task_id=task_id, attempt=attempt, cause=cause, cause_source="node")


class TestItCatchesLoss:
    """THE failure this verification exists for: records swept into a bundle
    that did not carry them."""

    def test_a_lost_attempt(self):
        lost, _ = compact_legs._regressions(
            [_row("t", 1), _row("u", 1)], [_row("t", 1)], ["t.1.start.json"]
        )
        assert lost == [("u", 1)]

    def test_a_lost_attempt_it_did_not_move_is_still_loss(self):
        """Compaction deletes files; a row vanishing is damage whoever moved
        it."""
        lost, _ = compact_legs._regressions(
            [_row("t", 1), _row("u", 1)], [_row("t", 1)], ["other.1.start.json"]
        )
        assert lost == [("u", 1)]

    def test_a_moved_attempt_that_came_back_different(self):
        """The same loss, one row at a time."""
        _, altered = compact_legs._regressions(
            [_row("t", 1, "completed")], [_row("t", 1, "failed")], ["t.1.start.json"]
        )
        assert altered == [("t", 1)]


class TestItToleratesTheWorldMovingOn:
    def test_a_live_task_advancing(self):
        """It was not moved, so it is not compaction's doing -- and the world
        does not stop for one."""
        assert compact_legs._regressions(
            [_row("live", 1, "running")], [_row("live", 1, "completed")], ["sealed.1.start.json"]
        ) == ([], [])

    def test_a_task_that_started_during_the_run(self):
        """News, not damage."""
        assert compact_legs._regressions(
            [_row("t", 1)], [_row("t", 1), _row("new", 1, "running")], ["t.1.start.json"]
        ) == ([], [])

    def test_nothing_moved_and_nothing_changed(self):
        assert compact_legs._regressions([_row("t", 1)], [_row("t", 1)], []) == ([], [])
