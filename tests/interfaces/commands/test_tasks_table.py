"""How `tasks` SPELLS a row for a terminal.

The renderer belongs to the command, not to the record: for any other surface
the payload is the interface. It spent a while in `shared/cloudtask/task_log`,
which put fixed-width column arithmetic inside the module the node imports
before `uv sync`.
"""

from __future__ import annotations

from src.interfaces.commands import tasks


class TestCodeLabel:
    """One short phrase for WHICH CODE a task ran.

    The branch when there is one, because that is the name the work has while it
    is being done: `worktree-hybrid-kernels` says what the arm IS, where
    `c13dcb7` says only which history it forked from and is shared by every
    worktree that has not committed yet.
    """

    def test_a_dirty_tree_is_marked_on_the_label(self):
        assert tasks.code_label({"git_branch": "main", "git_dirty": "0"}) == "main"
        assert tasks.code_label({"git_branch": "main", "git_dirty": "1"}) == "main+"

    def test_a_detached_checkout_falls_back_to_the_commit(self):
        assert tasks.code_label({"git_commit": "c13dcb7aaaa", "git_dirty": "0"}) == "c13dcb7"

    def test_a_task_recorded_before_this_existed_says_nothing(self):
        """Blank, not a plausible filler: those records genuinely do not know."""
        assert tasks.code_label({"op": "train"}) == ""

    def test_two_worktrees_on_one_dirty_commit_get_different_labels(self):
        """The case the commit alone cannot answer, which is the normal one."""
        rows = [
            {"git_commit": "c13dcb7a", "git_dirty": "1", "git_branch": branch}
            for branch in ("worktree-hybrid-kernels", "worktree-vector-cfr")
        ]
        assert len({tasks.code_label(row) for row in rows}) == 2


class TestFormatTable:
    def test_no_tasks_is_a_sentence_not_an_empty_table(self):
        assert "no task records" in tasks.format_table([])

    def test_every_column_is_present_for_a_full_row(self):
        row = {
            "task_id": "task-1",
            "attempt": 1,
            "what": "train ->5M",
            "run_id": "run-a",
            "git_branch": "main",
            "git_dirty": "0",
            "cause": "completed",
            "ended_at": "2026-08-10T00:00:00Z",
        }
        rendered = tasks.format_table([row])
        for expected in ("task-1", "train ->5M", "run-a", "main", "completed"):
            assert expected in rendered

    def test_a_missing_field_renders_blank_rather_than_none(self):
        """`None` in a column reads as a value; a blank reads as not known."""
        rendered = tasks.format_table([{"task_id": "t", "cause": "running"}])
        assert "None" not in rendered

    def test_an_estimate_below_a_minute_is_seconds(self):
        """`~0m` reads as "no estimate" when it means "nearly done" -- the first
        probe finished in under a minute and reported exactly that."""
        rendered = tasks.format_table([{"task_id": "t", "eta_seconds": 40}])
        assert "~40s" in rendered

    def test_a_long_estimate_is_hours_and_minutes(self):
        rendered = tasks.format_table([{"task_id": "t", "eta_seconds": 8040}])
        assert "~2h 14m" in rendered
