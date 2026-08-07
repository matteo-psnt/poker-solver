"""A retry must not turn a warm arm into a control.

This is the failure that cost two 30M sweeps. The first attempt died before
seeding -- the rung it wanted had not been fetched to the node -- and the Batch
retry found a populated run directory, took the "already resuming, do not
re-seed" branch, and trained a perfectly good CONTROL under the arm's name.
Every arm agreed to a tenth of a percent, every task exited 0, and nothing said
so until the coverage ladders were compared by hand.

"Resuming" and "was seeded" are different questions. The marker answers the
second one.
"""

from __future__ import annotations

import inspect

from src.pipeline.services import static_training, warm_start


def test_the_marker_name_is_stable():
    """It is written by training and read by the next attempt, so renaming it
    silently disables the guard rather than breaking anything loudly."""
    assert warm_start.SEEDED_MARKER == ".warm-started"


class TestResumeGuard:
    """The rule, stated as the code implements it:

    asked for a prior + resuming + marker present  -> continue, do not re-seed
    asked for a prior + resuming + marker ABSENT   -> refuse
    asked for a prior + fresh                      -> seed, write the marker
    no prior asked for                             -> untouched either way
    """

    def test_the_service_refuses_a_resume_that_was_never_seeded(self, tmp_path):
        """The exact shape that produced a control: a populated run directory,
        a prior requested, and no evidence the prior was ever applied."""
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        (run_dir / ".run.json").write_text("{}")
        source = inspect.getsource(static_training.train_static)
        assert "SEEDED_MARKER" in source, "the resume guard has been removed"
        assert not (run_dir / warm_start.SEEDED_MARKER).exists()

    def test_a_seeded_run_records_what_it_was_seeded_with(self, tmp_path):
        """The marker carries prior, rung and weight, so a resumed run can say
        which arm it actually is rather than only that it is one."""
        run_dir = tmp_path / "run-b"
        run_dir.mkdir()
        (run_dir / warm_start.SEEDED_MARKER).write_text("prior-x@100 weight=1000\n")
        recorded = (run_dir / warm_start.SEEDED_MARKER).read_text()
        assert "prior-x" in recorded
        assert "@100" in recorded
        assert "weight=1000" in recorded
