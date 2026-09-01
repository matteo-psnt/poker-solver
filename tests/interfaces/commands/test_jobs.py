"""Which jobs `jobs` shows, and which end of the list it truncates.

The truncation was reversed, and the shape of the bug is why it survived: the
DEFAULT path filters to live jobs first, and there are only ever a handful, so
`--limit` never bit there. It bit under `--all` -- the flag whose entire purpose
is to show everything -- where it hid the 52 newest jobs and rendered the 20
oldest.
"""

from __future__ import annotations

from src.interfaces.cloud.tasks.batch import BatchTask, Job
from src.interfaces.commands.jobs import select


def _job(name: str, *, state: str = "BatchJobState.COMPLETED", phase: str = "finished") -> Job:
    task = BatchTask(task=f"{name}-t", job=name, state=None, phase=phase)
    return Job(job=name, state=state, tasks=[task])


# The order Azure Batch actually returned on 2026-09-01: it opened at 08-31 and
# closed at 08-26, which is neither ascending nor descending.
BATCH_ORDER = [
    _job("poker-20260831"),
    _job("poker-20260831-big"),
    _job("poker-20260901"),
    _job("poker-20260901-big"),
    _job("poker-20260804"),
    _job("poker-20260805"),
    _job("poker-20260826"),
]


class TestTruncation:
    def test_all_keeps_the_newest_jobs_not_the_oldest(self):
        payload = select(BATCH_ORDER, show_all=True, limit=3)
        assert [job.job for job in payload.jobs] == [
            "poker-20260831-big",
            "poker-20260901",
            "poker-20260901-big",
        ]
        assert payload.hidden_jobs == 4

    def test_the_newest_job_is_rendered_last(self):
        """A terminal scrolls, so the newest belongs at the bottom -- which is
        also what `run`'s docstring promises and what the sort makes true."""
        payload = select(BATCH_ORDER, show_all=True, limit=0)
        assert payload.jobs[-1].job == "poker-20260901-big"
        assert payload.jobs[0].job == "poker-20260804"

    def test_limit_zero_hides_nothing(self):
        payload = select(BATCH_ORDER, show_all=True, limit=0)
        assert payload.hidden_jobs == 0
        assert len(payload.jobs) == len(BATCH_ORDER)


class TestTheDefaultView:
    def test_only_live_jobs_survive_without_all(self):
        jobs = [
            *BATCH_ORDER,
            _job("poker-20260901-huge", state="BatchJobState.ACTIVE", phase="running"),
        ]
        payload = select(jobs, show_all=False, limit=20)
        assert [job.job for job in payload.jobs] == ["poker-20260901-huge"]
        # `total_jobs` counts what was considered, so the hidden line reports the
        # finished ones rather than zero.
        assert payload.total_jobs == len(jobs)
        assert payload.hidden_jobs == len(BATCH_ORDER)

    def test_an_active_job_with_nothing_in_flight_is_not_live(self):
        """A job container stays open all day after its tasks have finished."""
        jobs = [_job("poker-20260901-mem", state="BatchJobState.ACTIVE", phase="finished")]
        assert select(jobs, show_all=False, limit=20).jobs == []
