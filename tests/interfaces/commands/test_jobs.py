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
        assert payload.filtered_jobs == 0

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
        # finished ones rather than zero -- and as FILTERED, not truncated: the
        # flag that shows them is `--all`, and `--limit 0` would change nothing.
        assert payload.total_jobs == len(jobs)
        assert payload.filtered_jobs == len(BATCH_ORDER)
        assert payload.hidden_jobs == 0

    def test_an_active_job_with_nothing_in_flight_is_not_live(self):
        """A job container stays open all day after its tasks have finished."""
        jobs = [_job("poker-20260901-mem", state="BatchJobState.ACTIVE", phase="finished")]
        assert select(jobs, show_all=False, limit=20).jobs == []


class TestCrashLoopIsVisible:
    """A retried task must not read like a healthy one.

    Batch reruns a failed task and the rerun is `running` like any other, so four
    consecutive OOM deaths rendered as one working task. Measured 09-27: four
    scores crash-looped for 4.7 hours while `jobs` showed them running, and only
    `tasks --run` (attempt numbers) exposed it.
    """

    def test_a_retried_task_renders_its_attempt_number(self, capsys):
        from src.interfaces.commands.jobs import JobsPayload, render

        looping = BatchTask(
            task="score-pcs-9101-600",
            job="poker-20260927",
            state=None,
            phase="running",
            retry_count=3,
        )
        first = BatchTask(
            task="score-pcs-9101-800", job="poker-20260927", state=None, phase="running"
        )
        render(
            JobsPayload(
                jobs=[
                    Job(job="poker-20260927", state="BatchJobState.ACTIVE", tasks=[looping, first])
                ],
                total_jobs=1,
            )
        )
        out = capsys.readouterr().out
        assert "ATTEMPT 4" in out, "a crash-looping task must say which attempt it is on"
        # The first attempt stays quiet, or every line carries noise and the
        # signal stops being one.
        assert "score-pcs-9101-800  ATTEMPT" not in out


class TestAutoscaleBlindness:
    """Batch's counters can miss queued tasks, and the pool then never scales.

    Measured 10-19: two tasks requeued after node deaths read `active` while
    `get_job_task_counts` kept them `completed`. `$PendingTasks` read 0 and
    `train` sat at zero nodes for 12 hours with work queued.
    """

    @staticmethod
    def _blind_job() -> Job:
        tasks = [
            BatchTask(task="stuck", job="poker-20261018", state=None, phase="queued"),
            BatchTask(task="busy", job="poker-20261018", state=None, phase="running"),
        ]
        return Job(job="poker-20261018", state="BatchJobState.ACTIVE", tasks=tasks, counted=1)

    def test_the_gap_is_rendered(self, capsys):
        from src.interfaces.commands.jobs import JobsPayload, render, unseen

        job = self._blind_job()
        assert unseen(job) == 1
        render(JobsPayload(jobs=[job], total_jobs=1))
        assert "autoscale cannot see 1" in capsys.readouterr().out

    def test_requeue_touches_only_the_queued_tasks_of_a_blind_job(self, monkeypatch):
        import argparse

        from src.interfaces.commands import jobs

        seen = Job(
            job="poker-20261019",
            state="BatchJobState.ACTIVE",
            tasks=[BatchTask(task="fine", job="poker-20261019", state=None, phase="queued")],
            counted=1,
        )
        requeued: list[str] = []
        monkeypatch.setattr(jobs.CloudConfig, "load", staticmethod(lambda: None))
        monkeypatch.setattr(jobs.batch, "client", lambda _config: None)
        monkeypatch.setattr(jobs.batch, "list_jobs", lambda _client: [])
        monkeypatch.setattr(jobs.batch, "attach_tasks", lambda *_a, **_k: [self._blind_job(), seen])
        monkeypatch.setattr(jobs.batch, "attach_counts", lambda _c, listed, **_k: listed)
        monkeypatch.setattr(
            jobs.batch, "requeue_task", lambda _c, _job, task: requeued.append(task)
        )
        args = argparse.Namespace(all=False, limit=20, requeue_unseen=True)
        payload = jobs.run(args)
        assert requeued == ["stuck"]
        assert payload.requeued == ["stuck"]
