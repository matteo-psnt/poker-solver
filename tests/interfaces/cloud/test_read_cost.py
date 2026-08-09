"""What these reads may NOT do, expressed as call counts.

Latency against Azure is invisible in a test and enormous in practice, so the
things that made a status screen take 22 seconds are pinned here as counts
instead: how many round trips a read is allowed to make, and which ones it must
not make at all. Every number below was a measured defect.

* `jobs` listed tasks for all 44 jobs at ~0.39s each to render the 2 that were
  active -- ~11s, and `tasks` paid it again for its reconcile.
* `tasks` downloaded 47 tiny task records one at a time: 9.1s of round trip.
* `--source share` fetched 37.17 MB of `keys-*/vocab.json` -- key tables from
  the DELETED dynamic backend, which nothing reads -- to answer questions that
  needed 0.06 MB of eval documents.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from azure.batch import BatchClient

from src.interfaces.cloud import batch, share, workspace
from src.interfaces.commands import jobs
from src.interfaces.errors import CommandError


def _as_client(fake: object) -> BatchClient:
    """Cast the stand-in to the SDK type these functions are declared against.

    The fake implements only the two or three methods they actually call --
    which is the point: a fake that had to satisfy the whole `BatchClient`
    surface would be a second Azure SDK, not a test.
    """
    return cast("BatchClient", fake)


class FakeBatch:
    """Counts calls. Nothing here talks to Azure."""

    def __init__(self, jobs_and_tasks: dict[str, tuple[str, list[str]]]):
        self._data = jobs_and_tasks
        self.list_tasks_calls: list[str] = []
        self.get_task_calls: list[tuple[str, str]] = []

    def list_jobs(self):
        return [_Obj(id=name, state=state) for name, (state, _) in self._data.items()]

    def list_tasks(self, job_id: str):
        self.list_tasks_calls.append(job_id)
        return [_task(t) for t in self._data[job_id][1]]

    def get_task(self, job_id: str, task_id: str):
        self.get_task_calls.append((job_id, task_id))
        return _task(task_id)


class _Obj:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def _task(task_id: str) -> _Obj:
    return _Obj(
        id=task_id,
        state="BatchTaskState.COMPLETED",
        creation_time=None,
        execution_info=_Obj(
            result=None, exit_code=0, failure_info=None, start_time=None, end_time=None
        ),
        node_info=None,
    )


ACCOUNT = {
    "job-old-1": ("BatchJobState.COMPLETED", ["t1"]),
    "job-old-2": ("BatchJobState.COMPLETED", ["t2"]),
    "job-live": ("BatchJobState.ACTIVE", ["t3"]),
}


class TestJobsDoesNotPayForHistory:
    def test_tasks_are_listed_only_for_jobs_that_can_survive_the_filter(self):
        """The default view shows live jobs, so the finished ones' tasks are
        fetched and then thrown away -- 42 of 44, at ~0.39s each."""
        client = FakeBatch(ACCOUNT)
        listed = batch.list_jobs(_as_client(client))
        batch.attach_tasks(_as_client(client), listed, want=jobs.is_active)

        assert client.list_tasks_calls == ["job-live"]

    def test_every_job_is_still_reported(self):
        """Cheaper must not mean less: `total_jobs` counts the whole account."""
        client = FakeBatch(ACCOUNT)
        attached = batch.attach_tasks(
            _as_client(client), batch.list_jobs(_as_client(client)), want=jobs.is_active
        )

        assert [job["job"] for job in attached] == list(ACCOUNT)
        assert [job["tasks"] for job in attached if job["job"] != "job-live"] == [[], []]

    def test_all_still_fetches_everything(self):
        client = FakeBatch(ACCOUNT)
        batch.attach_tasks(_as_client(client), batch.list_jobs(_as_client(client)), want=None)
        assert sorted(client.list_tasks_calls) == sorted(ACCOUNT)


class TestReconcileAsksAboutOpenQuestionsOnly:
    def test_one_targeted_call_per_unresolved_task(self):
        """Not one call per job in the account's history."""
        client = FakeBatch(ACCOUNT)
        record = batch.task_record(_as_client(client), "job-live", "t3")

        assert client.get_task_calls == [("job-live", "t3")]
        assert client.list_tasks_calls == []
        assert record is not None
        assert record["job"] == "job-live"


class TestDeadKeyTablesAreNotFetched:
    """`keys-<iter>/vocab.json` is the deleted dynamic backend's key table.

    Nothing in `src/` opens one, and they are permanently unreadable at HEAD --
    but they are JSON, so every metadata sync matched them on the suffix.
    """

    def test_a_key_table_directory_counts_as_checkpoint_data(self):
        assert share.is_snapshot_dir("keys-1080000")
        assert share.is_snapshot_path("run-a/keys-1080000/vocab.json")

    def test_zarr_snapshots_still_count(self):
        assert share.is_snapshot_dir("static-500000.zarr")
        assert share.is_snapshot_path("run-a/static-500000.zarr/0.0")

    def test_eval_records_are_still_record(self):
        """The regression this predicate already had once: depth is not the
        criterion, and `<run>/evals/record-*.json` is three deep."""
        assert not share.is_snapshot_path("run-a/evals/record-2026.json")
        assert not share.is_snapshot_path("run-a/.run.json")

    def test_a_run_named_like_a_key_table_is_not_swallowed(self):
        """The prefix must not be so broad it hides a real run directory."""
        assert not share.is_snapshot_dir("run-keys-experiment")
        assert not share.is_snapshot_dir("keysight")


class TestAPageCostsOneMaterialisation:
    """Five endpoints, one copy of the record -- pinned as a count.

    `/api/runs` and `/api/evals` each pulled the WHOLE record (12.4s), and a
    run's three detail panels (`runinfo`, `progress`, `curve`) pulled that one
    run three times over. Same few hundred kilobytes, five times per refresh,
    because a context manager that deletes its tree on exit cannot share it.

    This is the guard the fix needs: nothing about a new endpoint makes it
    obvious that it is about to add a whole sweep, and the latency it costs is
    invisible in a test.
    """

    @staticmethod
    def _count_pulls(monkeypatch) -> list[str | None]:
        pulls: list[str | None] = []

        def _fake(root, *, run):
            pulls.append(run)
            (root / "run-a").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(workspace, "_materialise", _fake)
        return pulls

    def test_every_reader_in_one_page_shares_one_tree(self, monkeypatch):
        pulls = self._count_pulls(monkeypatch)

        with workspace.shared_record_cache(ttl=60.0):
            for run in (None, None, "run-a", "run-a", "run-a"):
                with workspace.share_records(run=run):
                    pass

        assert pulls == [None], "a page's worth of readers pulled the record more than once"

    def test_a_scoped_reader_is_served_from_the_whole_record(self, monkeypatch):
        """Not a second, narrower pull: a scoped read is cheaper ONCE (3.7s
        against 12.4s) and more expensive three times, which is what a run's
        detail page does."""
        pulls = self._count_pulls(monkeypatch)

        with workspace.shared_record_cache(ttl=60.0), workspace.share_records(run="run-a"):
            pass

        assert pulls == [None]

    def test_an_unpublished_run_is_still_refused_by_name(self, monkeypatch):
        """The refusal a scoped pull used to make. Served from the whole tree
        there is no scoped pull to make it, and the reader's own "Run not found"
        names two local paths instead of what IS published."""
        self._count_pulls(monkeypatch)

        with (
            workspace.shared_record_cache(ttl=60.0),
            pytest.raises(CommandError, match="not published"),
            workspace.share_records(run="run-nope"),
        ):
            pass

    def test_the_command_line_still_pulls_per_read(self, monkeypatch):
        """Sharing is the server's concern. A one-shot reader gains nothing and
        would lose the guarantee it answers against the record as it is NOW."""
        pulls = self._count_pulls(monkeypatch)

        for _ in range(3):
            with workspace.share_records(run=None):
                pass

        assert pulls == [None, None, None]
