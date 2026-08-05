"""Every submission carries the submitting machine's git provenance.

A Batch node has no `.git` -- `share.SNAPSHOT_EXCLUDES` drops it from the code
tarball -- so `gitinfo`'s `git rev-parse` has nothing to answer from there. The
consequence went unnoticed for as long as training has been in the cloud:
`train_git_commit` and `eval_git_commit` were NULL on every cloud row. Measured
2026-08-04 on a probe task, where `ledger` printed an empty commit column.

The stamp goes on in `stage_and_queue` and not in any caller, because that is
what the module exists for -- `submit`, `score` and `repair-ladder` differ only
in the tasks they build, and a property every submission must have cannot be a
step three callers have to remember.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.cloud import dispatch, spec
from src.interfaces.errors import CommandError
from src.shared import gitinfo


@pytest.fixture(autouse=True)
def _cold_gitinfo():
    """Both readers are `lru_cache`d and other modules call them during a suite
    run; without clearing, these assert against a warmed value."""
    gitinfo.get_git_commit.cache_clear()
    gitinfo.is_git_dirty.cache_clear()
    yield
    gitinfo.get_git_commit.cache_clear()
    gitinfo.is_git_dirty.cache_clear()


def _task(**kwargs) -> spec.TaskSpec:
    return spec.TaskSpec(code_snapshot="snap", config="production", to=1000, **kwargs)


class TestStamping:
    def test_a_task_gains_the_commit_it_was_submitted_from(self, monkeypatch):
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")

        stamped = dispatch._stamped(_task())

        assert stamped.git_commit == "c" * 40
        assert stamped.git_dirty == "0"

    def test_a_dirty_tree_is_recorded_as_dirty(self, monkeypatch):
        """The snapshot is built from the WORKING TREE, not from HEAD, so on a
        dirty tree the commit is only the nearest ancestor. Saying so is the
        difference between "probably this code" and "this code, verified
        clean"."""
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "1")

        assert dispatch._stamped(_task()).git_dirty == "1"

    def test_it_reaches_the_node_environment(self, monkeypatch):
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")

        env = dispatch._stamped(_task()).environment()

        assert env["RUN_GIT_COMMIT"] == "c" * 40
        assert env["RUN_GIT_DIRTY"] == "0"

    def test_an_unstampable_machine_still_submits(self, monkeypatch):
        """No git, no checkout -- the task runs and records a null commit, which
        is exactly the status quo it is replacing. Refusing here would make a
        provenance nicety able to block work."""
        monkeypatch.delenv(gitinfo.COMMIT_ENV, raising=False)
        monkeypatch.delenv(gitinfo.DIRTY_ENV, raising=False)
        monkeypatch.setattr(gitinfo, "_run_git", lambda *a: None)

        stamped = dispatch._stamped(_task())

        assert stamped.git_commit == ""
        assert stamped.git_dirty == ""
        stamped.validate()

    def test_nothing_else_about_the_task_changes(self, monkeypatch):
        """`replace` on a frozen dataclass, so a typo in a field name raises
        rather than silently dropping an override."""
        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        original = _task(sets=("solver__dcfr=1.5",), experiment="exp-7", arm="control")

        stamped = dispatch._stamped(original)

        assert stamped.sets == ("solver__dcfr=1.5",)
        assert stamped.experiment == "exp-7"
        assert stamped.arm == "control"
        assert stamped.to == original.to


class TestTheNodeEnd:
    """`src.shared.node.plan` reads what `spec` writes."""

    def test_the_node_parses_what_the_submitter_stamped(self, monkeypatch):
        from src.shared.node import plan as node_plan

        monkeypatch.setenv(gitinfo.COMMIT_ENV, "c" * 40)
        monkeypatch.setenv(gitinfo.DIRTY_ENV, "0")
        env = dispatch._stamped(_task()).environment()

        parsed = node_plan.parse_environment(env)

        assert parsed.git_commit == "c" * 40
        assert parsed.git_dirty == "0"

    @pytest.mark.parametrize(
        ("dirty", "expected"),
        [("1", "DIRTY tree"), ("0", "clean tree"), ("", "tree state unknown")],
    )
    def test_the_wrapper_can_say_what_code_it_runs(self, dirty, expected):
        """The task log is the only place the answer appears while the task is
        still alive."""
        from src.shared.node import plan as node_plan

        task = node_plan.TaskPlan(
            op=node_plan.TRAIN, config="p", to=1, git_commit="c" * 40, git_dirty=dirty
        )
        assert expected in task.provenance
        assert task.provenance.startswith("cccccccccccc")

    def test_an_unstamped_task_says_so_rather_than_claiming_a_commit(self):
        from src.shared.node import plan as node_plan

        task = node_plan.TaskPlan(op=node_plan.TRAIN, config="p", to=1)
        assert "unknown" in task.provenance


class TestQueueLoop:
    """The loop that turns specs into queued tasks, which nothing covered.

    A rename once collapsed the spec and its generated id onto one name here, so
    `task.op` read an attribute off a `str`. Every submission would have raised;
    the whole suite still passed, and only `ty` objected.
    """

    @staticmethod
    def _stub(monkeypatch, calls):
        config = SimpleNamespace(share_name="share", pool_id="pool")
        monkeypatch.setattr(dispatch.CloudConfig, "load", staticmethod(lambda: config))
        monkeypatch.setattr(dispatch.share, "share_client", lambda _c: object())
        monkeypatch.setattr(dispatch.share, "publish_code_snapshot", lambda *a: "snap-1")
        monkeypatch.setattr(dispatch.batch, "client", lambda _c: object())
        monkeypatch.setattr(dispatch.batch, "ensure_job", lambda *a: "poker-20260805")

        def _submit(_client, job_id, task_id, task_spec, *, retries):
            calls.append({"job": job_id, "id": task_id, "spec": task_spec, "retries": retries})

        monkeypatch.setattr(dispatch.batch, "submit_task", _submit)

    def test_each_spec_is_queued_under_its_own_generated_id(self, monkeypatch):
        calls: list[dict] = []
        self._stub(monkeypatch, calls)

        payload = dispatch.stage_and_queue(lambda snap: [_task(), _task(run_id="run-b")])

        assert len(calls) == 2
        assert [c["id"] for c in calls] == payload["tasks"]
        assert len({c["id"] for c in calls}) == 2, "two submissions collided on one id"
        # The SPEC reaches Batch, not the id string -- the shape that broke.
        assert all(isinstance(c["spec"], spec.TaskSpec) for c in calls)
        assert all(isinstance(c["id"], str) for c in calls)

    def test_a_precompute_is_never_retried_but_everything_else_is(self, monkeypatch):
        calls: list[dict] = []
        self._stub(monkeypatch, calls)

        dispatch.stage_and_queue(
            lambda snap: [
                spec.TaskSpec(code_snapshot=snap, op=spec.PRECOMPUTE, config="production"),
                _task(),
            ]
        )

        assert calls[0]["retries"] == 0
        assert calls[1]["retries"] > 0

    def test_submitting_nothing_is_a_refusal_not_an_empty_job(self, monkeypatch):
        self._stub(monkeypatch, [])
        with pytest.raises(CommandError, match="Nothing to submit"):
            dispatch.stage_and_queue(lambda snap: [])
