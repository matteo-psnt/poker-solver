"""The two properties the sampling profiler has to have.

It cannot be validated here beyond these: `py-spy` needs root on macOS and
ptrace on Linux, there is no CI, and the attach path only exists on a node.
What IS testable is the pair that decides whether it is safe to leave armed —
it does nothing unless asked, and nothing it does can cost a task its exit code.
Everything else about it is unproven until a node has run one.
"""

from __future__ import annotations

import threading

import pytest

from src.shared.cloudtask.node import process, profile
from tests.shared.cloudtask.node.conftest import python


class _Entry:
    """What `Path.iterdir` yields, as `python_worker` reads it."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def profile_dir(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    return directory


class _Container:
    """The diagnostics container as `take_request` uses it: names to bodies."""

    SAS = "https://acct.blob.core.windows.net/diagnostics?sig=x"

    def __init__(self, monkeypatch, **objects: bytes) -> None:
        self.objects = dict(objects)
        monkeypatch.setattr(
            profile.blobstore, "read_object", lambda _s, name: self.objects.get(name)
        )
        monkeypatch.setattr(profile.blobstore, "exists", lambda _s, name: name in self.objects)
        monkeypatch.setattr(
            profile.blobstore,
            "put_bytes",
            lambda _s, name, body: self.objects.__setitem__(name, body) or len(body),
        )


class TestItStaysOffUntilAsked:
    def test_no_request_is_no_profile(self, monkeypatch):
        _Container(monkeypatch)
        assert profile.take_request("task-1", _Container.SAS) is None

    def test_a_request_fires_once(self, monkeypatch):
        """Marked served BEFORE the recording, so a refused ptrace does not
        leave a request that re-profiles every poll for six hours. MARKED, not
        deleted: no task SAS carries `delete`."""
        _Container(monkeypatch, **{f"task-1{profile.REQUEST_SUFFIX}": b"45"})

        assert profile.take_request("task-1", _Container.SAS) == 45
        assert profile.take_request("task-1", _Container.SAS) is None

    def test_an_empty_request_still_profiles(self, monkeypatch):
        """The intent is in the request existing. `touch` is a reasonable way
        to ask, and refusing over the contents would be the least useful
        reading."""
        _Container(monkeypatch, **{f"task-1{profile.REQUEST_SUFFIX}": b""})

        assert profile.take_request("task-1", _Container.SAS) == profile.DEFAULT_SECONDS

    def test_a_request_for_another_task_is_not_this_one(self, monkeypatch):
        _Container(monkeypatch, **{f"task-2{profile.REQUEST_SUFFIX}": b"30"})

        assert profile.take_request("task-1", _Container.SAS) is None

    def test_an_absurd_duration_is_clamped_not_honoured(self, monkeypatch):
        """A profiler holding a node for an hour is worse than no profiler."""
        _Container(monkeypatch, **{f"task-1{profile.REQUEST_SUFFIX}": b"999999"})

        assert profile.take_request("task-1", _Container.SAS) == profile.MAX_SECONDS


class TestItProfilesTheProcessDOINGTheWork:
    """The pid walk, pinned against the two process tables that fooled it.

    Both wrong picks produced a VALID profile of the wrong process, which is
    the failure a flamegraph cannot show you:

    - deepest-first gave 30s of `multiprocessing.resource_tracker`, a
      bookkeeping process that sits in `select()` and is spawned as deep as the
      workers;
    - cumulative CPU gave the COORDINATOR, 100% in `connection._recv`, because
      it had built 45M rows of shared arrays before forking and so led on total
      ticks while waiting on the workers it had started.
    """

    def _tables(self, monkeypatch, before, after) -> None:
        """Two successive readings of /proc, with no real sleep between them."""
        readings = iter([before, after])
        monkeypatch.setattr(profile, "_process_table", lambda: next(readings))
        monkeypatch.setattr(profile.time, "sleep", lambda _seconds: None)

    def test_the_worker_burning_cpu_beats_the_parent_that_burnt_it_earlier(self, monkeypatch):
        coordinator = profile.Proc("python3.13", 100, 90_000)
        self._tables(
            monkeypatch,
            {
                100: profile.Proc("uv", 1, 0),
                101: coordinator,
                102: profile.Proc("python3.13", 101, 10),
                103: profile.Proc("python3.13", 101, 0),
            },
            {
                100: profile.Proc("uv", 1, 0),
                101: coordinator,  # blocked in _recv; not another tick
                102: profile.Proc("python3.13", 101, 210),  # a worker, running
                103: profile.Proc("python3.13", 101, 0),  # resource_tracker
            },
        )

        assert profile.python_worker(100) == 102

    def test_a_stalled_run_still_gets_profiled(self, monkeypatch):
        """Nothing moving is the best reason to ask for a profile. Refusing one
        answers "why has it stopped" with "no interpreter"."""
        table = {
            100: profile.Proc("uv", 1, 0),
            101: profile.Proc("python3.13", 100, 5_000),
            102: profile.Proc("python3.13", 101, 900),
        }
        self._tables(monkeypatch, table, table)

        assert profile.python_worker(100) == 101

    def test_nothing_started_yet_is_not_a_pick(self, monkeypatch):
        table = {100: profile.Proc("uv", 1, 0), 101: profile.Proc("python3.13", 100, 0)}
        self._tables(monkeypatch, table, table)

        assert profile.python_worker(100) is None

    def test_a_busy_interpreter_outside_this_task_is_not_ours(self, monkeypatch):
        """Anything the start task left behind is an interpreter on this box
        that this task must not read."""
        self._tables(
            monkeypatch,
            {
                100: profile.Proc("uv", 1, 0),
                101: profile.Proc("python3.13", 100, 5),
                900: profile.Proc("python3.13", 1, 10),
            },
            {
                100: profile.Proc("uv", 1, 0),
                101: profile.Proc("python3.13", 100, 60),
                900: profile.Proc("python3.13", 1, 999_999),
            },
        )

        assert profile.python_worker(100) == 101


class TestItCannotCostATaskItsExitCode:
    def test_arming_it_does_not_change_what_the_child_reported(self, paths, log, profile_dir):
        """The whole feature is opt-in observation. If the number `run_guarded`
        returns moves, a wrong terminal cause follows it into the task record,
        and that is permanent -- it suppresses reconciliation."""
        argv = python("import sys", "sys.exit(3)")

        code = process.run_guarded(
            argv, cwd=paths.work.parent, timeout=5, log=log, profile_dir=profile_dir
        )

        assert code == 3

    def test_a_profiler_that_raises_is_swallowed(self, profile_dir, monkeypatch):
        """Every failure mode here is a property of the box -- ptrace refused,
        a slow store, no `py-spy` -- and none is a reason to lose the work."""
        _Container(monkeypatch, **{f"task-1{profile.REQUEST_SUFFIX}": b"1"})
        monkeypatch.setattr(profile, "POLL_SECONDS", 0.01)

        def _explode(*_args, **_kwargs):
            raise RuntimeError("ptrace: Operation not permitted")

        monkeypatch.setattr(profile, "record", _explode)

        served = threading.Event()
        monkeypatch.setattr(profile, "python_worker", lambda _root: served.set() or 1234)

        stop = threading.Event()
        thread = threading.Thread(
            target=profile.watch,
            args=(1, profile_dir, "task-1", print, stop, _Container.SAS),
            daemon=True,
        )
        thread.start()
        assert served.wait(timeout=3), "the request was never picked up"
        stop.set()
        thread.join(timeout=3)

        assert not thread.is_alive(), "the profiler thread died on an exception"

    def test_a_task_with_no_credential_never_profiles(self, monkeypatch):
        """A dispatch always mints one; a task run without is not a path that
        should reach the container and guess."""
        _Container(monkeypatch, **{f"task-1{profile.REQUEST_SUFFIX}": b"1"})
        assert profile.take_request("task-1", "") is None
