"""Asking a running task for a stack sample, and finding the right file back.

The node names a profile `<task>.<attempt>.<n>.speedscope.json`, and neither the
attempt nor the counter is knowable here: a RETRIED task starts counting at 1
again, so `<task>.1.1...` can already be on the share before this call writes a
request. Claiming it would report a stale profile as the answer to a question
just asked -- the exact failure a flamegraph cannot show you, because a plausible
one from an earlier attempt looks like a result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.interfaces.commands import profile
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node import profile as node_profile

if TYPE_CHECKING:
    import argparse

TASK = "pcs-production-to100-pyspy-075303-23861"


def _args(**overrides) -> argparse.Namespace:
    return profile.COMMAND.arguments(**{"task": TASK, "no_wait": True, **overrides})


class _Share:
    """The share as this command uses it: a name -> body dict, and a listing."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(names or [])
        self.written: dict[str, str] = {}
        self.downloaded: list[str] = []
        # Appended to the listing after this many list calls, standing in for the
        # node serving the request while the command polls.
        self.appears: tuple[int, str] | None = None
        self.listings = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(profile.CloudConfig, "load", staticmethod(lambda: self))
        monkeypatch.setattr(profile.share, "share_client", lambda _config: self)
        monkeypatch.setattr(profile.share, "list_entries", self._list)
        monkeypatch.setattr(profile.share, "write_text", self._write)
        monkeypatch.setattr(profile.share, "download_file", self._download)
        monkeypatch.setattr(profile, "POLL_SECONDS", 0.01)

    # CloudConfig stand-in.
    share_name = "poker"

    def _list(self, _service, _share, _path, **_kwargs):
        self.listings += 1
        if self.appears and self.listings >= self.appears[0] and self.appears[1] not in self.names:
            self.names.append(self.appears[1])
        return [
            profile.share.ShareEntry(name=name, is_directory=False, size=1) for name in self.names
        ]

    def _write(self, _service, _share, path, body) -> None:
        self.written[path] = body

    def _download(self, _service, _share, path, destination) -> None:
        self.downloaded.append(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}")


class TestTheRequest:
    def test_it_writes_the_seconds_the_node_reads(self, monkeypatch):
        """The BODY is the duration. An empty one profiles for the default, so a
        request written wrong is a shorter profile rather than an error."""
        remote = _Share()
        remote.install(monkeypatch)

        profile.run(_args(seconds=45))

        path = f"{node_profile.PROFILES_DIRNAME}/{TASK}{node_profile.REQUEST_SUFFIX}"
        assert remote.written == {path: "45"}

    def test_without_a_task_it_is_refused_before_any_azure_call(self, monkeypatch):
        """`CloudConfig.load()` shells out to Terraform, so validating after it
        charges a round trip for a typo."""

        def _never() -> None:
            raise AssertionError("CloudConfig.load ran before the arguments were validated")

        monkeypatch.setattr(profile.CloudConfig, "load", staticmethod(_never))

        with pytest.raises(CommandError):
            profile.run(profile.COMMAND.arguments(task=None))


class TestTheWait:
    def test_it_returns_the_profile_the_node_just_wrote(self, monkeypatch):
        remote = _Share()
        remote.appears = (2, f"{TASK}.0.1{node_profile.PROFILE_SUFFIX}")
        remote.install(monkeypatch)

        payload = profile.run(_args(no_wait=False, out="/tmp/profiles"))

        assert payload.landed == f"{TASK}.0.1{node_profile.PROFILE_SUFFIX}"
        assert payload.downloaded is not None

    def test_a_profile_from_an_earlier_attempt_is_not_claimed(self, monkeypatch):
        """A retry restarts the counter, so the name this request will produce
        may ALREADY be on the share. Taking the newest name would answer with a
        profile recorded before anyone asked."""
        stale = f"{TASK}.0.1{node_profile.PROFILE_SUFFIX}"
        remote = _Share([stale])
        remote.install(monkeypatch)
        monkeypatch.setattr(profile, "SETTLE_SECONDS", 0.05)

        # `seconds` is part of the deadline, so a real duration would make every
        # not-served test wait out a profile nobody is recording.
        payload = profile.run(_args(no_wait=False, seconds=0))

        assert payload.landed is None
        assert not remote.downloaded

    def test_another_tasks_profile_is_not_this_ones(self, monkeypatch):
        """Two arms run side by side on this pool, and both are armed."""
        remote = _Share()
        remote.appears = (2, f"other-task.0.1{node_profile.PROFILE_SUFFIX}")
        remote.install(monkeypatch)
        monkeypatch.setattr(profile, "SETTLE_SECONDS", 0.05)

        # `seconds` is part of the deadline, so a real duration would make every
        # not-served test wait out a profile nobody is recording.
        payload = profile.run(_args(no_wait=False, seconds=0))

        assert payload.landed is None

    def test_nothing_landing_says_where_the_reason_is(self, monkeypatch, capsys):
        """ptrace refused, py-spy unresolvable and a task that is not running all
        look identical here; only the node's log tells them apart."""
        remote = _Share()
        remote.install(monkeypatch)
        monkeypatch.setattr(profile, "SETTLE_SECONDS", 0.05)

        profile.render(profile.run(_args(no_wait=False, seconds=0)))

        assert f"logs --task {TASK}" in capsys.readouterr().out
