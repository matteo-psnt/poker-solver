"""Addressing a run by a fragment of its id.

Ids are long, share a prefix and differ only at the END, so the piece a
person remembers is the tail -- `1095`, not `run-production-025433-1095`.
Typing the whole thing was the most tedious part of every reader command.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.interfaces.commands._base import resolve_run_dir
from src.interfaces.errors import CommandError


class TestRunFragmentResolution:
    """Typing a whole run id was the most tedious thing about every reader.

    Ids are long, share a prefix and differ only at the END, so the piece a
    person remembers is the tail — `1095`, not `run-production-025433-1095`.
    """

    @staticmethod
    def _runs(tmp_path):
        for name in (
            "run-production-025433-1095",
            "run-ochs_dose_r100-105223-25247",
            "run-ochs_dose_r100-105241-16780",
        ):
            (tmp_path / name).mkdir()
        return str(tmp_path)

    def test_a_tail_fragment_resolves(self, tmp_path):
        root = self._runs(tmp_path)
        assert resolve_run_dir("1095", root).name == "run-production-025433-1095"

    def test_an_exact_id_still_wins_over_fragment_search(self, tmp_path):
        root = self._runs(tmp_path)
        assert resolve_run_dir("run-production-025433-1095", root).name == (
            "run-production-025433-1095"
        )

    def test_an_ambiguous_fragment_names_its_candidates(self, tmp_path):
        """Taking the first match would answer about a different run than asked."""
        root = self._runs(tmp_path)
        with pytest.raises(CommandError, match="matches 2 runs") as caught:
            resolve_run_dir("ochs", root)
        assert "105223-25247" in str(caught.value)
        assert "105241-16780" in str(caught.value)

    def test_a_fragment_with_glob_characters_is_a_literal(self, tmp_path):
        """`*` must not quietly match a run the caller never named."""
        root = self._runs(tmp_path)
        with pytest.raises(CommandError, match="Run not found"):
            resolve_run_dir("*", root)

    def test_no_match_is_still_a_refusal_not_a_crash(self, tmp_path):
        with pytest.raises(CommandError, match="Run not found"):
            resolve_run_dir("nope", self._runs(tmp_path))


class TestEmptyIsNotARun:
    """`Path("")` is `PosixPath(".")`, and `.is_dir()` is True.

    So an empty identifier resolved to the CURRENT DIRECTORY and was returned as
    a run. Nothing downstream could tell that from a real answer: the caller's
    own `run_dir.is_dir()` guard passes, the refusal never happens, and the
    failure surfaces a minute later inside the loader as a missing checkpoint.

    Reachable in production, not just in theory: the blueprint host's systemd
    unit interpolates `--run ${RUN}` from an env file that ships with `RUN=`
    empty, so it resolved to `WorkingDirectory` — `/mnt/work/code`, the code
    checkout, offered up as a trained run.
    """

    def test_an_empty_run_is_refused(self, tmp_path):
        with pytest.raises(CommandError, match="No run given"):
            resolve_run_dir("", str(tmp_path))

    def test_whitespace_is_refused_too(self, tmp_path):
        """systemd expands an unset variable to an empty word, and a shell that
        quotes it can hand over a space."""
        with pytest.raises(CommandError, match="No run given"):
            resolve_run_dir("   ", str(tmp_path))

    def test_it_does_not_quietly_become_the_working_directory(self, tmp_path, monkeypatch):
        """The actual production shape: cwd exists and is a directory, so the
        old code returned it."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(CommandError):
            resolve_run_dir("", str(tmp_path / "runs"))

    def test_an_explicit_dot_still_works(self, tmp_path, monkeypatch):
        """`.` is a deliberate path, unlike an empty string that got there by
        interpolating an unset variable."""
        monkeypatch.chdir(tmp_path)
        assert resolve_run_dir(".", str(tmp_path / "runs")) == Path()


class TestDispatchResolvesTheFragmentToo:
    """MEASURED failure: `score --run 15261` reached the node as `15261`.

    Readers resolve a fragment locally; dispatch did not, so the id travelled
    raw to a node that has no fragment matcher. It cost a snapshot upload, a
    node allocation and three retries before `FATAL no such run on the share:
    15261`. Resolving must happen before the payload is built.
    """

    PUBLISHED = (
        "run-train-production-to30M-eq1000-230841-15261",
        "run-train-production-to30M-w3000-114628-25986",
        "run-production-025433-1095",
    )

    @staticmethod
    def _share(monkeypatch, names):
        from src.interfaces.cloud.config import CloudConfig
        from src.interfaces.cloud.store import share, workspace

        monkeypatch.setattr(
            CloudConfig,
            "load",
            classmethod(
                lambda cls: SimpleNamespace(storage_account="a", share_name="s", share_key="k")
            ),
        )
        monkeypatch.setattr(share, "share_client", lambda config: object())
        monkeypatch.setattr(
            share,
            "list_entries",
            lambda service, share_name, path: [
                SimpleNamespace(name=name, is_directory=True) for name in names
            ],
        )
        return workspace

    def test_a_fragment_becomes_the_full_id_before_dispatch(self, monkeypatch):
        workspace = self._share(monkeypatch, self.PUBLISHED)
        assert workspace.resolve_published_run("15261") == self.PUBLISHED[0]

    def test_an_exact_id_is_returned_unchanged(self, monkeypatch):
        workspace = self._share(monkeypatch, self.PUBLISHED)
        assert workspace.resolve_published_run(self.PUBLISHED[1]) == self.PUBLISHED[1]

    def test_an_ambiguous_fragment_is_refused_here_not_on_a_node(self, monkeypatch):
        """The whole point: fail in the terminal, not after a pool spin-up."""
        workspace = self._share(monkeypatch, self.PUBLISHED)
        with pytest.raises(CommandError, match="matches 2 runs"):
            workspace.resolve_published_run("to30M")

    def test_an_unpublished_run_names_what_is_published(self, monkeypatch):
        workspace = self._share(monkeypatch, self.PUBLISHED)
        with pytest.raises(CommandError, match="is not published"):
            workspace.resolve_published_run("no-such-run")

    def test_score_puts_the_full_id_in_the_task_it_queues(self, monkeypatch):
        """The pin for the actual failure: what `score` SENDS, not what it can resolve."""
        import argparse

        from src.interfaces.cloud.tasks import dispatch
        from src.interfaces.commands import score

        self._share(monkeypatch, self.PUBLISHED)
        queued: list = []

        def fake(make_tasks, **_):
            queued.extend(make_tasks("snap-1"))
            return dispatch.Dispatched(op="score", code_snapshot="snap-1", job_id="j", tasks=["t"])

        monkeypatch.setattr(dispatch, "stage_and_queue", fake)
        payload = score.run(
            argparse.Namespace(
                run="15261",
                method="exact_br",
                at="",
                timeout=0,
                pool="train",
                flags=[],
                json=False,
            )
        )

        assert payload.run_id == self.PUBLISHED[0]
        assert [task.run_id for task in queued] == [self.PUBLISHED[0]]
