"""Addressing a run by a fragment of its id.

Ids are long, share a prefix and differ only at the END, so the piece a
person remembers is the tail -- `1095`, not `run-production-025433-1095`.
Typing the whole thing was the most tedious part of every reader command.
"""

from __future__ import annotations

from pathlib import Path

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
