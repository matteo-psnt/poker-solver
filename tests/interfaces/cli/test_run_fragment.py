"""Addressing a run by a fragment of its id.

Ids are long, share a prefix and differ only at the END, so the piece a
person remembers is the tail -- `1095`, not `run-production-025433-1095`.
Typing the whole thing was the most tedious part of every reader command.
"""

from __future__ import annotations

import pytest

from src.interfaces.cli.commands._base import resolve_run_dir
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
