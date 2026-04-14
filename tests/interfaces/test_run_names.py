"""One definition of what a run fragment means, shared by two layers.

The cloud side decides which run to MATERIALISE from the share; the command
side then resolves a directory inside the tree it got. A disagreement between
them would pull one run and answer about another.
"""

from __future__ import annotations

import pytest

from src.interfaces import run_names

PUBLISHED = [
    "run-production-025433-1095",
    "run-ochs_dose_r100-105223-25247",
    "run-ochs_dose_r100-105241-16780",
]


class TestMatching:
    def test_a_tail_fragment_identifies_one_run(self):
        assert run_names.matching("1095", PUBLISHED) == ["run-production-025433-1095"]

    def test_an_ambiguous_fragment_returns_every_candidate(self):
        assert len(run_names.matching("ochs", PUBLISHED)) == 2

    def test_no_match_is_empty_rather_than_an_error(self):
        """The caller decides what a miss means; here it is just no candidates."""
        assert run_names.matching("nope", PUBLISHED) == []

    def test_an_exact_name_wins_over_the_longer_names_containing_it(self):
        """The case the local path never exercises.

        `resolve_run_dir` stats the directory first, so an exact id short-circuits
        before it ever asks here. On the SHARE there is nothing to stat — so if
        this rule were substring-only, publishing `run-a-2` alongside `run-a`
        would make the full id `run-a` permanently ambiguous and unusable.
        """
        assert run_names.matching("run-a", ["run-a", "run-a-2", "run-a-3"]) == ["run-a"]

    def test_matches_are_ordered_so_a_refusal_reads_the_same_way_twice(self):
        shuffled = list(reversed(PUBLISHED))
        assert run_names.matching("ochs", shuffled) == run_names.matching("ochs", PUBLISHED)


class TestAmbiguousMessage:
    def test_it_names_the_candidates_because_the_next_step_is_choosing_one(self):
        message = run_names.ambiguous_message("ochs", run_names.matching("ochs", PUBLISHED))
        assert "105223-25247" in message
        assert "105241-16780" in message

    @pytest.mark.parametrize("total", [7, 40])
    def test_a_long_candidate_list_is_summarised_not_dumped(self, total):
        matches = [f"run-x-{index}" for index in range(total)]
        message = run_names.ambiguous_message("x", matches)
        assert f"+{total - 6} more" in message
