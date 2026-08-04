"""One wording of a leg, shared across a layering boundary.

``spec`` builds the task id a leg is submitted under and ``leg_log``
describes the same leg when the record is read back. These live with the
definition rather than with either caller, so neither can drift alone.
"""

from __future__ import annotations

import pytest

from src.shared.describe import compact_count, flag_value


class TestCompactCount:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (150_000_000, "150M"),
            (1_000_000, "1M"),
            (1_500_000, "1.5M"),
            (250_000, "250k"),
            (7, "7"),
        ],
    )
    def test_a_target_reads_as_the_number_a_person_would_say(self, value, expected):
        assert compact_count(value) == expected


class TestFlagValue:
    def test_reads_both_spellings_of_a_passthrough_flag(self):
        assert flag_value(("--br-board-seed", "13"), "--br-board-seed") == "13"
        assert flag_value(("--br-board-seed=13",), "--br-board-seed") == "13"

    def test_a_flag_that_is_absent_or_has_no_value_is_empty(self):
        assert flag_value(("--br-flops", "4"), "--br-board-seed") == ""
        assert flag_value(("--br-board-seed",), "--br-board-seed") == ""
