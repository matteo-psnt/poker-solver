"""A retired name is answered with where the question went.

~15 invocations of a retired command across 56 sessions, each answered with
argparse's 34-name "invalid choice" and followed by `--help` and a guess. The
table has to stay honest in both directions: a name in it must be dead, and a
name it points at must be alive, or it is the second registry it must not be.
"""

from __future__ import annotations

import re

import pytest

from src.interfaces.cli import headless
from src.interfaces.commands import BY_NAME, RETIRED

_NAMED = re.compile(r"`([a-z][a-z-]*)(?: |`)")


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_name_is_not_a_live_command(name):
    assert name not in BY_NAME, f"{name} is registered; drop it from RETIRED"


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_the_replacement_it_names_is_live(name):
    for replacement in _NAMED.findall(RETIRED[name]):
        assert replacement in BY_NAME, f"{name} points at `{replacement}`, which does not exist"


def test_the_cli_answers_with_the_line_and_exit_1(capsys):
    assert headless.main(["share-rm", "stats"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: share-rm was retired: ")
    assert "`runs`" in err


def test_a_live_command_is_never_mistaken_for_a_retired_one(monkeypatch):
    """The check keys on the first bare token, so `runs --limit 5` must reach
    argparse and not the table even though `--limit` precedes nothing retired."""
    assert headless._retired_command(["runs", "--limit", "5"]) is None
    assert headless._retired_command(["--json", "pool"]) == "pool"


class TestAnIdMisfireSaysWhereTheIdGoes:
    """Ten calls across the sessions typed the id as a positional or under the
    wrong flag; each was followed by a `--help`. The convention is one flag per
    kind of id, so the hint names the three flags rather than guessing one."""

    def test_a_positional_run_id_gets_the_hint(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["poker-solver", "progress", "run-x"])
        with pytest.raises(SystemExit):
            headless.build_parser(["progress", "run-x"]).parse_args(["progress", "run-x"])
        err = capsys.readouterr().err
        # argparse reports the missing `--run` before the stray positional.
        assert "the following arguments are required: --run" in err
        assert "ids are flags here: --run <run>, --task <task>, --job <job>" in err
        assert "`poker-solver progress --help`" in err

    def test_a_task_id_under_the_wrong_flag_gets_it_too(self, monkeypatch, capsys):
        argv = ["progress", "--task", "score-pcs-7245-seed17-161143-32442"]
        monkeypatch.setattr("sys.argv", ["poker-solver", *argv])
        with pytest.raises(SystemExit):
            headless.build_parser(argv).parse_args(argv)
        assert "ids are flags here" in capsys.readouterr().err

    def test_an_unrelated_error_does_not(self, monkeypatch, capsys):
        argv = ["progress", "--last", "many"]
        monkeypatch.setattr("sys.argv", ["poker-solver", *argv])
        with pytest.raises(SystemExit):
            headless.build_parser(argv).parse_args(argv)
        assert "ids are flags here" not in capsys.readouterr().err
