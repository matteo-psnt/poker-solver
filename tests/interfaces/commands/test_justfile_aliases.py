"""The justfile's aliases are passthroughs, so they can silently outlive a command.

`just fetch` called `poker-solver fetch`, which was deleted along with the
local-storage path. Nothing noticed: a passthrough recipe is just a string, so
it kept appearing in `just --list` as though it were a thing you could do, and
failed only when someone typed it.

This is the cheapest fix for that whole class -- it costs one file read and
catches every future rename or deletion at the moment it happens rather than at
the moment someone reaches for the alias.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from src.interfaces.commands import COMMANDS

JUSTFILE = pathlib.Path(__file__).resolve().parents[3] / "justfile"

# The passthrough form: `uv run poker-solver <cmd>`. `{{args}}` is the
# `cli` escape hatch, whose subcommand comes from the caller and cannot be
# checked here.
INVOCATION = re.compile(r"uv run poker-solver\s+([a-z][a-z-]*)")

NAMES = sorted(command.name for command in COMMANDS)

"""Why an exact set and not a floor.

A recipe earns a place only by doing something the bare command does not:
`submit`, `score` and `repair-ladder` reshape positionals into flags, and
`serve` is invoked by the `console` recipes as the second half of build-then-
serve. Nine pure passthroughs were removed because they retyped a command
unchanged. A `>= n` floor caught a broken regex but not a passthrough creeping
back -- the failure that actually happens -- so the guard names all four.
"""
INVOKED_SUBCOMMANDS = ["repair-ladder", "score", "serve", "submit"]


def _invoked() -> list[str]:
    return sorted(set(INVOCATION.findall(JUSTFILE.read_text())))


def test_the_justfile_invokes_exactly_the_subcommands_it_should():
    """Catches a silently-broken regex AND a passthrough added back."""
    assert _invoked() == INVOKED_SUBCOMMANDS


@pytest.mark.parametrize("name", _invoked())
def test_every_aliased_subcommand_exists(name):
    assert name in NAMES, (
        f"the justfile has a `{name}` alias, but no such subcommand is registered. "
        f"Delete the recipe or fix the name. Registered: {NAMES}"
    )


def test_the_escape_hatch_is_present():
    """Without it the alias list grows a recipe per flag, which is how it
    drifted in the first place."""
    assert "cli *args:" in JUSTFILE.read_text()
