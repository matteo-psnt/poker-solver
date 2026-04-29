"""Readers answer against the published record, never a local tree.

A reader that grew a `--runs-dir` would have two possible answers about a run,
and the local one is stale by construction: nothing on a laptop is a source of
truth about a run that trained in the cloud. The failure mode is that a stale
answer is indistinguishable from a fresh one at the point of reading, so this
cannot be caught by using the command — only by checking the flag surface.

`AGENTS.md` has asserted this rule in prose for weeks. Prose does not fail.
"""

from __future__ import annotations

import argparse

import pytest

from src.interfaces.commands import COMMANDS

# The three a node invokes. They write to /mnt/work before publishing, so a
# local directory is exactly what they need.
NODE_SIDE = frozenset({"train-static", "precompute", "evaluate"})

LOCAL_SOURCE_FLAG = "--runs-dir"


def _flags(command) -> set[str]:
    parser = argparse.ArgumentParser(prog=command.name, add_help=False)
    command.add_arguments(parser)
    return {option for action in parser._actions for option in action.option_strings}


def test_the_introspection_finds_flags():
    """A `_flags` that silently returned nothing would pass every test below."""
    seen = {flag for command in COMMANDS for flag in _flags(command)}
    assert len(seen) > 20, f"found almost no flags -- the introspection is broken: {seen}"


def test_the_node_side_commands_are_registered_under_these_names():
    """If a rename left NODE_SIDE naming nothing, the rule below would still
    pass while enforcing nothing at all."""
    registered = {command.name for command in COMMANDS}
    missing = NODE_SIDE - registered
    assert not missing, f"NODE_SIDE names commands that do not exist: {sorted(missing)}"


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.name)
def test_only_node_side_commands_take_a_local_runs_dir(command):
    if command.name in NODE_SIDE:
        return
    assert LOCAL_SOURCE_FLAG not in _flags(command), (
        f"`{command.name}` declares {LOCAL_SOURCE_FLAG}, so it can answer from a local "
        f"tree instead of the published record — a second answer that is stale by "
        f"construction. If this command genuinely runs on a node, add it to NODE_SIDE."
    )
