"""The registry names commands without importing them -- and must not lie.

Laziness buys 1.42s -> 0.23s on every invocation, and it buys it by holding a
COPY of each command's help line so `--help` can be answered without importing
27 modules. A copy can drift, and a drifted copy is invisible: the listing would
simply describe a command as something it no longer is.

So the two halves are pinned here. That the refs match their modules, and that
the parser built lazily is the same parser as the one built eagerly -- because
"only build the subcommand argv names" is an optimisation that is only correct
while those two agree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from src.interfaces.cli import headless
from src.interfaces.commands import BY_NAME, COMMANDS, load_all

# On every subcommand whether or not its own flags were built: `--json` and
# `--log-level` come from the shared parent parser, and argparse adds `-h`.
COMMON_FLAGS = {"--json", "--log-level", "-h", "--help"}


def _options(parser: argparse.ArgumentParser) -> dict[str, set[str]]:
    """Every subcommand's flags, by name, out of an assembled parser."""
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    found = {}
    for action in actions:
        for name, sub in action.choices.items():
            found[name] = {
                option for sub_action in sub._actions for option in sub_action.option_strings
            }
    return found


class TestTheRefsMatchTheirModules:
    """The one thing a lazy registry can get wrong that an eager one cannot."""

    def test_every_ref_loads(self):
        """The module is DERIVED from the name (hyphens to underscores) rather
        than stored, so this is what enforces the convention."""
        assert len(load_all()) == len(COMMANDS)

    @pytest.mark.parametrize("ref", COMMANDS, ids=lambda r: r.name)
    def test_the_ref_agrees_with_the_command_it_names(self, ref):
        command = ref.load()
        assert command.name == ref.name
        assert command.help == ref.help, (
            f"the registry describes '{ref.name}' as {ref.help!r} but the module "
            f"says {command.help!r} — `--help` is showing the stale one"
        )

    def test_names_are_unique(self):
        assert len(BY_NAME) == len(COMMANDS)


class TestTheLazyParserIsTheEagerParser:
    """Building only what argv names is correct exactly while these agree."""

    def test_the_full_parser_declares_every_command(self):
        assert set(_options(headless.build_parser())) == set(BY_NAME)

    @pytest.mark.parametrize("name", sorted(BY_NAME), ids=str)
    def test_one_command_gets_the_same_flags_either_way(self, name):
        eager = _options(headless.build_parser())[name]
        lazy = _options(headless.build_parser([name]))[name]
        assert lazy == eager

    def test_a_command_argv_does_not_name_is_left_unbuilt(self):
        """The saving itself. `jobs` must not cause `evaluate`'s import.

        An unbuilt subcommand is not flagless: `--json` and `--log-level` are
        given to every one as parents, so those are what remains when a command
        contributes nothing of its own.
        """
        built = _options(headless.build_parser(["jobs"]))
        assert built["jobs"] - COMMON_FLAGS, "the named command was not built"
        assert built["evaluate"] == COMMON_FLAGS, "an unrelated command was built anyway"

    def test_every_command_is_still_listed(self):
        """A subcommand left unbuilt must still be REACHABLE -- it appears in
        `--help` and in argparse's "invalid choice", and typing it works."""
        assert set(_options(headless.build_parser(["jobs"]))) == set(BY_NAME)


class TestNamingACommandImportsNothing:
    """Run in a subprocess: in-process, some earlier test has already imported
    half the tree into the worker and `sys.modules` would answer about that."""

    def test_importing_the_registry_imports_no_command_module(self):
        code = (
            "import sys, importlib;"
            "importlib.import_module('src.interfaces.commands');"
            "loaded=[m for m in sys.modules if m.startswith('src.interfaces.commands.')"
            " and not m.endswith('._base')];"
            "print(' '.join(sorted(loaded)))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == [], (
            "importing the registry imported command modules: " + done.stdout
        )

    def test_help_does_not_import_the_heavy_commands(self):
        """`--help` is the invocation with the least excuse to import anything,
        and it was importing the engine, numba and scipy to list 27 names."""
        code = (
            "import contextlib, io, sys;"
            "from src.interfaces.cli.headless import main;"
            "buffer = io.StringIO();"
            "err = contextlib.suppress(SystemExit);"
            "exec('with err, contextlib.redirect_stdout(buffer): main([\"--help\"])');"
            "print(' '.join(sorted({m.split('.')[0] for m in sys.modules}"
            " & {'scipy','sklearn','numba'})), file=sys.stderr)"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stderr.split() == [], f"`--help` imported: {done.stderr}"
