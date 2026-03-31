"""Every flag `run_leg.sh` passes must be one the command actually declares.

This is the cheapest possible check for the most expensive possible typo. A
flag the node's CLI does not recognise is an argparse exit 2, and argparse
exits *before* doing any work -- so the failure costs a code snapshot upload, a
~3-minute pool spin-up and three node allocations (Batch retries) to discover
something a string comparison could have found. `score.py` documents that cost
in a comment about `--method`; it then happened for real to a sibling command.

The case that motivated this: the wrapper builds one shared `ARGS` array that
unconditionally carries `--workers` -- correct for `train-static`, which
defaults to 1 and once trained single-threaded on a 16-vCPU node -- and a newer
`train-vector` branch splatted that same array into a command that declares no
`--workers`. Three identical attempts, each dead about four seconds in.

What this can and cannot see: it reads the LITERAL flags the wrapper writes.
Passthrough arrays carry flags chosen by the submitter at runtime and are
skipped by name (see PASSTHROUGH_ARRAYS) -- those are checked where they are
built, in `spec.py`, not here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from src.interfaces.cli.commands import COMMANDS

WRAPPER = Path("infra/run_leg.sh")

# Added by `headless.build_parser` to every subcommand, so they are legal
# everywhere and appear in no command's own `add_arguments`.
COMMON_FLAGS = {"--json", "--log-level"}

# Arrays holding flags the SUBMITTER chose, not ones the wrapper wrote. `EXTRA`
# is `evaluate`'s passthrough (`score --run r -- --br-flops 8`); its contents
# are unknowable here and are validated against `evaluate` by `score`'s own
# method/flag plumbing.
PASSTHROUGH_ARRAYS = {"EXTRA"}

INVOCATION = re.compile(r"uv run poker-solver-run\s+([a-z-]+)(.*)")
ARRAY_DEF = re.compile(r"^\s*([A-Z_]+)\+?=\((.*)\)\s*$")
ARRAY_REF = re.compile(r'"\$\{([A-Z_]+)\[@\]\}"')
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations, so one shell command is one string."""
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _array_flags(lines: list[str]) -> dict[str, set[str]]:
    """Literal flags contributed to each array, anywhere in the file."""
    arrays: dict[str, set[str]] = {}
    for line in lines:
        match = ARRAY_DEF.match(line)
        if match:
            name, body = match.group(1), match.group(2)
            arrays.setdefault(name, set()).update(FLAG.findall(body))
    return arrays


def _declared(command_name: str) -> set[str]:
    command = next((c for c in COMMANDS if c.name == command_name), None)
    if command is None:
        return set()
    parser = argparse.ArgumentParser(prog=command_name, add_help=False)
    command.add_arguments(parser)
    return {option for action in parser._actions for option in action.option_strings}


def _invocations(text: str | None = None) -> list[tuple[str, set[str]]]:
    """(command, flags it is invoked with) for every wrapper call site."""
    lines = _logical_lines(WRAPPER.read_text() if text is None else text)
    arrays = _array_flags(lines)

    found: list[tuple[str, set[str]]] = []
    for line in lines:
        match = INVOCATION.search(line)
        if not match:
            continue
        name, rest = match.group(1), match.group(2)
        flags = set(FLAG.findall(rest))
        for array in ARRAY_REF.findall(rest):
            if array not in PASSTHROUGH_ARRAYS:
                flags |= arrays.get(array, set())
        found.append((name, flags - COMMON_FLAGS))
    return found


def test_the_wrapper_actually_invokes_something():
    """A parser that silently matches nothing would pass every test below."""
    invocations = _invocations()
    assert invocations, "found no `uv run poker-solver-run` call sites — parser is broken"
    assert {name for name, _ in invocations} <= {command.name for command in COMMANDS}


def _undeclared(command_name: str, flags: set[str]) -> list[str]:
    """The flags `command_name` is passed but does not accept."""
    return sorted(flags - _declared(command_name))


@pytest.mark.parametrize(("command_name", "flags"), _invocations(), ids=lambda v: str(v))
def test_every_flag_the_wrapper_passes_is_declared(command_name: str, flags: set[str]):
    declared = _declared(command_name)
    assert declared, f"`{command_name}` is invoked by the wrapper but is not a registered command"
    unknown = _undeclared(command_name, flags)
    assert not unknown, (
        f"run_leg.sh passes {unknown} to `{command_name}`, which does not declare it. "
        f"argparse exits 2 before doing any work, so this costs a snapshot upload, a "
        f"pool spin-up and every Batch retry to discover. Declared: {sorted(declared)}"
    )


# The defect this exists for, in miniature: a shared array carrying `--workers`
# splatted into a command that declares no such flag. `progress` stands in for
# the vector trainer -- it is a registered command that genuinely lacks it, so
# the fixture cannot rot into a tautology the way naming a real branch would.
BROKEN_WRAPPER = """
ARGS=()
ARGS+=(--workers "${RUN_WORKERS:-1}")
"${GUARD[@]}" uv run poker-solver-run progress \\
  --run "$RUN_ID" \\
  "${ARGS[@]}" 2>&1 | tee -a "$LEG_LOG"
"""


class TestTheCheckCatchesTheRealDefect:
    """A check that cannot fail is worse than none: it reads as coverage."""

    def test_the_flag_is_traced_through_the_shared_array(self):
        ((name, flags),) = _invocations(BROKEN_WRAPPER)
        assert name == "progress"
        assert "--workers" in flags, "the array splat was not followed"
        assert "--run" in flags, "flags on the invocation line itself were missed"

    def test_and_is_reported_as_undeclared(self):
        ((_, flags),) = _invocations(BROKEN_WRAPPER)
        assert _undeclared("progress", flags) == ["--workers"]

    def test_a_passthrough_array_is_not_blamed(self):
        """`EXTRA` holds submitter-chosen flags; guessing about them would make
        this fail on legitimate `score --run r -- --br-flops 8` usage."""
        wrapper = 'EXTRA+=(--not-a-real-flag)\nuv run poker-solver-run progress "${EXTRA[@]}"\n'
        ((_, flags),) = _invocations(wrapper)
        assert flags == set()
