"""Every flag the node wrapper passes must be one the command actually declares.

This is the cheapest possible check for the most expensive possible typo. A
flag the node's CLI does not recognise is an argparse exit 2, and argparse
exits *before* doing any work -- so the failure costs a code snapshot upload, a
~3-minute pool spin-up and three node allocations (Batch retries) to discover
something a string comparison could have found. `score.py` documents that cost
in a comment about `--method`; it then happened for real to a sibling command.

The case that motivated it: the wrapper built one shared `ARGS` array that
unconditionally carried `--workers` -- correct for `train-static`, which
defaults to 1 and once trained single-threaded on a 16-vCPU node -- and a newer
`train-vector` branch splatted that same array into a command declaring no
`--workers`. Three identical attempts, each dead about four seconds in.

It used to regex `run_task.sh` for `uv run poker-solver` call sites and
trace shell array splats, because the argv existed only as shell. Now
`plan.TaskPlan` BUILDS the argv, so this asks the real parser to accept the real
list -- which also removes the regex's blind spot: a flag assembled from a
variable was invisible to it.

The wrapper lives in `src.shared`, which `.importlinter` forbids from importing
`src.interfaces`. That is the right way round: the node must not need the
command registry at runtime, and the check belongs here, on the client, where
the registry already is.
"""

from __future__ import annotations

import argparse
import tomllib

import pytest

from src.interfaces.commands import load_all
from src.shared import repo
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import handlers as node_handlers
from src.shared.cloudtask.node import plan as node_plan

# Added by `headless.build_parser` to every subcommand, so they are legal
# everywhere and appear in no command's own `add_arguments`.
COMMON_FLAGS = {"--json", "--log-level"}


def _declared(command_name: str) -> set[str]:
    command = next((c for c in load_all() if c.name == command_name), None)
    if command is None:
        return set()
    parser = argparse.ArgumentParser(prog=command_name, add_help=False)
    command.add_arguments(parser)
    return {option for action in parser._actions for option in action.option_strings}


def _flags(argv: list[str]) -> set[str]:
    return {token for token in argv if token.startswith("--")} - COMMON_FLAGS


def _undeclared(argv: list[str]) -> list[str]:
    return sorted(_flags(argv) - _declared(argv[0]))


def _plan(**overrides) -> node_plan.TaskPlan:
    defaults: dict = {"op": TaskName.TRAIN, "config": "production", "to": 1}
    return node_plan.TaskPlan(**{**defaults, **overrides})


# Every optional field, present and absent. The old regex saw flags on
# conditional branches because it read the whole file; a pure-function check
# only sees the argv it is handed, so the matrix has to supply the branches.
TRAIN_CASES = {
    "bare": _plan(),
    "tagged": _plan(experiment="exp-7", arm="control", parent="run-x"),
    "with-overrides": _plan(sets=("solver__dcfr=1.5", "system__note=two words")),
    "with-checkpoint-every": _plan(checkpoint_every=250_000),
    "continuing": _plan(run_id="run-a", checkpoint_every=1_000_000, experiment="e", arm="a"),
}

EVAL_CASES = {
    "at-a-rung": (_plan(op=TaskName.EVALUATE, run_id="run-a"), "1000000"),
    "latest": (_plan(op=TaskName.EVALUATE, run_id="run-a"), ""),
    "explicit-method": (
        _plan(op=TaskName.EVALUATE, run_id="run-a", eval_method="lookahead"),
        "500000",
    ),
}


@pytest.mark.parametrize("task", TRAIN_CASES.values(), ids=list(TRAIN_CASES))
def test_every_flag_a_training_task_passes_is_declared(task):
    argv = task.commands[0]
    assert _declared(argv[0]), f"`{argv[0]}` is not a registered command"
    assert not _undeclared(argv), _message(argv)


@pytest.mark.parametrize(("task", "rung"), EVAL_CASES.values(), ids=list(EVAL_CASES))
def test_every_flag_an_evaluate_task_passes_is_declared(task, rung):
    # `eval_flags` is the SUBMITTER's passthrough (`score --run r --
    # --br-flops 8`); its contents are unknowable here and are validated
    # against `evaluate` by `score`'s own method/flag plumbing.
    argv = task.commands[0]
    assert not _undeclared(argv), _message(argv)


def test_every_flag_a_precompute_task_passes_is_declared():
    argv = _plan(op=TaskName.PRECOMPUTE).commands[0]
    assert not _undeclared(argv), _message(argv)


def _message(argv: list[str]) -> str:
    return (
        f"the node wrapper passes {_undeclared(argv)} to `{argv[0]}`, which does not "
        f"declare it. argparse exits 2 before doing any work, so this costs a snapshot "
        f"upload, a pool spin-up and every Batch retry to discover. "
        f"Declared: {sorted(_declared(argv[0]))}"
    )


class TestTheCheckCatchesTheRealDefect:
    """A check that cannot fail is worse than none: it reads as coverage.

    `progress` stands in for the vector trainer -- a registered command that
    genuinely lacks `--workers`, so the fixture cannot rot into a tautology the
    way naming a real branch would.
    """

    def test_an_undeclared_flag_is_reported(self):
        assert _undeclared(["progress", "--run", "run-a", "--workers", "16"]) == ["--workers"]

    def test_a_declared_flag_is_not(self):
        assert _undeclared(["progress", "--run", "run-a"]) == []

    def test_the_common_flags_are_legal_everywhere(self):
        assert _undeclared(["progress", "--run", "run-a", "--json"]) == []


class TestTheWrapperInvokesACommandThatExists:
    """The flags are checked above; this checks the BINARY they are passed to.

    `handlers.py` hardcodes `["uv", "run", "<name>", *argv]` and `pyproject.toml`
    declares what `<name>` installs. Nothing tied them together, so renaming the
    script -- which is exactly what happened when `poker-solver-run` lost its
    suffix -- could leave the node calling a binary that no longer exists.

    That failure is invisible locally and expensive remotely: `uv run` on an
    unknown script fails after the code snapshot upload and the pool spin-up,
    with an error about the launcher rather than about the task. Same shape as
    the undeclared-flag defect this module was written for, one level up.
    """

    def _script_names(self) -> set[str]:
        pyproject = repo.ROOT / "pyproject.toml"
        return set(tomllib.loads(pyproject.read_text())["project"]["scripts"])

    def test_the_node_calls_a_script_the_project_installs(self):
        argv = node_handlers._cli(["progress", "--run", "run-a"])
        assert argv[:2] == ["uv", "run"], argv
        assert argv[2] in self._script_names(), (
            f"the node invokes `{argv[2]}`, which pyproject.toml does not install. "
            f"Declared scripts: {sorted(self._script_names())}"
        )

    def test_the_check_would_notice_a_rename(self):
        """Guards the guard: a typo'd name must not silently pass."""
        assert "poker-solver-run" not in self._script_names()
