"""Every flag `submit-vector` sends must be one `vector-sweep` accepts.

The two commands are separate parsers on opposite sides of a wire: `submit-vector`
builds an argv on a laptop and `vector-sweep` parses it on a node, hours later
and only if a node was free. So a flag added to one and not the other fails at
the far end of a queue -- the arm dies immediately, having waited for a node to
say `unrecognized arguments`, which is how `--train-boards` first behaved.

Asserting the pass-through *parses* rather than listing the flag names keeps
this from becoming a third declaration that can itself drift.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from src.engine.solver.vector import KERNELS
from src.interfaces.commands import submit_vector, vector_sweep

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vector-sweep")
    vector_sweep.add_arguments(parser)
    return parser


def _submit_args(**overrides: object) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="submit-vector")
    submit_vector.add_arguments(parser)
    args = parser.parse_args(["--abstraction", "buckets-F10T20R30-r200-ae5a7e66"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_every_dispatched_flag_is_one_the_node_accepts(kernel: str) -> None:
    """The argv a submit builds must parse on the node, for every kernel."""
    args = _submit_args(train_boards=8, score_boards=32, checkpoints="10,20", config="")
    argv = list(submit_vector._flags(args, kernel, derive=6000))

    parsed = _sweep_parser().parse_args([*argv, "--abstraction", "x"])

    assert parsed.kernel == kernel


def test_train_boards_rides_the_wire_only_when_asked() -> None:
    """Default stays off, so existing curves keep their in-sample meaning."""
    off = submit_vector._flags(_submit_args(train_boards=0), "hand-space", derive=6000)
    on = submit_vector._flags(_submit_args(train_boards=8), "hand-space", derive=6000)

    assert "--train-boards" not in off
    assert "--train-boards" in on
    assert on[on.index("--train-boards") + 1] == "8"


__all__: Sequence[str] = ()
