"""A warm-started training leg, as the node will run it.

Seeding and training are ONE leg on purpose. As two, a Batch retry would re-run
the seeding step over a run that had already trained, laying the prior back on
top of real progress -- and the absolute-target contract that makes retries safe
everywhere else would not save it, because the damage happens before training
starts. ``train_static`` therefore ignores the prior whenever it is continuing,
and the node just passes the flags through.
"""

from __future__ import annotations

import pathlib

from src.shared.node import plan as node_plan

ENV = {
    "RUN_OP": "train",
    "RUN_CONFIG": "production",
    "RUN_TO": "5000000",
    "RUN_WARM_START_FROM": "vec-production-064428-20334",
    "RUN_WARM_START_WEIGHT": "1000",
    "AZ_BATCH_TASK_ID": "t1",
}


def _argv(**overrides: str) -> list[str]:
    return node_plan.parse_environment({**ENV, **overrides}).train_argv()


class TestWarmStartArgv:
    def test_the_prior_reaches_the_command_line(self):
        argv = _argv()
        assert argv[argv.index("--warm-start-from") + 1] == "vec-production-064428-20334"
        assert argv[argv.index("--warm-start-weight") + 1] == "1000"

    def test_an_unseeded_leg_carries_neither_flag(self):
        """A blank source must not become ``--warm-start-from ""``, which would
        send the trainer looking for a run directory named empty string."""
        argv = _argv(RUN_WARM_START_FROM="")
        assert "--warm-start-from" not in argv
        assert "--warm-start-weight" not in argv

    def test_the_weight_is_optional(self):
        """Absent weight falls to the service default rather than being passed
        as 0, which would claim a prior of no strength at all."""
        argv = _argv(RUN_WARM_START_WEIGHT="")
        assert "--warm-start-from" in argv
        assert "--warm-start-weight" not in argv

    def test_seeding_does_not_disturb_the_absolute_target(self):
        argv = _argv()
        assert argv[argv.index("--iterations") + 1] == "5000000"

    def test_workers_is_still_passed(self):
        """The scalar trainer needs it; only the vector kernel does not."""
        assert "--workers" in _argv()


class TestTheNodeFetchesThePrior:
    """A leg that seeds must bring the prior down first.

    The trainer resolves a bare run id under its runs directory, so if nothing
    fetched that run the path names an empty directory and the leg dies on a
    missing checkpoint -- on a node, after a snapshot upload and a pool spin-up.
    """

    def test_the_runner_fetches_a_named_prior(self):
        source = pathlib.Path("src/shared/node/runner.py").read_text()
        train = source.split("def _train(", 1)[1].split("\ndef ", 1)[0]
        assert "warm_start_from" in train, "the runner never looks at the prior"
        assert "fetch_current_rung" in train
