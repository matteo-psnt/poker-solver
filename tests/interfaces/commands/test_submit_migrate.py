"""The sweep's command line survives the crossing to the node."""

from __future__ import annotations

import argparse

from src.interfaces.commands import migrate_checkpoints, submit_migrate


def _flags(**over):
    args = argparse.Namespace(runs=None, limit=0, verify=False, pool=None, timeout="6h")
    for key, value in over.items():
        setattr(args, key, value)
    return submit_migrate._flags(args)


class TestTheFlagsReachTheSweepIntact:
    """Emitted here, parsed there. A flag that does not survive is a sweep
    doing something other than what was asked, silently."""

    def test_several_runs_arrive_as_several_runs(self):
        """`--runs` takes `nargs="*"`, so a REPEATED flag keeps only the last:
        `--runs a --runs b` parses to `["b"]` and the sweep would migrate one
        run of two and report success."""
        parsed = _parse(_flags(runs=["run-a", "run-b", "run-c"]))
        assert parsed.runs == ["run-a", "run-b", "run-c"]

    def test_a_limit_survives(self):
        assert _parse(_flags(limit=5)).limit == 5

    def test_nothing_asked_for_is_nothing_sent(self):
        assert _flags() == ()
        parsed = _parse(())
        assert parsed.runs is None
        assert parsed.limit == 0

    def test_verify_survives(self):
        """It existed on the sweep and had no way to reach the node: the
        submitter never emitted it, so `--verify` silently ran a real sweep."""
        assert _parse(_flags(verify=True)).verify is True

    def test_both_together(self):
        parsed = _parse(_flags(runs=["run-a", "run-b"], limit=3))
        assert parsed.runs == ["run-a", "run-b"]
        assert parsed.limit == 3


def _parse(flags):
    parser = argparse.ArgumentParser()
    migrate_checkpoints.add_arguments(parser)
    return parser.parse_args(list(flags))
