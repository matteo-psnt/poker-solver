"""The sweep's command line survives the crossing to the node."""

from __future__ import annotations

import argparse

from src.interfaces.commands import migrate_checkpoints, submit_migrate


def _flags(**over):
    args = argparse.Namespace(
        runs=None,
        pool=None,
        timeout="6h",
    )
    for key, value in over.items():
        setattr(args, key, value)
    return submit_migrate._flags(args)


def _parse(flags):
    """Parse on the node side, as `migrate-checkpoints` will."""
    parser = argparse.ArgumentParser()
    migrate_checkpoints.add_arguments(parser)
    return parser.parse_args(list(flags))


class TestTheFlagsReachTheSweepIntact:
    """Emitted here, parsed there. A flag that does not survive is a sweep
    doing something other than what was asked, silently."""

    def test_several_runs_arrive_as_several_runs(self):
        """`--runs` takes `nargs="*"`, so a REPEATED flag keeps only the last:
        `--runs a --runs b` parses to `["b"]` and the sweep would migrate one
        run of two and report success."""
        parsed = _parse(_flags(runs=["run-a", "run-b", "run-c"]))
        assert parsed.runs == ["run-a", "run-b", "run-c"]

    def test_nothing_asked_for_is_nothing_sent(self):
        assert _flags() == ()
        parsed = _parse(())
        assert parsed.runs is None

    def test_every_flag_the_sweep_takes_can_be_asked_for(self):
        """The general form of the `--verify` gap: a flag declared on the node
        side that no dispatch can emit is a capability that does not exist.
        `--share` is the exception -- the node's own mount, never dispatched."""
        sweep = argparse.ArgumentParser()
        migrate_checkpoints.add_arguments(sweep)
        submitter = argparse.ArgumentParser()
        submit_migrate.add_arguments(submitter)

        emitted = {a.dest for a in submitter._actions}
        unreachable = {a.dest for a in sweep._actions} - emitted - {"help", "share"}
        assert not unreachable, f"the sweep accepts flags nothing can send: {sorted(unreachable)}"
