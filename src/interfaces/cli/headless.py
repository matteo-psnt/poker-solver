"""Non-interactive (headless) entrypoint for training and evaluation.

Unlike the questionary menu in :mod:`src.interfaces.cli.app`, every operation
here is fully specified by flags and emits a machine-readable summary. This is
the surface scripts and cloud jobs use, so a cloud job is a shell invocation of
this module rather than a provider-specific reimplementation.

The subcommands themselves live in :mod:`src.interfaces.cli.commands`, one
module each. This file only wires them up and decides how output is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

from src.interfaces.cli.commands import COMMANDS
from src.interfaces.errors import CommandError
from src.shared.jsonio import json_default
from src.shared.log import configure_logging, pin_level_for_children


def build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI from the command registry."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Emit the result as JSON on stdout.")
    common.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help=(
            "Override the run config's system.log_level. Exported to spawned "
            "workers, so a whole run answers to one setting."
        ),
    )

    parser = argparse.ArgumentParser(
        prog="poker-solver-run",
        description="Headless training/evaluation entrypoint for scripts and cloud runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = sub.add_parser(command.name, parents=[common], help=command.help)
        command.add_arguments(subparser)
        # The Command itself rides on the namespace, so dispatch and rendering
        # both come from one object and cannot disagree about which they are.
        subparser.set_defaults(command_impl=command)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    command = args.command_impl
    if args.log_level:
        # Into the environment, not just this logger: spawned workers build
        # their level from the run config, and the flag must outrank it there.
        pin_level_for_children(args.log_level)
        configure_logging(args.log_level)
    try:
        if args.json:
            # Library layers log to stderr, but third-party writers (numba, zarr)
            # can still print to stdout; redirect so the JSON blob is the ONLY
            # thing on stdout and machine consumers can parse it directly.
            with contextlib.redirect_stdout(sys.stderr):
                payload = command.run(args)
            print(json.dumps(payload, indent=2, default=json_default))
        else:
            payload = command.run(args)
            command.render(payload)
    except CommandError as error:
        # This is where the command line puts back what the core no longer
        # assumes. A bad request used to `raise SystemExit(msg)` from wherever
        # it was detected, which printed to stderr and exited 1; that is
        # reproduced exactly here, and only here, so no other surface inherits
        # it. Anything that is NOT a CommandError still tracebacks -- a bug
        # should look like one.
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
