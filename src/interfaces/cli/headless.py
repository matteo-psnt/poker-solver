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
from src.shared.jsonio import json_default
from src.shared.log import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI from the command registry."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Emit the result as JSON on stdout.")

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
    if args.json:
        # Library layers log to stderr, but third-party writers (numba, zarr) can
        # still print to stdout; redirect so the JSON blob is the ONLY thing on
        # stdout and machine consumers can parse it directly.
        with contextlib.redirect_stdout(sys.stderr):
            payload = command.run(args)
        print(json.dumps(payload, indent=2, default=json_default))
    else:
        payload = command.run(args)
        command.render(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
