"""THE entrypoint. Every operation is fully specified by flags.

There is no interactive counterpart any more. A questionary menu sat beside this
for a long time and was progressively hollowed out -- no local-compute door, no
config editor, nothing that read an abstraction -- until every item it still
offered was a wizard around flags this file already accepts. It was deleted
rather than maintained as a worse second way to say the same thing.

That leaves two surfaces, split by who is asking. This one is for anything that
can be scripted -- a cloud job is a shell invocation of this module rather than
a provider-specific reimplementation, and an agent drives it the same way. The
web console is for a human reading, and it reaches these same commands through
``Command.invoke`` rather than reimplementing them.

The subcommands live in :mod:`src.interfaces.commands`, one module each.
This file only wires them up and decides how output is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence

from src.interfaces.commands import BY_NAME, COMMANDS
from src.interfaces.errors import CommandError
from src.shared.jsonio import json_default
from src.shared.log import configure_logging, pin_level_for_children


def _named_command(argv: Sequence[str]) -> str | None:
    """Which subcommand ``argv`` asks for, without parsing it.

    The first token that IS a subcommand name. Every flag on this CLI belongs to
    a subcommand -- ``--json`` and ``--log-level`` are given to each one as
    parents -- so nothing before the subcommand takes a value that could be
    mistaken for it. Anything unrecognised returns None and argparse produces
    its own "invalid choice" listing every name, unchanged.
    """
    return next((token for token in argv if token in BY_NAME), None)


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    """Assemble the CLI from the command registry.

    Every subcommand is always LISTED -- ``--help`` is complete and the "invalid
    choice" message names them all. Only the one ``argv`` asks for has its flags
    built, because building them means importing that command's module, and
    importing all 27 cost 1.2s on every invocation to run one.

    ``argv=None`` builds all of them. That is what a test or an introspection
    tool wants, and it is also the definition the lazy path has to stay
    equivalent to -- ``tests/interfaces/commands/test_registry.py`` compares the
    two parsers flag by flag.
    """
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
        prog="poker-solver",
        description="Train, dispatch and read the record. Every operation is flag-driven.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    # Two different absences, and conflating them silently undid the whole
    # thing: NO argv means "build everything" (a caller introspecting the CLI),
    # while argv that names no subcommand -- `--help`, or nothing at all --
    # needs no command's flags built at all. It only needs the listing.
    wanted = None if argv is None else _named_command(argv)
    for ref in COMMANDS:
        subparser = sub.add_parser(ref.name, parents=[common], help=ref.help)
        if argv is not None and ref.name != wanted:
            continue
        command = ref.load()
        command.add_arguments(subparser)
        # The Command itself rides on the namespace, so dispatch and rendering
        # both come from one object and cannot disagree about which they are.
        subparser.set_defaults(command_impl=command)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    # Resolved before the parser is built, not by it: which subcommand was asked
    # for is what decides how much of the tool has to be imported.
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser(argv).parse_args(argv)
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
