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
import sys
import textwrap
from typing import TYPE_CHECKING

from src.interfaces import telemetry
from src.interfaces.commands import BY_NAME, COMMANDS, GROUPS
from src.interfaces.errors import CommandError
from src.shared import jsonio
from src.shared.log import configure_logging, pin_level_for_children

if TYPE_CHECKING:
    from collections.abc import Sequence


def _named_command(argv: Sequence[str]) -> str | None:
    """Which subcommand ``argv`` asks for, without parsing it.

    The first token that IS a subcommand name. Every flag on this CLI belongs to
    a subcommand -- ``--json`` and ``--log-level`` are given to each one as
    parents -- so nothing before the subcommand takes a value that could be
    mistaken for it. Anything unrecognised returns None and argparse produces
    its own "invalid choice" listing every name, unchanged.
    """
    return next((token for token in argv if token in BY_NAME), None)


# Wide enough for `abstraction-coupling`, the longest name, plus a gutter. Fixed
# rather than measured off the terminal: `--help` output that reflows with the
# window cannot be diffed, and the listing is what a reader compares run to run.
_NAME_COLUMN = 22
_HELP_WIDTH = 79 - _NAME_COLUMN - 4


def _listing() -> str:
    """The subcommands, under their group headings, for ``--help``'s epilog.

    argparse renders subcommands as one flat block and has no notion of a group
    among them, so the listing is built here and the block it would print is
    suppressed by giving `add_parser` no ``help``. That also takes the 32-name
    choice list out of ``usage:``, which `metavar` replaces with `<command>`.

    NOT a second registry: it reads `GROUPS`, and a command absent from there is
    absent from the CLI entirely.
    """
    lines = ["commands:"]
    for index, group in enumerate(GROUPS):
        lines.append(f"{'\n' if index else ''}  {group.title}")
        for ref in group.refs:
            wrapped = textwrap.wrap(ref.help, _HELP_WIDTH) or [""]
            lines.append(f"    {ref.name:<{_NAME_COLUMN}}{wrapped[0]}")
            # A continuation lines up under the first, so the name column stays
            # a column and a two-line entry does not read as two commands.
            lines.extend(f"    {'':<{_NAME_COLUMN}}{rest}" for rest in wrapped[1:])
    return "\n".join(lines)


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    """Assemble the CLI from the command registry.

    Every subcommand is always LISTED -- ``--help`` is complete and the "invalid
    choice" message names them all. Only the one ``argv`` asks for has its flags
    built, because building them means importing that command's module, and
    importing them all cost 1.2s on every invocation to run one.

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
        epilog=_listing(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # `metavar` so `usage:` reads `<command>` instead of every name in one
    # 900-character token. The "invalid choice" message still lists them all,
    # which is what `_named_command` leans on.
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    # Two different absences, and conflating them silently undid the whole
    # thing: NO argv means "build everything" (a caller introspecting the CLI),
    # while argv that names no subcommand -- `--help`, or nothing at all --
    # needs no command's flags built at all. It only needs the listing.
    wanted = None if argv is None else _named_command(argv)
    for ref in COMMANDS:
        # No `help=`: that is what makes argparse print its own flat block of
        # every subcommand, which `_listing` replaces with a grouped one.
        subparser = sub.add_parser(ref.name, parents=[common], description=ref.help)
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
    # `execute` rather than `run`: it is the seam both surfaces share, and the
    # only place a command's duration and outcome are observed. Calling the
    # handler directly here would make the command line -- the surface that runs
    # the expensive things -- the one absent from its own activity log.
    try:
        with telemetry.surface("cli"):
            if args.json:
                # Library layers log to stderr, but third-party writers (numba,
                # zarr) can still print to stdout; redirect so the JSON blob is
                # the ONLY thing on stdout and machine consumers can parse it.
                with contextlib.redirect_stdout(sys.stderr):
                    payload = command.execute(args)
                print(jsonio.dumps(payload, indent=2))
            else:
                payload = command.execute(args)
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
