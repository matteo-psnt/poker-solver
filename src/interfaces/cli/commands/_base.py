"""What a headless subcommand is, and the helpers they share.

A :class:`Command` carries its parser, its handler AND its renderer together.
That is the point of the type: a subcommand used to be assembled from an
``_add_*_parser`` in one module and a ``RENDERERS`` entry in another, so one
could exist without the other -- which is how ``checkpoint-profile`` came to
borrow the evaluate renderer and die on a missing key. Here it cannot be
registered without all three.

``run`` takes an :class:`argparse.Namespace`, which reads like a command line
leaking into the core, and would be one if the only way to build a Namespace
were to parse ``sys.argv``. :meth:`Command.invoke` is the other way: the parser
already carries every flag, its default and whether it is required, so it is a
usable schema for a caller that has no command line at all. That keeps ONE
declaration of what a command accepts -- a second surface that re-declared the
flags could disagree with the first, and the disagreement would show up as a
missing key at render time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.interfaces.errors import CommandError
from src.pipeline.evaluation.ledger import rebuild_ledger


@dataclass(frozen=True)
class Command:
    """One `poker-solver-run` subcommand."""

    name: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], dict[str, Any]]
    render: Callable[[dict[str, Any]], None]
    help: str = ""

    def arguments(self, **overrides: Any) -> argparse.Namespace:
        """Build this command's arguments without a command line.

        Both halves of the check matter and neither is available to a caller
        assembling a Namespace by hand. An unknown key is rejected rather than
        ignored, because a silently-dropped ``limit=5`` looks like a command
        that ignores its own flag; a required flag left out is rejected here
        rather than surfacing as a ``None`` several frames into ``run``.
        """
        parser = argparse.ArgumentParser(prog=self.name, add_help=False)
        self.add_arguments(parser)
        # `_actions` is private, and there is no public accessor for "every flag
        # this parser knows". The alternative is `parse_args([])`, which cannot
        # be used: it rejects a parser with any required flag, which is most of
        # them, and it exits the process on failure -- the behaviour this seam
        # exists to remove.
        actions = parser._actions  # noqa: SLF001
        defaults = {action.dest: action.default for action in actions}
        required = {action.dest for action in actions if action.required}

        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            known = ", ".join(sorted(defaults)) or "(none)"
            raise CommandError(f"{self.name}: no such argument {unknown}. Accepts: {known}")
        missing = sorted(required - set(overrides))
        if missing:
            raise CommandError(f"{self.name}: missing required argument(s) {missing}")
        return argparse.Namespace(**{**defaults, **overrides})

    def invoke(self, **overrides: Any) -> dict[str, Any]:
        """Answer this command's question and return the payload, unrendered.

        The entry point for every surface that is not the command line. It
        raises :class:`CommandError` where the command line would have exited,
        so a caller polling several commands survives one of them failing.
        """
        return self.run(self.arguments(**overrides))


def resolve_run_dir(run: str, runs_dir: str) -> Path:
    """Resolve a run identifier (name under ``runs_dir``) or an explicit path."""
    as_path = Path(run)
    if as_path.is_dir():
        return as_path
    candidate = Path(runs_dir) / run
    if candidate.is_dir():
        return candidate
    raise CommandError(f"Run not found: '{run}' (looked at {as_path} and {candidate})")


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set key__path=value`` into the config loader's override kwargs.

    Values go through JSON so ``1000``/``true``/``null`` arrive as the types the
    strict config models require; anything JSON rejects stays a plain string.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise CommandError(f"--set expects KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def add_source_argument(parser: argparse.ArgumentParser) -> None:
    """The `--source` flag every reading command shares.

    `local` reads whatever `fetch` last put in `--runs-dir`. `share` answers the
    question against the published record directly, without keeping a copy --
    see :mod:`src.interfaces.cloud.workspace` for why it materialises rather
    than reading in place.
    """
    parser.add_argument(
        "--source",
        default="local",
        choices=["local", "share"],
        help="local = the fetched copy in --runs-dir; share = the published record.",
    )


@contextmanager
def records_root(args: argparse.Namespace) -> Iterator[Path]:
    """The runs directory a reading command should work against.

    Under `--source share` this is a temporary tree holding the published JSON,
    removed on exit. The reader itself stays ordinary local-path code.
    """
    if getattr(args, "source", "local") == "share":
        # Imported here: only the share path needs the Azure SDK, and a local
        # read must not require cloud credentials to be configured.
        from src.interfaces.cloud.workspace import share_records

        with share_records(run=getattr(args, "run", None) or None) as root:
            yield root
    else:
        yield Path(args.runs_dir)


def ledger_for(args: argparse.Namespace, root: Path) -> Path:
    """Where the eval index lives for this source.

    Under `--source share` there is no shared ledger FILE, and deliberately so:
    a second writable file on a share with no atomic append is the contention
    the per-run records were introduced to remove. The index is instead REBUILT
    from the published documents, which every one of them can now support --
    each carries its own provenance.
    """
    if getattr(args, "source", "local") == "share":
        derived = root / "eval_ledger.jsonl"
        rebuild_ledger(root, derived)
        return derived
    return Path(args.ledger)
