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

from src.interfaces import run_names
from src.interfaces.errors import CommandError
from src.pipeline.evaluation.ledger import rebuild_ledger


@dataclass(frozen=True)
class Command:
    """One `poker-solver` subcommand."""

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
    """Resolve a run identifier (name under ``runs_dir``) or an explicit path.

    A FRAGMENT resolves too, the way a git short hash does. Run ids are long,
    share a prefix, and differ only at the end -- ``run-production-025433-1095``
    -- so the piece that actually identifies one is usually its tail. Matching
    anywhere in the name rather than only at the front is what makes ``runinfo
    1095`` work, and typing the whole id was the single most tedious thing
    about every reader command.

    Ambiguity is an error that NAMES the candidates: silently taking the first
    match would answer a question about a different run than the one asked
    about, and every reader here is used to make a decision.
    """
    """Empty is not a run
    -------------------
    ``Path("")`` is ``PosixPath(".")`` and ``.is_dir()`` is True, so an empty
    identifier used to resolve to the CURRENT DIRECTORY and be returned as a
    run. Nothing downstream could tell that from a real answer: the caller's
    ``run_dir.is_dir()`` check passes, the refusal this function exists to make
    never happens, and the failure surfaces a minute later inside the loader as
    a missing checkpoint.

    It is reachable from the systemd unit on the blueprint host, whose
    ``ExecStart`` interpolates ``--run ${RUN}`` from an env file that ships with
    ``RUN=`` empty. There it resolves to ``WorkingDirectory``, i.e.
    ``/mnt/work/code`` -- the code checkout, offered up as a trained run.
    """
    if not run.strip():
        raise CommandError("No run given: --run needs a run id, a fragment of one, or a path.")

    as_path = Path(run)
    if as_path.is_dir():
        return as_path
    root = Path(runs_dir)
    exact = root / run
    if exact.is_dir():
        return exact
    names = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    matches = run_names.matching(run, names)
    if len(matches) == 1:
        return root / matches[0]
    if matches:
        raise CommandError(run_names.ambiguous_message(run, matches))
    raise CommandError(f"Run not found: '{run}' (looked at {as_path} and {exact})")


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


@contextmanager
def records_root(args: argparse.Namespace) -> Iterator[Path]:
    """The published record, materialised into a temporary tree.

    There is no longer a ``--source`` to choose. Every reader answers against
    the share, because the share is the only place a run exists: work runs on
    the pool, the node publishes, and nothing on this machine is a source of
    truth about it. A local copy could only ever be a stale second answer to a
    question the share can already answer -- measured at 2.9s for `progress`
    and 3.7s for `ledger`, which is what made this practical.

    The tree is removed on exit; the reader itself stays ordinary local-path
    code and never learns that Azure exists.
    """
    # Imported here rather than at module scope: this is the one place the
    # command layer needs the Azure SDK, and hoisting it would make every
    # `--help` pay for importing it.
    from src.interfaces.cloud.store.workspace import share_records

    with share_records(run=getattr(args, "run", None) or None) as root:
        yield root


def ledger_for(args: argparse.Namespace, root: Path) -> Path:  # noqa: ARG001
    """The eval index, REBUILT from the published documents.

    There is deliberately no ledger FILE on the share: a second writable file
    on a share with no atomic append is the contention the per-run records were
    introduced to remove. Each published document carries its own provenance,
    so the index is derived on demand instead of stored.
    """
    derived = root / "eval_ledger.jsonl"
    rebuild_ledger(root, derived)
    return derived
