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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.interfaces import run_names, telemetry
from src.interfaces.errors import CommandError
from src.pipeline.evaluation.ledger import rebuild_ledger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pydantic import BaseModel

# pydantic is named in annotations only. A runtime import costs 0.10-0.13s on
# EVERY invocation, including the `--help` the lazy registry took from 3.15s to
# 0.18s; `from __future__ import annotations` keeps it out of the interpreter.

# Every command returns a MODEL. This was ``BaseModel | dict[str, Any]`` while
# they were converted one at a time; the union is gone, so a handler that
# returns a dict is now a type error at pre-commit rather than a shape nobody
# checks.
type Payload = BaseModel


@dataclass(frozen=True)
class Command:
    """One `poker-solver` subcommand.

    ``run`` returns a MODEL, not a dict, and that is the single declaration of
    what this command answers. `contract.py` imports it instead of restating it,
    `response_model` puts it in the OpenAPI schema, and `types.gen.ts` is
    generated from that -- so the shape exists once, in the module that produces
    it. Before this it existed three times (here as an untyped dict, in
    `contract.py` as a pydantic model, and in a hand-written `PAYLOADS` fixture),
    and renaming a REQUIRED field passed 1061 tests without a murmur.

    ``render`` takes the same model, so a renderer reading a field the payload
    does not carry is a type error rather than the runtime ``KeyError`` that
    made bundling parser, handler and renderer onto one object necessary.

    Serialisation is each surface's own business, at its own boundary:
    `jsonio.dumps` handles a model wherever it appears, `_compose` dumps so a
    view can join over plain data, and nothing here decides for them.

    ``render`` is typed ``Any`` rather than ``BaseModel`` on purpose: each
    renderer names its OWN payload type, and a ``Callable[[JobsPayload], None]``
    is not a ``Callable[[BaseModel], None]`` -- parameters are contravariant, and
    the checker is right to say so. The checking that matters happens inside each
    renderer, against the annotation it declares.
    """

    name: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], Payload]
    render: Callable[[Any], None]
    help: str = ""

    def declared(self) -> tuple[dict[str, Any], set[str]]:
        """This command's flag defaults, and which of them are required.

        Read from a parser built here rather than stored, so there is still ONE
        declaration of what a command accepts -- the `add_arguments` it already
        has. `_actions` is private and there is no public accessor for "every
        flag this parser knows"; the alternative is ``parse_args([])``, which
        cannot be used because it rejects a parser with any required flag (most
        of them) and exits the process on failure, which is the behaviour this
        seam exists to remove.
        """
        parser = argparse.ArgumentParser(prog=self.name, add_help=False)
        self.add_arguments(parser)
        actions = parser._actions  # noqa: SLF001
        return (
            {action.dest: action.default for action in actions},
            {action.dest for action in actions if action.required},
        )

    def execute(self, args: argparse.Namespace) -> Payload:
        """Run this command's handler. **The seam every surface goes through.**

        `invoke` is not that seam and cannot be: the command line builds its Namespace
        by parsing argv and calls the handler directly, so anything wrapped around
        `invoke` would see the console and miss the CLI. Kept separate from :attr:`run`
        so the handler stays an ordinary function a test can call; a guard test fails if
        a surface calls `run` directly and slips past this.

        Preparing the observation is itself guarded, which is not belt and braces:
        `asked_for` evaluates ``value != default`` on caller-supplied values, and one
        whose ``__ne__`` returns a non-bool -- a numpy array, and `invoke` accepts
        anything -- would raise out of here and fail a working command.
        """
        return self._observed(args) if telemetry.enabled() else self.run(args)

    def _observed(self, args: argparse.Namespace) -> Payload:
        """:meth:`execute`, with the observation attached."""
        try:
            asked = telemetry.asked_for(self.add_arguments, args, self.declared()[0])
        except Exception:  # noqa: BLE001 — never the reason a command fails
            asked = {}
        with telemetry.observe(self.name, asked):
            return self.run(args)

    def arguments(self, **overrides: Any) -> argparse.Namespace:
        """Build this command's arguments without a command line.

        Both halves of the check matter and neither is available to a caller
        assembling a Namespace by hand. An unknown key is rejected rather than
        ignored, because a silently-dropped ``limit=5`` looks like a command
        that ignores its own flag; a required flag left out is rejected here
        rather than surfacing as a ``None`` several frames into ``run``.
        """
        defaults, required = self.declared()

        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            known = ", ".join(sorted(defaults)) or "(none)"
            raise CommandError(f"{self.name}: no such argument {unknown}. Accepts: {known}")
        missing = sorted(required - set(overrides))
        if missing:
            raise CommandError(f"{self.name}: missing required argument(s) {missing}")
        return argparse.Namespace(**{**defaults, **overrides})

    def invoke(self, **overrides: Any) -> Payload:
        """Answer this command's question and return the payload, unrendered.

        The entry point for every surface that is not the command line. It
        raises :class:`CommandError` where the command line would have exited,
        so a caller polling several commands survives one of them failing.
        """
        return self.execute(self.arguments(**overrides))

    def invoke_as[T](self, model: type[T], **overrides: Any) -> T:
        """:meth:`invoke`, narrowed to the payload the caller expects.

        For a command built out of ANOTHER command -- `cost` is `tasks` plus
        arithmetic, `status` is three panels -- which needs the fields, not an
        opaque payload. :meth:`invoke` cannot be typed for them: it answers for
        every command, so its return is the union of every payload there is.

        The check is real rather than a cast. Asking `tasks` for its rows and
        getting something else back is a wiring mistake, and it should say so
        here rather than as an attribute error inside the arithmetic.
        """
        payload = self.invoke(**overrides)
        if not isinstance(payload, model):
            raise CommandError(
                f"{self.name} answered with {type(payload).__name__}, "
                f"but {model.__name__} was expected"
            )
        return payload


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
    # ``Path("")`` is ``PosixPath(".")`` and ``.is_dir()`` is True, so an empty
    # identifier would return the CURRENT DIRECTORY as a run. Reachable: the
    # blueprint host's systemd unit ships ``RUN=`` empty.
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

    Every reader answers against the share, because the share is the only place
    a run exists: work runs on the pool, the node publishes, and nothing on this
    machine is a source of truth about it. A local copy could only ever be a
    stale second answer to a question the share can already answer -- measured at
    2.9s for `progress` and 3.7s for `ledger`, which is what made this practical.

    The tree is removed on exit; the reader itself stays ordinary local-path
    code and never learns that Azure exists.
    """
    # Imported here rather than at module scope: this is the one place the
    # command layer needs the Azure SDK, and hoisting it would make every
    # `--help` pay for importing it.
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- see above
        share_records,
    )

    with share_records(run=getattr(args, "run", None) or None) as root:
        yield root


def ledger_for(root: Path) -> Path:
    """The eval index, REBUILT from the published documents.

    There is deliberately no ledger FILE on the share: a second writable file
    on a share with no atomic append is the contention the per-run records were
    introduced to remove. Each published document carries its own provenance,
    so the index is derived on demand instead of stored.
    """
    derived = root / "eval_ledger.jsonl"
    rebuild_ledger(root, derived)
    return derived
