"""The ``RUN_*`` environment contract, declared once and read from both ends.

A task travels from the submitter to the node as environment variables. That
crossing used to be a hand-written pair -- ``spec.TaskSpec.environment`` built a
25-key dict and ``node.plan.parse_environment`` read the same 25 keys back --
with nothing but a test holding them together, and the test scanned
``plan.py``'s SOURCE TEXT for each key, so a key named only in a comment counted
as consumed.

The cost was not the typing. It was that widening the wire meant editing both
halves plus two dataclasses, and a feature already contorted to avoid it:
:class:`~src.shared.cloudtask.kinds.VectorSweepTask` overloads ``config`` to
mean an abstraction directory and ``arm`` to mean a kernel rather than add the
fields it wanted. When the transport starts shaping the domain model, the
transport is the thing to fix.

So :data:`KEYS` is the declaration and both ends derive from it. Adding a knob
is one entry here plus a field on whichever shapes carry it.

Why not just walk the dataclass fields
--------------------------------------
Because the two ends are genuinely different shapes, and that difference is
load-bearing rather than accidental:

* ``RUN_TIMEOUT`` is written as ``6h`` and read as ``timeout_seconds``.
* ``RUN_EVAL_AT`` is one rung at submit and ``eval_rungs``, a tuple, on the node
  -- one task scores the ladder the submitter named.
* ``workers`` is unset at submit and resolves to the node's own core count on
  arrival, because the node is the only place that knows it.

A field-name walk would have to special-case all three, so the codec is written
down per key instead. The attribute names are strings on purpose: this module
is imported by both ends, and ``shared -> interfaces`` is forbidden, so it can
name ``TaskSpec`` no more than it can import it.

Stdlib only, and the 3.10 floor: the node imports this before ``uv sync``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 6 * 3600
DEFAULT_EVAL_METHOD = "exact_br"

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class BadWireValueError(ValueError):
    """A payload the node cannot decode. Fatal: never silently an empty value."""


def parse_duration(raw: str | None) -> int:
    """``6h`` / ``90m`` / ``3600`` -> seconds.

    The wall-clock ceiling for the training process itself. The task-level
    ``maxWallClockTime`` (P1D) is not a backstop for a hang -- it is longer than
    any task is meant to run, so a wedged process bills a full node-day before
    Batch acts. One task proved it: training died, the process could not exit,
    and the task stayed ``running`` indefinitely.
    """
    text = (raw or "").strip().lower()
    if not text:
        return DEFAULT_TIMEOUT_SECONDS
    unit = _UNITS.get(text[-1])
    seconds = _int(text, 0) if unit is None else _int(text[:-1], 0) * unit
    # A non-positive ceiling would make the guard fire before the trainer has
    # started, which reads as a hang (124) rather than as the bad input it is.
    return seconds if seconds > 0 else DEFAULT_TIMEOUT_SECONDS


def _int(raw: str | None, default: int = 0) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> str:
    """An int that is written only when set, so an unused knob stays absent."""
    return str(value) if value else ""


def json_list(raw: str | None, name: str) -> tuple[str, ...]:
    """A JSON array, not a space-separated string.

    The elements are LISTS whose members may contain ``=``, a space or a
    newline -- a config override's value routinely does. Splitting on
    whitespace silently cut such a value in half.

    A malformed payload is FATAL, never an empty list: an override that was
    meant to be applied and was not is a run that answers a different question
    than the one asked.
    """
    if not raw:
        return ()
    try:
        decoded = json.loads(raw)
    except ValueError as error:
        raise BadWireValueError(f"could not decode {name}: {error}") from error
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise BadWireValueError(f"{name} must be a JSON array of strings, got: {raw!r}")
    return tuple(item for item in decoded if item)


def _dump_list(value: Any) -> str:
    return json.dumps(list(value or ()))


@dataclass(frozen=True)
class Key:
    """One environment variable, and how each end sees it.

    ``spec`` and ``plan`` are ATTRIBUTE NAMES, not values -- the submitter reads
    ``spec`` off its own dataclass and the node passes ``plan`` as a keyword.
    They differ wherever the two ends hold genuinely different shapes.
    """

    env: str
    spec: str
    plan: str
    encode: Callable[[Any], str] = str
    decode: Callable[[str], Any] = str


"""Every key on the wire. The order is the order they are emitted."""
KEYS: tuple[Key, ...] = (
    Key("CODE_SNAPSHOT", "code_snapshot", "code_snapshot"),
    Key("RUN_OP", "op", "op"),
    Key("RUN_CONFIG", "config", "config"),
    Key("RUN_TO", "to", "to", str, _int),
    Key("RUN_ID", "run_id", "run_id"),
    Key("RUN_EXPERIMENT", "experiment", "experiment"),
    Key("RUN_ARM", "arm", "arm"),
    Key("RUN_PARENT", "parent", "parent"),
    Key("RUN_SETS_JSON", "sets", "sets", _dump_list, lambda raw: json_list(raw, "RUN_SETS_JSON")),
    # Written as a duration, read as seconds.
    Key("RUN_TIMEOUT", "timeout", "timeout_seconds", str, parse_duration),
    # Empty at submit: only the node knows its own core count, and
    # `parse_environment` fills a zero in from `os.cpu_count()`.
    Key("RUN_WORKERS", "workers", "workers", lambda v: "" if v is None else str(v), _int),
    Key("RUN_CHECKPOINT_EVERY", "checkpoint_every", "checkpoint_every", str, _int),
    Key(
        "RUN_EVAL_METHOD",
        "eval_method",
        "eval_method",
        str,
        lambda raw: raw or DEFAULT_EVAL_METHOD,
    ),
    # One rung at submit; the ladder the node will score.
    Key(
        "RUN_EVAL_AT",
        "eval_at",
        "eval_rungs",
        str,
        lambda raw: tuple(rung for rung in (raw or "").split(",") if rung),
    ),
    Key(
        "RUN_EVAL_FLAGS_JSON",
        "eval_flags",
        "eval_flags",
        _dump_list,
        lambda raw: json_list(raw, "RUN_EVAL_FLAGS_JSON"),
    ),
    Key("RUN_FORCE_PUBLISH", "force_publish", "force_publish", lambda v: "1" if v else "", bool),
    Key("RUN_GIT_COMMIT", "git_commit", "git_commit"),
    # Three-state ("1"/"0"/""), unlike the boolean above: `gitinfo` distinguishes
    # "verified clean" from "unknown", and that distinction is what makes a bare
    # hash worth recording.
    Key("RUN_GIT_DIRTY", "git_dirty", "git_dirty"),
    Key("RUN_GIT_BRANCH", "git_branch", "git_branch"),
    Key("RUN_UNIVERSE_BOARDS", "universe_boards", "universe_boards", _number, _int),
    Key("RUN_UNIVERSE_SEED", "universe_seed", "universe_seed", _number, _int),
    Key("RUN_DTYPE", "dtype", "dtype"),
    Key("RUN_WARM_START_FROM", "warm_start_from", "warm_start_from"),
    Key("RUN_WARM_START_WEIGHT", "warm_start_weight", "warm_start_weight", _number, _int),
    Key("RUN_WARM_START_AT", "warm_start_at", "warm_start_at", _number, _int),
)


def encode(source: object) -> dict[str, str]:
    """Every key, from whatever holds the submit-side attributes.

    Emitted even when empty. An absent variable and an empty one are the same
    thing to the node, so writing them all keeps the contract visible in one
    place rather than implied by which branch happened to set what.
    """
    return {key.env: key.encode(getattr(source, key.spec)) for key in KEYS}


def decode(env: dict[str, str]) -> dict[str, Any]:
    """The node-side keyword arguments, from the task's environment."""
    return {key.plan: key.decode(env.get(key.env, "")) for key in KEYS}
