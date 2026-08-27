"""Answering one question with SEVERAL commands, concurrently.

A screen is rarely one command. `status` needs three; a run's page needs five.
Answering them one after another is not merely slower, it is *differently* slow:
these are latency, not work -- a share read is ~120 round trips for 0.23 MB --
so a serial screen costs the SUM of its panels while a concurrent one costs its
slowest. Measured against the live pool for `status`: 0.9s + 11s + 23s serially,
against 23s together.

**It composes; it does not read.** Every part is a :meth:`Command.invoke`, so
there is exactly one implementation of each question and a composed view cannot
drift from the command that owns it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.interfaces.errors import attempt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.interfaces.commands._base import Command


@dataclass(frozen=True)
class Part:
    """One command's contribution to a composed answer.

    ``key`` names the contribution rather than the command, because a view may
    ask the same command two different questions -- two runs to compare, say --
    and "which of these is the control" is not something the command's own name
    can carry.
    """

    key: str
    command: Command
    arguments: dict[str, Any] = field(default_factory=dict)


# A `type` alias, so the annotation stays lazy and `Command` can remain a
# TYPE_CHECKING import: this module is imported by every surface and the command
# base drags the registry in with it.
type Invoke = Callable[[Command, dict[str, Any]], Any]


def _invoke(command: Command, arguments: dict[str, Any]) -> Any:
    """Run the command directly: what ``invoke=None`` means, and the default.

    Private, and the default is `None` rather than this, so a caller that does
    not memoise never has to name it.
    """
    return command.invoke(**arguments)


def _answer(part: Part, invoke: Invoke) -> dict[str, Any]:
    """Answer one part, or record why it could not be answered.

    Which failures are survivable is :func:`~src.interfaces.errors.attempt`'s
    decision and not this module's -- an expired ``az login`` and an unreachable
    endpoint are the two any composed screen exists to outlive, and every
    surface needs the same list. Anything else PROPAGATES: a bug in a part is
    still a bug, and swallowing it would make this the place exceptions go to be
    quietly rendered as "unavailable".

    The classification is dropped and only the message kept. A part that failed
    is greyed out with a reason either way; the caller that needs the kind (the
    console, picking a status code) asks `attempt` itself.
    """
    started = time.perf_counter()
    payload, failure = attempt(lambda: invoke(part.command, part.arguments))
    elapsed = time.perf_counter() - started
    # Dumped, so a join reads plain data. A view cross-references payloads it did
    # not produce and cannot be typed against all of them at once; the models are
    # what the COMMANDS are checked against, and the envelope is checked by
    # `contract.py` on the way out.
    dump = getattr(payload, "model_dump", None)
    return {
        "payload": dump() if callable(dump) else payload,
        "error": failure.message if failure else None,
        # The fan-out is only as fast as its slowest part, and WHICH part that is
        # was not answerable from the payload: `elapsed_seconds` gives the total,
        # so one slow panel and a serial regression look identical from outside.
        "elapsed_seconds": round(elapsed, 2),
    }


def fan_out(parts: Sequence[Part], invoke: Invoke | None = None) -> dict[str, dict[str, Any]]:
    """Answer every part concurrently, keyed by :attr:`Part.key`.

    One thread per part. These block on the network essentially all of the time,
    so the pool is sized to the work rather than to the machine. The parts SHARE
    an Azure client now (`blob._CONTAINERS`, so a long-lived server keeps one
    pooled TLS connection rather than handshaking per call); the SDK's clients
    are documented thread-safe for requests, and `published_record` already
    fanned one out across a pool before this.

    ``invoke`` is how a part is run, so a caller that already memoises commands
    can hand its memo in. The server does: without it a composed view re-ran
    every command it is made of, so `tasks` -- a 15,684-row read and a 0.94s
    join -- was paid again for each run a person clicked, while `/api/tasks`
    served the same answer from cache.

    Concurrency is safe against the caches in front of these reads, and that is
    checked rather than assumed: `web.cache.TtlCache.get` is single-flight per
    key, and `cloud.store.workspace.SharedTrees.acquire` is single-flight and
    refcounted. Concurrent misses WAIT for the first producer instead of each
    starting a sweep, so N parts asking the same underlying question still cost
    one answer.
    """
    if not parts:
        return {}
    run = invoke or _invoke
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = {part.key: pool.submit(_answer, part, run) for part in parts}
        return {key: future.result() for key, future in futures.items()}


def compose(
    op: str,
    parts: Sequence[Part],
    join: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
    invoke: Invoke | None = None,
) -> dict[str, Any]:
    """A composed payload: the parts, plus when they were answered and how long.

    ``elapsed_seconds`` is the wall clock of the whole fan-out, and a composed
    screen whose elapsed time equals the sum of its parts has silently become
    serial.

    ``join`` draws cross-references BETWEEN parts once they are answered, returning
    extra top-level fields. A callback rather than keyword arguments because the
    joins depend on answers that do not exist until the fan-out returns.

    A join runs only over parts that SUCCEEDED, and must tolerate a missing key:
    parts fail independently by design, and a join that raises because one panel was
    unavailable gives up the property the fan-out exists for.
    """
    started = time.perf_counter()
    answered = fan_out(parts, invoke)
    composed = {
        "op": op,
        "at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "parts": answered,
    }
    if join is not None:
        composed.update(join(answered))
    return composed


def payloads(answered: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The parts that SUCCEEDED, unwrapped, for a join to read.

    A failed part is absent rather than present-and-``None``, so a join written
    as ``payloads(parts).get("tasks")`` cannot accidentally treat "Azure did not
    answer" as "there are no tasks" -- which are different facts that a UI must
    not render the same way.
    """
    return {
        key: part["payload"]
        for key, part in answered.items()
        if part.get("error") is None and part.get("payload") is not None
    }
