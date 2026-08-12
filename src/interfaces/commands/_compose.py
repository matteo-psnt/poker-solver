"""Answering one question with SEVERAL commands, concurrently.

A screen is rarely one command. `status` needs three; a run's page needs five.
Answering them one after another is not merely slower, it is *differently* slow:
these are latency, not work -- a share read is ~120 round trips for 0.23 MB --
so a serial screen costs the SUM of its panels while a concurrent one costs its
slowest. Measured against the live pool for `status`: 0.9s + 11s + 23s serially,
against 23s together.

This module is the fan-out, extracted from `status`, which had the only copy.
The console needed the same three properties and would otherwise have written
them a second time -- and one of them is a trap that has already been paid for
once (see :func:`_bound`).

**It composes; it does not read.** Every part is a :meth:`Command.invoke`, so
there is exactly one implementation of each question and a composed view cannot
drift from the command that owns it. That is the rule the console is
subordinate to, and the reason the previous browser UI was deleted: it grew
services that answered questions the CLI already answered, they drifted, and
nothing failed until someone looked.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.interfaces.commands._base import Command
from src.interfaces.errors import attempt


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


def _answer(part: Part) -> dict[str, Any]:
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
    payload, failure = attempt(lambda: part.command.invoke(**part.arguments))
    return {"payload": payload, "error": failure.message if failure else None}


def _bound(part: Part) -> Callable[[], dict[str, Any]]:
    """:func:`_answer`, bound to a COPY of the CALLING thread's context.

    **The copy has to be taken here, on the caller.** Taking it inside the
    worker would copy the worker's own context -- the fresh one that already
    lost everything -- so it would look like a fix and change nothing.

    This is not hypothetical. `telemetry._SURFACE` is a ContextVar, and a raw
    ``pool.submit`` starts its task with a fresh context in which it reverts to
    its default: the three panels carrying the real Azure round-trip cost, which
    is the whole reason the screen is measured, were filed `unknown` while the
    thin wrapper around them was filed `cli`. Fixed in `d67411f`, and it would
    have come straight back in any second copy of this fan-out.

    A copy PER SUBMIT, not one shared copy: :meth:`Context.run` is not
    re-entrant, so handing the same context to concurrent threads raises.

    A zero-argument closure rather than ``pool.submit(context.run, _answer,
    part)`` because `Context.run` is typed with a ParamSpec that does not
    compose through `submit`, and the checker is right to say so.
    """
    context = copy_context()
    return lambda: context.run(_answer, part)


def fan_out(parts: Sequence[Part]) -> dict[str, dict[str, Any]]:
    """Answer every part concurrently, keyed by :attr:`Part.key`.

    One thread per part. These block on the network essentially all of the time,
    so the pool is sized to the work rather than to the machine -- and each part
    builds its own Azure client, so there is no shared mutable state between
    them.

    Concurrency is safe against the two caches in front of these reads, and that
    is checked rather than assumed: `web.cache.TtlCache.get` is single-flight
    per key, and `cloud.store.workspace.SharedTrees.acquire` is single-flight
    and refcounted. Concurrent misses WAIT for the first producer instead of
    each starting a sweep, so N parts asking the same underlying question still
    cost one materialisation of the record.
    """
    if not parts:
        return {}
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = {part.key: pool.submit(_bound(part)) for part in parts}
        return {key: future.result() for key, future in futures.items()}


def compose(
    op: str,
    parts: Sequence[Part],
    join: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A composed payload: the parts, plus when they were answered and how long.

    ``elapsed_seconds`` is the wall clock of the whole fan-out, which is the
    number that says whether concurrency is working. A composed screen whose
    elapsed time equals the sum of its parts has silently become serial, and
    nothing else about the payload would show it.

    ``join`` is where a view draws cross-references BETWEEN parts that have now
    been answered -- it takes them and returns the extra top-level fields. It is
    a callback rather than keyword arguments because the joins depend on the
    answers, which do not exist until the fan-out has returned.

    Joining is not computing, and that distinction is the whole licence this
    module operates under. `task.run_id == run` is the same join the console
    currently does client-side after downloading the entire task log to discard
    ~95% of it; moving it here is a question of where the data already is, not
    of this layer learning what a task means. A join may filter, group and
    cross-reference payloads. It may not derive a quantity -- the moment it
    computes something no command can answer, the console has a second read
    path again, and `tests/interfaces/web/test_no_second_read_path.py` is what
    says so.

    A join runs only over parts that SUCCEEDED. It must tolerate a missing key:
    parts fail independently by design, and a view whose join raises because one
    panel was unavailable has given up the property the fan-out exists for.
    """
    started = time.perf_counter()
    answered = fan_out(parts)
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
