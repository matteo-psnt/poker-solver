"""What a Batch task's state MEANS, named once.

Batch's own words are actively misleading and cannot be changed: ``active`` is
queued, ``completed`` is stopped-not-succeeded, and ``-9`` is a cancellation
rather than a crash. Four places classified those strings independently -- two in
Python, two in TypeScript -- each answering a different question, each having
rediscovered the ``active`` problem on its own, and nothing anywhere would have
noticed if one drifted.

The questions are genuinely different and stay different. What was missing is a
vocabulary to ask them in:

- :data:`OCCUPIES_A_NODE` -- the clock is running. Cost accounting closes an open
  interval on this and nothing else.
- :data:`PENDING` -- work exists but holds no node. Queue depth, not node time.
- ``OCCUPIES_A_NODE | PENDING`` -- worth listing as in-flight.

Why the cause spellings are derived rather than replaced
-------------------------------------------------------
:func:`phase_of` is the classification; the strings in ``_CAUSE_BY_PHASE`` are
the historical ones the SHARE already holds. Records written months ago carry
``"cause": "preparing"``, and ``task_history.TERMINAL_CAUSES`` gates
reconciliation against those words -- renaming one would make every historical
task look unresolved. So the phase is the concept and the cause string is its
on-disk spelling, declared here in one place instead of assumed in four.
"""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    """Where a Batch task is, in words that mean what they say."""

    QUEUED = "queued"
    """Batch calls this ``active``, which reads as the opposite of what it is."""
    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    """Batch's ``completed``: it STOPPED. Success lives in the exit code."""
    UNKNOWN = "unknown"


"""Which phases mean a node is committed RIGHT NOW.

``QUEUED`` is deliberately absent, and this is the expensive one. A task waiting
for a node has no ``started_at``, so anything that runs an open interval to
``now`` on it credits queue time as node time -- which reported four attempts
abandoned on 2026-08-04 as 455 of 718 node-hours, growing by four hours per
elapsed hour. `tests/shared/test_task_states.py` pins that.
"""
OCCUPIES_A_NODE = frozenset({Phase.STARTING, Phase.RUNNING})

"""Work that exists and holds no node. Queue depth, never node time."""
PENDING = frozenset({Phase.QUEUED})

"""In flight: it has a node or it is waiting for one. What `jobs` lists."""
IN_FLIGHT = OCCUPIES_A_NODE | PENDING


_PHASE_BY_STATE: dict[str, Phase] = {
    "active": Phase.QUEUED,
    "preparing": Phase.STARTING,
    "running": Phase.RUNNING,
    "completed": Phase.FINISHED,
}

"""The phase's spelling in a task record on the share. Frozen by history.

Only the phases a task can be observed IN appear here: a finished task's cause
comes from its outcome (or the node's own account), not from its phase.
"""
_CAUSE_BY_PHASE: dict[Phase, str] = {
    Phase.QUEUED: "active",
    Phase.STARTING: "preparing",
    Phase.RUNNING: "running",
}

"""Causes on the share that mean the clock is running -- DERIVED, not restated.

`task_history.LIVE_CAUSES` is this. Deriving it is what keeps "which phases
occupy a node" a single decision rather than two lists that happen to agree.
"""
LIVE_CAUSES = frozenset(_CAUSE_BY_PHASE[phase] for phase in OCCUPIES_A_NODE)


def phase_of(state: str | None) -> Phase:
    """``BatchTaskState.RUNNING`` -> :attr:`Phase.RUNNING`.

    Tolerates the bare word as well as the enum's ``repr``, because a state
    reaches this from two directions: freshly off the SDK, and out of a task
    record written by an earlier version that had already shortened it.
    """
    word = (state or "").rsplit(".", 1)[-1].lower()
    return _PHASE_BY_STATE.get(word, Phase.UNKNOWN)


def cause_of(phase: Phase) -> str:
    """A phase's spelling in a task record. The identity for anything else."""
    return _CAUSE_BY_PHASE.get(phase, str(phase))


class Outcome(StrEnum):
    """What a task that STOPPED actually achieved.

    Separate from :class:`Phase` because ``FINISHED`` says only that it is over.
    Colouring on the phase alone made a cancellation and a crash identical.
    """

    DONE = "done"
    FAILED = "failed"
    TIMED_OUT = "timed out"
    CANCELLED = "cancelled"


"""What a Batch exit code MEANT, in words. The recurring ones only.

Anything else stays a bare number rather than being guessed at -- and this is on
the PAYLOAD, so the terminal gets the explanation too. It used to exist only in
the console, which meant `poker-solver tasks` printed 137 and left the reader to
know what it was.

124 and 137 are kept apart deliberately: a wrong terminal cause suppresses
reconciliation permanently, so a hang must never be recorded as an OOM.
"""
EXIT_MEANING: dict[int, str] = {
    0: "clean",
    2: "the CLI rejected a flag — argparse exits before doing any work",
    124: "the wall-clock guard fired (a hang, not a crash)",
    137: "SIGKILL from outside — the OOM killer",
    -9: "Batch killed it on cancellation",
}


def outcome_of(exit_code: int | None) -> Outcome:
    """What a finished task achieved. ``None`` reads as clean -- Batch omits the
    code for a task that never ran a process to completion, and there is nothing
    to report as a failure."""
    if exit_code in (0, None):
        return Outcome.DONE
    if exit_code == 124:
        return Outcome.TIMED_OUT
    if exit_code == -9:
        return Outcome.CANCELLED
    return Outcome.FAILED


def exit_meaning(exit_code: int | None) -> str | None:
    """The recurring exit codes explained; ``None`` for anything unfamiliar."""
    return None if exit_code is None else EXIT_MEANING.get(exit_code)
