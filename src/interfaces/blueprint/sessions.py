"""Live play sessions: the one piece of mutable state in the whole design.

Everything else the blueprint server does is a pure function of a loaded run.
A hand in progress is not: it has an identity, a next expected input, and a
lifetime. That makes this the only module here that needs a lock and an eviction
rule, and keeping it to one module is the point.

Bounded, and oldest-first
-------------------------
Sessions are capped and the least-recently-touched is dropped when the cap is
hit. A browser tab that navigates away never says goodbye, so without a bound
this is a memory leak whose rate is set by how many times someone reloads the
page. Dropping the oldest is right rather than refusing the newest: a person
sitting down to play should never be turned away because of tabs they closed
yesterday.

A dropped session is gone, and asking for it says so
----------------------------------------------------
:meth:`Sessions.get` raises rather than returning ``None`` for an unknown id, so
a client cannot mistake "your hand was evicted" for "your hand is over" -- the
second is a result and the first means the hand you were playing no longer
exists.
"""

from __future__ import annotations

import itertools
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable

from src.engine.search.heads_up_session import HeadsUpHand
from src.engine.solver.policy_source import ScorableBlueprint

DEFAULT_LIMIT = 64


class UnknownSessionError(KeyError):
    """No session by that id -- never started, or evicted since."""


class Sessions:
    """A bounded, thread-safe store of hands in progress.

    Thread-safe because FastAPI runs synchronous handlers in a threadpool, so two
    requests genuinely do touch this at once. The lock covers the bookkeeping
    only; a hand's own advance happens outside it, and two requests against the
    SAME session are the client's problem -- serialising them here would hide a
    double-submit rather than let the second one fail on the state it finds.
    """

    def __init__(
        self,
        blueprint: ScorableBlueprint,
        *,
        limit: int = DEFAULT_LIMIT,
        seeds: Callable[[], int] | None = None,
    ):
        self._blueprint = blueprint
        self._limit = limit
        self._hands: OrderedDict[str, HeadsUpHand] = OrderedDict()
        self._lock = threading.Lock()
        # Injectable so a test can pin a hand without pinning the id, and so the
        # server's own seeds are unguessable rather than sequential.
        self._seeds = seeds or (lambda: secrets.randbits(63))
        self._buttons = itertools.count()

    def start(
        self,
        *,
        human_seat: int,
        button: int | None = None,
        seed: int | None = None,
    ) -> tuple[str, HeadsUpHand]:
        """Deal a new hand and return its id.

        ``button`` defaults to alternating, which is what a real heads-up session
        does -- holding it fixed would let a player see only one side of every
        spot and read that as the blueprint's whole strategy.
        """
        seat = self._next_button() if button is None else button
        hand = HeadsUpHand(
            self._blueprint,
            human_seat=human_seat,
            button=seat,
            seed=self._seeds() if seed is None else seed,
        )
        session_id = secrets.token_urlsafe(9)
        with self._lock:
            self._hands[session_id] = hand
            while len(self._hands) > self._limit:
                self._hands.popitem(last=False)
        return session_id, hand

    def get(self, session_id: str) -> HeadsUpHand:
        """The hand, marked as most recently used. Raises if it is not here."""
        with self._lock:
            hand = self._hands.get(session_id)
            if hand is None:
                raise UnknownSessionError(session_id)
            self._hands.move_to_end(session_id)
            return hand

    def drop(self, session_id: str) -> None:
        """Forget a session. Idempotent -- leaving is not an error."""
        with self._lock:
            self._hands.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._hands)

    def _next_button(self) -> int:
        return next(self._buttons) % 2
