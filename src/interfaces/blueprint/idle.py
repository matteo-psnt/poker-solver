"""Knowing when nobody is using this, so the box can turn itself off.

The blueprint host is woken on demand and costs money while it runs, so the
question "is anyone still here?" has to be answered by the process that actually
knows -- and then acted on, because a box you have to remember to stop is a box
that runs all weekend.

Idle means no requests, not no sessions
---------------------------------------
A hand in progress is not a reason to stay up; a person *playing* one is. Those
differ: a browser tab left open on a half-finished hand holds a session forever
and sends nothing. So the clock is reset by traffic, and a live session that is
genuinely being played resets it on every action anyway.

Exiting rather than deallocating
--------------------------------
This module ends the *process*; the box's own systemd unit turns that into a
deallocate. Two reasons that split is worth it. A server that called the Azure
control plane would need credentials and a role assignment to be testable at
all, and it would be the only part of this package that could not run on a
laptop. And a process that simply exits is something systemd already knows how
to escalate -- so "stop serving" and "stop paying" stay separable, which is what
you want the first time the deallocate misfires.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# How often the watcher wakes to check. Coarse on purpose: the timeout is in
# minutes, and a tighter tick would only burn a wakeup to sharpen a deadline
# nobody is measuring.
POLL_SECONDS = 15.0


class IdleWatch:
    """Tracks time since the last request and fires once when it exceeds a limit.

    ``on_expire`` defaults to sending this process SIGTERM -- the signal systemd
    and uvicorn both already handle -- but is injectable so a test can observe
    the decision without taking the interpreter down with it.
    """

    def __init__(
        self,
        timeout_seconds: float,
        *,
        on_expire: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._on_expire = on_expire or _terminate_self
        self._last_seen = clock()
        self._lock = threading.Lock()
        self._fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        """A non-positive timeout means "stay up", which is what a laptop wants."""
        return self.timeout_seconds > 0

    def touch(self) -> None:
        """Record activity. Called on every request, so it must stay trivial."""
        with self._lock:
            self._last_seen = self._clock()

    def idle_seconds(self) -> float:
        with self._lock:
            return self._clock() - self._last_seen

    def expired(self) -> bool:
        return self.enabled and self.idle_seconds() >= self.timeout_seconds

    def check(self) -> bool:
        """Fire ``on_expire`` if the limit has passed. Returns whether it fired.

        Fires at most once. Without that, a shutdown that takes a few seconds to
        take effect would be re-triggered on every tick, and the second signal
        would interrupt the graceful drain the first one started.
        """
        with self._lock:
            if self._fired or not self.enabled:
                return False
            if self._clock() - self._last_seen < self.timeout_seconds:
                return False
            self._fired = True
        logger.info("Idle for %.0fs — shutting down.", self.timeout_seconds)
        self._on_expire()
        return True

    def start(self) -> None:
        """Run the check on a daemon thread. A no-op when disabled."""
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="idle-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # `wait` rather than `sleep`, so stopping is immediate rather than up to
        # a full tick late -- which matters in a test far more than in life.
        while not self._stop.wait(POLL_SECONDS):
            if self.check():
                return


def _terminate_self() -> None:
    """The default expiry action: ask this process to shut down gracefully."""
    os.kill(os.getpid(), signal.SIGTERM)
