"""Knowing when nobody is using this, so the box can turn itself off.

The blueprint host is woken on demand and costs money while it runs, so a box
you have to remember to stop is a box that runs all weekend.

Idle means no REQUESTS, not no sessions. A browser tab left open on a
half-finished hand holds a session forever and sends nothing, while a hand
genuinely being played resets the clock on every action.

This module ends the PROCESS; the box's systemd unit turns that into a
deallocate. Keeping those separable is what makes the server testable on a
laptop -- calling the Azure control plane here would need credentials and a role
assignment -- and it is what you want the first time the deallocate misfires.

The exit code is 42 because every other code is taken and means something else:
0 is a deliberate stop, and 143 is SIGTERM, which is what ``systemctl stop`` and
the ``restart`` in ``deploy.sh`` produce. Deallocating a box mid-deploy is the
same bug as never deallocating it. The unit lists 42 in ``SuccessExitStatus`` so
systemd does not restart it, and the guard deallocates on it and nothing else.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# How often the watcher wakes to check. Coarse on purpose: the timeout is in
# minutes, and a tighter tick would only burn a wakeup to sharpen a deadline
# nobody is measuring.
POLL_SECONDS = 15.0

# Must agree with `SuccessExitStatus=` and the `deallocate-if-idle` guard, both
# in `infra/serve/main.tf`; `test_idle.py` pins them together. Distinct from 0
# (deliberate stop) and 143 (SIGTERM, which a deploy produces).
IDLE_EXIT_CODE = 42


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

    @property
    def fired(self) -> bool:
        """Whether expiry has happened.

        Read AFTER the server loop returns, to tell "the box put itself to bed"
        apart from every other way a server can stop. The signal alone cannot
        carry that: a graceful SIGTERM from here and one from `systemctl stop`
        are the same signal and the same exit code.
        """
        with self._lock:
            return self._fired

    def expire_with(self, action: Callable[[], None]) -> None:
        """Replace what expiry DOES, after construction.

        The app builds the watch but only the caller holds the server, and on
        the hosted box expiry has to end the process with a specific code --
        which a signal cannot deliver.

        MEASURED: the default ``SIGTERM myself`` cannot be used there, because
        uvicorn re-raises the captured signal after restoring the default
        handler. ``uvicorn.run()`` therefore never returns on that path and the
        process dies as 143, so any exit code the caller wanted is unreachable.
        Setting ``should_exit`` instead lets the server return normally and the
        caller decide how to exit.
        """
        with self._lock:
            self._on_expire = action

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
            action = self._on_expire
        logger.info("Idle for %.0fs — shutting down.", self.timeout_seconds)
        action()
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
