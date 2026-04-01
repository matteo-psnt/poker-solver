"""The one thing the console records rather than reads.

Every other panel asks a question and forgets the answer. Node-hours cannot work
that way: Batch retains no node history, so a number nobody wrote down is gone.
The series therefore begins when this thread begins, and the page says so.

A daemon thread rather than a scheduler or a cron: it must not keep the process
alive on shutdown, and it has exactly one job.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.interfaces.cli.commands import pool_status
from src.shared import pool_samples

log = logging.getLogger(__name__)

# 15s is chosen against what is being measured, not what is cheap: the pool
# scales 0->N->0 on its own, and a coarse interval would miss short bursts
# entirely -- exactly the ones nobody remembers running.
INTERVAL_SECONDS = 15.0


class PoolSampler:
    """Records the pool's node count until asked to stop."""

    def __init__(
        self,
        *,
        path: Path = pool_samples.DEFAULT_PATH,
        interval: float = INTERVAL_SECONDS,
    ) -> None:
        self._path = path
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Prune once, then sample until stopped.

        Pruning here rather than per sample: at ~5,760 lines a day the file is
        small, and rewriting it every 15s would be pure waste.
        """
        try:
            removed = pool_samples.prune(self._path)
            if removed:
                log.info("pruned %d pool sample(s) past retention", removed)
        except OSError:
            log.warning("could not prune %s; sampling anyway", self._path, exc_info=True)

        self._thread = threading.Thread(target=self._loop, name="pool-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            # Waiting on the event, not sleeping: a stop is acted on immediately
            # rather than after up to a full interval.
            self._stop.wait(self._interval)

    def _sample_once(self) -> None:
        """One observation. NOTHING here may raise.

        A sampler that dies on a transient cloud error stops recording silently,
        and the loss is unrecoverable -- there is no backfill. An expired
        `az login` should cost a gap in the series, which `integrate` reports as
        unobserved, not the end of the series.
        """
        try:
            payload = pool_status.COMMAND.invoke()
            pool_samples.append(
                int(payload.get("current_dedicated_nodes") or 0),
                payload.get("vm_size"),
                path=self._path,
            )
        except Exception:
            log.debug("pool sample skipped", exc_info=True)
