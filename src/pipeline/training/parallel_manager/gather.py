"""Result-queue gathering shared by coordinator broadcast operations."""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


def gather_worker_results(
    manager: SharedArrayWorkerManager,
    accept: Callable[[dict[str, object]], bool],
    expected: int,
    timeout: float,
    description: str,
    verbose: bool = False,
) -> tuple[list[dict[str, object]], bool]:
    """
    Collect `expected` accepted results from the manager's result queue.

    Loops until `expected` accepted messages arrive; unrecognized messages are
    discarded without consuming a completion slot, so a stray late result can
    never make a broadcast operation report success while acks are missing.

    Returns (results, interrupted); `interrupted` is True if a KeyboardInterrupt
    arrived while waiting (the gather keeps waiting so workers finish cleanly).
    """
    results: list[dict[str, object]] = []
    interrupted = False
    while len(results) < expected:
        try:
            raw_result = manager.result_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"Timeout waiting for {description} ({len(results)}/{expected} received)"
            )
        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                logger.info(
                    f"⚠️  Interrupt received; waiting for {description}...",
                )
            continue
        if not isinstance(raw_result, dict):
            if verbose:
                logger.info(f"[Master] Ignoring unexpected non-dict result: {raw_result}")
            continue
        result = cast(dict[str, object], raw_result)
        if accept(result):
            results.append(result)
        elif verbose:
            logger.info(f"[Master] Ignoring unexpected result: {result}")
    return results, interrupted
