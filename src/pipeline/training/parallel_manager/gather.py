"""Result-queue gathering shared by coordinator broadcast operations."""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

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
                print(
                    f"⚠️  Interrupt received; waiting for {description}...",
                    flush=True,
                )
            continue
        if not isinstance(raw_result, dict):
            if verbose:
                print(f"[Master] Ignoring unexpected non-dict result: {raw_result}", flush=True)
            continue
        result = cast(dict[str, object], raw_result)
        if accept(result):
            results.append(result)
        elif verbose:
            print(f"[Master] Ignoring unexpected result: {result}", flush=True)
    return results, interrupted
