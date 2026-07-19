"""Per-checkpoint phase profiling, durable across container teardown.

A checkpoint's cost is spread across three layers that no single one can see:
the coordinator's cross-worker key collection (``pipeline``), the array/key-table
write (``engine``), and — on Modal — the Volume commit that happens after training
returns (``modal_app``). Local profiling showed the write is ~65% of a checkpoint
and grows as O(total infosets), but that extrapolates to far less than the
minutes-scale checkpoints observed in production, so the remainder lives somewhere
only a real cloud run can measure.

This module is the common accumulator. Phases report into the active recording via
:func:`phase`; :func:`emit` appends one JSON object per checkpoint to
``<run_dir>/checkpoint_profile.jsonl`` — inside the run directory, so on Modal it
lands on the Volume and survives the container. Recording is always on: the
overhead is a handful of ``perf_counter`` calls against a phase measured in seconds.

Never raises into the checkpoint path. A profiling bug must not cost a run its
progress, so every entry point swallows its own errors.

Read the result with ``poker-solver-run checkpoint-profile <run-id>``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_FILENAME = "checkpoint_profile.jsonl"


@dataclass
class CheckpointRecording:
    """One checkpoint's accumulated phase timings and artifact stats.

    Fields whose meaning is not obvious from the type:

    - ``phases``: phase name -> seconds. Nested phases are recorded flat, so
      summing every entry double-counts; ``total_seconds`` is authoritative.
    - ``stats``: non-timing facts a phase chose to attach (file counts, bytes).
    - ``total_seconds``: filled by :func:`emit`, measured from recording start.
    """

    iteration: int
    started_at: float
    phases: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    total_seconds: float = 0.0


_local = threading.local()


def _current() -> CheckpointRecording | None:
    return getattr(_local, "recording", None)


@contextmanager
def recording(iteration: int) -> Iterator[CheckpointRecording | None]:
    """Open a recording for one checkpoint. Nested calls are no-ops."""
    if _current() is not None:
        yield _current()
        return
    rec = CheckpointRecording(iteration=iteration, started_at=time.perf_counter())
    _local.recording = rec
    try:
        yield rec
    finally:
        _local.recording = None


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Time a named phase into the active recording; a no-op if none is open."""
    rec = _current()
    if rec is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        rec.phases[name] = rec.phases.get(name, 0.0) + (time.perf_counter() - start)


def add_stats(**values: Any) -> None:
    """Attach non-timing facts (file counts, byte sizes) to the active recording."""
    rec = _current()
    if rec is not None:
        rec.stats.update(values)


def measure_tree(root: Path) -> dict[str, int]:
    """Count files and bytes under ``root``.

    Modal Volume commit cost tracks file *count* far more than total size, and the
    Zarr DirectoryStore emits one file per chunk per array, so this is the stat
    that decides whether consolidating chunks is worth it.
    """
    files = 0
    total_bytes = 0
    try:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    except OSError:
        return {"files": 0, "bytes": 0}
    return {"files": files, "bytes": total_bytes}


def emit(run_dir: Path | None, rec: CheckpointRecording | None, **extra: Any) -> None:
    """Append one recording to ``<run_dir>/checkpoint_profile.jsonl``."""
    if run_dir is None or rec is None:
        return
    try:
        rec.total_seconds = time.perf_counter() - rec.started_at
        row: dict[str, Any] = {
            "iteration": rec.iteration,
            "total_seconds": round(rec.total_seconds, 4),
            "phases": {k: round(v, 4) for k, v in rec.phases.items()},
            **rec.stats,
            **extra,
        }
        path = Path(run_dir) / PROFILE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        return


def record_volume_commit(run_dir: Path | None, seconds: float, **extra: Any) -> None:
    """Append a standalone row for the Modal Volume commit.

    The commit happens after training returns, outside any checkpoint, so it gets
    its own row rather than a phase inside one.
    """
    if run_dir is None:
        return
    try:
        row = {"event": "volume_commit", "total_seconds": round(seconds, 4), **extra}
        path = Path(run_dir) / PROFILE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        return
