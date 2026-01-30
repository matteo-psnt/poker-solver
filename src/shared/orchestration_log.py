"""Client-side orchestration log for remote (Modal) runs.

The in-container run metadata (``.run.json``) records what a *living* process did:
per-attempt start/end iterations and compute time. It structurally cannot record
how an attempt *died* — a guillotined (cancelled) or OOM-killed container is gone
before it can write anything. Modal, as an external observer, sees those deaths.

This module is the join between the two worlds:

- ``record_spawn`` runs on the launching client the instant ``.spawn()``/``.remote()``
  returns — before anything can die — persisting the ``object_id`` → ``run_id``
  correlation that later lets us ask Modal "what happened to this call?".
- ``snapshot_call`` asks exactly that, via ``FunctionCall.get_call_graph()``, and
  records the resulting status/exit-cause into our own durable log so it survives
  Modal's limited result-retention window.

Both append newline-delimited JSON to ``data/orchestration.jsonl`` (one process,
append-only), mirroring the eval ledger. Merging this log with each run's attempts
by ``run_id`` reconstructs the full wall-clock timeline — including the gaps between
attempts (provisioning, resume fights) that live on neither side alone.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ORCHESTRATION_LOG = Path("data/orchestration.jsonl")

# Modal ``InputStatus`` name → coarse exit-cause label. The dead process cannot
# report these; Modal can. Note FAILURE conflates in-container exceptions with
# OOM-kills — Modal surfaces both as FAILURE, so OOM is a FAILURE subtype we can't
# split from here (the container logs, if retained, are the only finer signal).
_EXIT_CAUSE_BY_STATUS: dict[str, str] = {
    "SUCCESS": "completed",
    "TERMINATED": "cancelled",  # guillotine: call cancelled / detached client killed
    "TIMEOUT": "timeout",  # hit the function timeout wall
    "FAILURE": "failed",  # in-container error OR OOM-kill (indistinguishable here)
    "INIT_FAILURE": "init_failed",  # container/setup never came up
    "PENDING": "running",  # still executing at snapshot time
}


def classify_status(status_name: str) -> str:
    """Map a Modal ``InputStatus`` name to a coarse exit-cause label."""
    return _EXIT_CAUSE_BY_STATUS.get(status_name, "unknown")


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def record_spawn(
    *,
    run_id: str,
    function: str,
    object_id: str,
    resources: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    launched_at: str | None = None,
    path: Path = DEFAULT_ORCHESTRATION_LOG,
) -> dict[str, Any]:
    """Persist the ``object_id`` → ``run_id`` correlation at launch time.

    This is the one write that always succeeds regardless of how the remote call
    later dies, because it runs on the client immediately after submission. ``run_id``
    may be empty for launches that don't target a specific existing run (e.g. a fresh
    train that mints its run id in-container) — the object_id still anchors the call.
    """
    record: dict[str, Any] = {
        "event": "spawn",
        "ts": launched_at or datetime.now().isoformat(),
        "run_id": run_id,
        "function": function,
        "object_id": object_id,
        "resources": resources or {},
    }
    if extra:
        record["extra"] = extra
    _append(path, record)
    return record


def _query_modal_status(object_id: str, call: Any | None) -> tuple[str | None, str | None]:
    """Return ``(status_name, error)`` for a Modal call, degrading gracefully.

    Imported lazily so this module stays usable (and unit-testable) without Modal.
    Any failure — Modal unreachable, retention expired, unknown id — is returned as
    an error string rather than raised, since a snapshot is best-effort enrichment.
    """
    try:
        import modal

        function_call = call if call is not None else modal.FunctionCall.from_id(object_id)
        graph = function_call.get_call_graph()
        if not graph:
            return None, "empty call graph"
        # graph[0] is the root call. These entrypoints spawn flat, childless calls, so
        # the root's status is the call's status; nested spawns would need a walk.
        status = graph[0].status
        return getattr(status, "name", str(status)), None
    except Exception as exc:  # best-effort enrichment; never fail the caller
        return None, f"{type(exc).__name__}: {exc}"


def snapshot_call(
    *,
    run_id: str,
    function: str,
    object_id: str,
    call: Any | None = None,
    path: Path = DEFAULT_ORCHESTRATION_LOG,
) -> dict[str, Any]:
    """Query Modal for a call's execution status and append a snapshot record.

    Pass ``call`` when a live ``FunctionCall`` is already in hand (blocking paths);
    otherwise it is reconstructed from ``object_id`` via ``FunctionCall.from_id``.
    Never raises — a failed query is recorded as ``modal_status=None`` with the error.
    """
    status_name, error = _query_modal_status(object_id, call)
    record: dict[str, Any] = {
        "event": "snapshot",
        "ts": datetime.now().isoformat(),
        "run_id": run_id,
        "function": function,
        "object_id": object_id,
        "modal_status": status_name,
        "exit_cause": classify_status(status_name) if status_name else "unknown",
    }
    if error:
        record["query_error"] = error
    _append(path, record)
    return record
