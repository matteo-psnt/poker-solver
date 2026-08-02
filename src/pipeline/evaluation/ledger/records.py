"""Building and persisting one evaluation record.

The per-run `evals/record-*.json` file is the source of truth; the ledger is a
rebuildable cache, which is what makes concurrent evaluation from several
machines safe."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.ledger.tiers import _knob_hash
from src.shared.gitinfo import get_git_commit, is_git_dirty
from src.shared.jsonio import json_default

LEDGER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = Path("data/eval_ledger.jsonl")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunProvenance:
    """Provenance of the evaluated run, recorded verbatim in each ledger row.

    Plain fields rather than the training layer's ``RunMetadata`` — the ledger
    only needs these scalars, and taking them directly keeps evaluation from
    importing training.
    """

    run_id: str
    git_commit: str | None
    git_dirty: bool | None
    config_name: str
    card_abstraction_hash: str | None
    action_config_hash: str | None
    # Experiment lineage, copied onto the row so a report can group and attribute
    # arms without opening every run's .run.json. None on unaffiliated runs.
    experiment_id: str | None = None
    arm: str | None = None
    parent_run_id: str | None = None
    # `config_name` is not identity (it comes from system.config_name in the YAML),
    # so an override-variant is only distinguishable from its base by this.
    config_hash: str | None = None


def build_record(
    *,
    provenance: RunProvenance,
    method: str,
    estimator: str,
    infosets: int,
    knobs: dict[str, Any],
    results: dict[str, Any],
    result_path: Path,
    timestamp: str,
    checkpoint_iteration: int | None = None,
) -> dict[str, Any]:
    """Compose the compact ledger row (no per-hand samples — those live in the payload)."""
    samples = results.get("pair_samples_mbb") or []
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id": provenance.run_id,
        "method": method,
        "estimator": estimator,
        # Two commits matter and mean different things: the code that produced the
        # checkpoint, and the code that measured it (LBR methodology changes across
        # commits). Both are recorded so neither has to be reconstructed later.
        "train_git_commit": provenance.git_commit,
        "train_git_dirty": provenance.git_dirty,
        "eval_git_commit": get_git_commit(),
        "eval_git_dirty": is_git_dirty(),
        "config_name": provenance.config_name,
        "config_hash": provenance.config_hash,
        "card_abstraction_hash": provenance.card_abstraction_hash,
        "action_config_hash": provenance.action_config_hash,
        "experiment_id": provenance.experiment_id,
        "arm": provenance.arm,
        "parent_run_id": provenance.parent_run_id,
        # WHICH checkpoint produced this number. A run id alone does not identify
        # one: the same run is evaluated at successive iterations, so without this
        # two rows for one run are indistinguishable and a stale read looks like a
        # real result. ``infosets`` was already being passed in and silently dropped.
        "checkpoint_iteration": checkpoint_iteration,
        "infosets": infosets,
        "knobs": knobs,
        "results": {
            "exploitability_mbb": results.get("exploitability_mbb"),
            "std_error_mbb": results.get("std_error_mbb"),
            "num_hands": results.get("num_hands"),
            "n": len(samples),
        },
        "result_path": payload_pointer(result_path, provenance.run_id),
    }


def record_evaluation(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    provenance: RunProvenance,
    method: str,
    estimator: str,
    knobs: dict[str, Any],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Persist one evaluation: non-clobbering payload under the run dir + a ledger row.

    The single recording path shared by every caller (local CLI and Modal), so a
    cloud eval and a local eval produce the same on-disk provenance and can be paired
    by :func:`tier_mismatches` without either surface reimplementing the schema.

    ``payload`` must carry ``results`` (with the per-hand ``pair_samples_mbb``) and
    ``infosets``. Returns the payload path and the appended record.
    """
    results = payload["results"]
    slug = eval_slug(knobs)
    result_path = write_payload(run_dir, payload, slug)
    record = build_record(
        provenance=provenance,
        method=method,
        estimator=estimator,
        infosets=payload["infosets"],
        knobs=knobs,
        results=results,
        result_path=result_path,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        checkpoint_iteration=payload.get("checkpoint_iteration"),
    )
    write_record(run_dir, record, slug)
    append_record(record, ledger_path)
    return result_path, record


def eval_slug(knobs: dict[str, Any]) -> str:
    """Unique per-eval filename stem: UTC stamp + knob hash + random suffix.

    The random suffix is what makes this safe for concurrent writers. Microsecond
    stamps alone collide when several boxes evaluate the same run with the same
    knobs -- exactly the fan-out shape a noise-floor sweep produces -- and the repo
    already learned this for ``run_id`` (``uuid4().hex[:6]`` in session.py). UTC so
    names from boxes in different timezones still sort.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}-{_knob_hash(knobs)}-{uuid.uuid4().hex[:6]}"


def write_payload(run_dir: Path, payload: dict[str, Any], slug: str) -> Path:
    """Write the full eval payload to a non-overwriting per-eval file under the run dir.

    Named by timestamp + knob hash + random suffix so re-evaluating a run under
    different (or the same) settings never clobbers a prior result — the pre-ledger
    ``evaluate_result.json`` was overwritten on every eval, discarding history.
    """
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    path = evals_dir / f"eval-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, default=json_default))
    return path


def write_record(run_dir: Path, record: dict[str, Any], slug: str) -> Path:
    """Write the complete ledger row beside its payload, under the run directory.

    This is the durable copy. The shared ``eval_ledger.jsonl`` is an append to a
    file every writer shares, which is the one shared-mutable-state in the system
    and the one thing that has actually lost data (12/14 rows in a parallel sweep).
    A uniquely-named file under the run being evaluated has no such contention: a
    run directory has a single writer, so this write cannot race anything.

    The full row is written, not just the payload -- ``eval_git_commit``, ``knobs``
    and ``timestamp`` are captured at record time and exist nowhere else, so a
    payload alone cannot reconstruct a row.
    """
    records_dir = run_dir / "evals"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"record-{slug}.json"
    path.write_text(json.dumps(record, indent=2, default=json_default))
    return path


def append_record(record: dict[str, Any], ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append one row to the ledger JSONL cache, creating it if needed.

    Best-effort convenience so the common single-machine case needs no rebuild.
    The durable copy is :func:`write_record`; anything lost here is recoverable
    with ``ledger --rebuild``.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default) + "\n")


def record_instant(record: dict[str, Any]) -> datetime:
    """When an eval happened, as an aware datetime, for ordering.

    Two timestamp vintages coexist and must not be compared as strings. Rows
    written before the UTC switch are naive *local* time -- that is what
    ``datetime.now()`` produced at write time -- while new rows carry an explicit
    ``+00:00``. Lexicographic comparison would skew the two apart by the writer's
    UTC offset, so naive values are attached to the local zone (the zone that
    actually produced them) rather than reinterpreted as UTC.

    Unparseable or missing timestamps sort first, keeping them visible rather
    than dropping them.
    """
    raw = str(record.get("timestamp") or "")
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    # astimezone() on a naive value interprets it as local time, which is the
    # assumption we want here and the reason this is not `replace(tzinfo=UTC)`.
    return stamp.astimezone() if stamp.tzinfo is None else stamp


def payload_pointer(result_path: Path, run_id: str) -> str:
    """Portable, run-relative pointer to a payload: ``<run_id>/evals/<file>``.

    Rows used to store a working-directory-relative path, which resolves to nothing
    on a machine that mounts its data elsewhere -- or, worse, to a *different*
    machine's local ``data/``. Anchoring at the run id makes the pointer mean the
    same thing wherever the runs directory happens to live.
    """
    return f"{run_id}/{result_path.parent.name}/{result_path.name}"


def load_payload(record: dict[str, Any], runs_dir: Path | None = None) -> dict[str, Any]:
    """Load the full per-eval payload a ledger row points at.

    Tries, in order: the run-relative pointer under ``runs_dir`` (current format),
    the stored path as-is (legacy CWD-relative rows), and finally the payload's
    basename under the run's ``evals/`` — the filename is unique by construction, so
    that last one recovers a row whose pointer was written by an older layout.
    """
    stored = str(record["result_path"])
    run_id = record.get("run_id") or ""
    candidates: list[Path] = []
    if runs_dir is not None:
        candidates.append(runs_dir / stored)
    candidates.append(Path(stored))
    if runs_dir is not None and run_id:
        candidates.append(runs_dir / run_id / "evals" / Path(stored).name)

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())

    raise FileNotFoundError(f"Eval payload for run '{run_id}' not found at {stored}")
