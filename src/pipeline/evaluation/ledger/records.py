"""Building and persisting one evaluation record.

The per-run `evals/<slug>.json` document is the ONLY thing written. There is no
stored ledger: the index is derived from these documents on every read, which is
what makes concurrent evaluation from several machines safe.

Recording used to also append a row to `data/eval_ledger.jsonl`, defaulted at
module scope and never overridden by the node wrapper, so every cloud eval wrote
a stored index the architecture said did not exist. A stored index beside a
derived one is a second answer waiting to disagree."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.ledger.tiers import _knob_hash
from src.shared import records as record_store
from src.shared import task_log
from src.shared.gitinfo import get_git_commit, is_git_dirty

LEDGER_SCHEMA_VERSION = record_store.REGISTRY["eval_ledger.jsonl"].version

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
    """Compose one evaluation entire: provenance, knobs, and full results.

    One document, not the three shapes this replaced. An evaluation used to be a
    180K payload, a 4K record summarising it, and a ledger row identical to the
    record -- the same measurement stored three times. Provenance lived only in
    the record and the samples only in the payload, so neither could be rebuilt
    from the other: 59 of the 78 evals on disk had no matching record and were
    unrecoverable.

    The ledger row is DERIVED from this by :func:`ledger_row`, and that
    derivability is what makes every eval file rebuildable.
    """
    return {
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
        # WHICH task produced this number. Ambient, like the git commits above.
        # Without it there is no key joining an eval document to the task that
        # wrote it: three evaluations of one checkpoint at three board seeds
        # produced three documents and three tasks with nothing connecting the
        # pairs, and correlating them by timestamp is exactly what fails here --
        # concurrent evals of one run have completely overlapping intervals.
        # Empty off a node, where there is no task to point at.
        "task_id": task_log.current_task_id(),
        "knobs": knobs,
        "results": results,
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
    timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Persist one evaluation as a single non-clobbering document under the run dir.

    The single recording path shared by every caller (local CLI and node), so a
    cloud eval and a local eval produce the same on-disk provenance and can be paired
    by :func:`tier_mismatches` without either surface reimplementing the schema.

    ``payload`` must carry ``results`` (with the per-hand ``pair_samples_mbb``) and
    ``infosets``. Returns the document path and the document.
    """
    slug = eval_slug(knobs)
    document = build_record(
        provenance=provenance,
        method=method,
        estimator=estimator,
        infosets=payload["infosets"],
        knobs=knobs,
        results=payload["results"],
        result_path=run_dir / "evals" / f"{slug}.json",
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        checkpoint_iteration=payload.get("checkpoint_iteration"),
    )
    path = write_eval(run_dir, document, slug)
    return path, document


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


def ledger_row(document: dict[str, Any]) -> dict[str, Any]:
    """The compact index row derived from one evaluation document.

    Everything except the bulk results, which stay in the document. The ledger is
    an index over the documents, so this must be derivable from any of them --
    that derivability is what makes `ledger --rebuild` able to regenerate a row
    it has lost.
    """
    results = document.get("results") or {}
    samples = results.get("pair_samples_mbb") or []
    row = {k: v for k, v in document.items() if k != "results"}
    row["results"] = {
        "exploitability_mbb": results.get("exploitability_mbb"),
        "std_error_mbb": results.get("std_error_mbb"),
        "num_hands": results.get("num_hands"),
        "n": len(samples),
    }
    return row


def write_eval(run_dir: Path, document: dict[str, Any], slug: str) -> Path:
    """Write one evaluation to ``evals/<slug>.json``. Never overwrites.

    The slug is timestamp + knob hash + random suffix, so re-evaluating a run
    under the same settings cannot clobber a prior result and concurrent writers
    on several boxes cannot collide.
    """
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    path = evals_dir / f"{slug}.json"
    record_store.write_snapshot(path, document, record_store.REGISTRY["evals/*.json"])
    return path


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
