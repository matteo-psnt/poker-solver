"""The `backfill-record` subcommand: import the published record into Postgres.

Read-only against the share and IDEMPOTENT against the database, so it can be
run repeatedly while the two stores are kept in step. That is the whole safety
argument for importing everything first and pruning later: a row is reversible
by re-running this, a deleted file is not.

It lives in `commands/` rather than beside the adapter because it needs BOTH
sides -- `tier_key` from the pipeline and the tables from the adapter -- and the
composition root is the only place allowed to know both. `an_adapter_does_not_do_the_work`
is what makes that not a matter of taste.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.interfaces.commands._base import Command, records_root
from src.interfaces.errors import CommandError
from src.shared import run_events, task_history
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse

DSN_ENV = "POKER_SOLVER_RECORD_DSN"

# The namespace for the deterministic event ids below. Any fixed uuid does; what
# matters is that it never changes, or every event re-imports as a new row.
_EVENT_NS = uuid.UUID("6f1a9c2e-1f4a-4a1e-9f7d-0b2c3d4e5f60")

# `.complete-static-<iteration>.zarr`, the same shape `prune-checkpoints` matches.
_RUNG = re.compile(r"^static-(\d+)\.zarr$")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver backfill-record`."""
    parser.add_argument(
        "--apply", action="store_true", help="Actually write. Without it, counts what it would."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=None,
        help="Limit to this run; repeatable.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare what the share holds against what the database holds, and report the gap.",
    )


class Counts(BaseModel):
    runs: int = 0
    events: int = 0
    checkpoints: int = 0
    evals: int = 0
    legs: int = 0


class Divergence(BaseModel):
    """One run the two stores disagree about, and by how much."""

    run: str
    kind: str
    on_share: int
    in_database: int


class BackfillPayload(BaseModel):
    op: Literal["backfill-record"] = "backfill-record"
    applied: bool = False
    verified: bool = False
    share: Counts = Field(default_factory=Counts)
    database: Counts = Field(default_factory=Counts)
    skipped: list[str] = Field(default_factory=list)
    # Empty is the answer this command exists to produce during dual write.
    divergences: list[Divergence] = Field(default_factory=list)


def _digest(record: dict[str, Any]) -> str:
    """The pairing tier, hashed.

    Delegates to `tier_key` and never re-derives it. That function is the one
    implementation of which knobs make two evals comparable -- ~30 of them,
    including `policy_threshold` and `avg_gamma` -- and a second implementation
    here would drift into pairing rows that must not be compared.
    """
    from src.pipeline.evaluation.ledger import tiers  # noqa: PLC0415

    return hashlib.sha256(
        json.dumps(list(tiers.tier_key(record)), sort_keys=True, default=str).encode()
    ).hexdigest()[:32]


def _event_uuid(run_id: str, index: int, body: dict[str, Any]) -> uuid.UUID:
    """A DETERMINISTIC id, so re-importing the same event is a no-op.

    Position plus content: two `progress` events in one run can carry identical
    bodies, and hashing content alone would silently collapse them into one.
    """
    material = f"{run_id}|{index}|{json.dumps(body, sort_keys=True, default=str)}"
    return uuid.uuid5(_EVENT_NS, material)


def _rows_for_run(run_dir: Path, models: Any) -> tuple[Any, list[Any], list[Any]] | None:
    """One run's `runs` row, its events and its checkpoints, or None if unreadable."""
    from src.pipeline.training.run_tracker.metadata import RunMetadata  # noqa: PLC0415

    try:
        events = run_events.read(run_dir)
        metadata = RunMetadata.load(run_dir)
    except (OSError, ValueError, KeyError):
        return None

    # Through `metadata`, NOT the `created` event. A run written before the
    # event log has a `.run.json` and no log at all, so every field read off
    # `created` came back None for nine of them -- their arm, experiment and
    # provenance silently absent from the database while the share had them.
    # `RunMetadata` is the one thing that reads both layouts, and this is the
    # fourth defect in this migration caused by going round it.
    created = run_events.head(events) or {}

    def field(name: str) -> Any:
        value = getattr(metadata, name, None)
        return created.get(name) if value is None else value

    run = models.Run(
        run_id=run_dir.name,
        config_name=metadata.config_name or created.get("config_name") or "",
        kernel=field("kernel"),
        arm=field("arm"),
        experiment_id=field("experiment_id"),
        parent_run_id=field("parent_run_id"),
        action_config_hash=field("action_config_hash"),
        card_abstraction_hash=field("card_abstraction_hash"),
        config_hash=field("config_hash"),
        # Lossless: `verify_trainer_knobs` compares whole config blocks.
        config=_as_dict(field("config")),
        git_commit=field("git_commit"),
        git_dirty=field("git_dirty"),
        git_branch=field("git_branch"),
        code_snapshot=field("code_snapshot"),
        storage_capacity=field("storage_capacity"),
        started_at=metadata.started_at,
        completed_at=getattr(metadata, "completed_at", None),
        status=metadata.status or "running",
        iterations=metadata.iterations or 0,
        num_infosets=getattr(metadata, "num_infosets", None),
        runtime_seconds=metadata.runtime_seconds or 0.0,
    )

    # `attempt` is carried on the event rather than inferred, because the
    # run-vs-attempt distinction is what stops a dead attempt's `died` reading as
    # the run's own status.
    attempt = 0
    event_rows = []
    for index, body in enumerate(events):
        if body.get(run_events.EVENT_KEY) == run_events.ATTEMPT_STARTED:
            attempt = int(body.get("index", attempt))
        event_rows.append(
            models.RunEvent(
                run_id=run_dir.name,
                attempt=attempt,
                event=str(body.get(run_events.EVENT_KEY) or ""),
                at=body.get("ts"),
                body=body,
                event_uuid=_event_uuid(run_dir.name, index, body),
            )
        )

    # From the MARKERS, not the manifest. `prune-checkpoints` deletes a
    # snapshot and its marker without rewriting the manifest that advertises it
    # -- that disagreement is what `verify_published_rungs` exists to absorb --
    # so the manifest still names 769 rungs this session deleted. The marker is
    # the share's own claim that a rung is there; the manifest is a stale index.
    # `_published_rungs` in prune-checkpoints reads exactly these, for exactly
    # this reason.
    manifest: dict[str, Any] = {}
    manifest_path = run_dir / "STATIC_CHECKPOINT.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}

    current = manifest.get("iteration")
    checkpoint_rows = []
    for marker in sorted(run_dir.glob(f"{archive.MARKER_PREFIX}static-*.zarr")):
        match = _RUNG.match(marker.name[len(archive.MARKER_PREFIX) :])
        if not match:
            continue
        iteration = int(match.group(1))
        checkpoint_rows.append(
            models.Checkpoint(
                run_id=run_dir.name,
                iteration=iteration,
                # Where it WILL live. Nothing has moved to Blob yet, so this is
                # the address the migration writes to, not a claim bytes are
                # there -- the fingerprint is what a loader actually checks.
                blob_uri=f"rungs/{run_dir.name}/{iteration}",
                fingerprint=manifest.get("fingerprint") or "",
                abstraction_id=manifest.get("abstraction_id"),
                is_current=(iteration == current),
            )
        )
    return run, event_rows, checkpoint_rows


def _as_dict(value: Any) -> dict[str, Any]:
    """A config as JSON, whichever shape the fold handed back."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def _eval_rows(run_dir: Path, models: Any) -> list[Any]:
    """Every eval document under one run, filtered EXACTLY as the ledger filters.

    Three exclusions, all copied from `ledger/queries.py` rather than invented,
    because each one is a measured failure:

    * `eval-*` and `record-*` are the two pre-substrate shapes. They still sit
      beside the documents that replaced them and a legacy record points at the
      OLD filename, so reading both enters one evaluation twice -- measured at
      63 rows becoming 110.
    * a document with no `run_id` cannot be attached to anything.
    * a document with no knobs or no timestamp CANNOT BE TIERED. It hashes into
      a tier of `(method, None, ...)`, sorts to year 1 AD, and since tiers rank
      by coverage a pile of them becomes the default curve. The document stays
      on disk; only the index withholds it.

    The last one is also why `recorded_at` is NOT NULL and this does not paper
    over it: an untimestamped eval has no place in a comparison index.
    """
    rows = []
    evals = run_dir / "evals"
    if not evals.is_dir():
        return rows
    for path in sorted(evals.glob("*.json")):
        if path.name.startswith(("eval-", "record-")):
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not doc.get("run_id") or not doc.get("knobs") or not doc.get("timestamp"):
            continue
        results = doc.get("results") or {}
        knobs = doc.get("knobs") or {}
        rows.append(
            models.Eval(
                eval_id=path.stem,
                run_id=run_dir.name,
                checkpoint_iteration=doc.get("checkpoint_iteration"),
                method=str(doc.get("method") or doc.get("estimator") or ""),
                # `base_seed`, which is where the seed actually lives.
                base_seed=knobs.get("base_seed"),
                knobs=knobs,
                tier_digest=_digest(doc),
                exploitability_mbb=results.get("exploitability_mbb"),
                std_error_mbb=results.get("std_error_mbb"),
                num_hands=results.get("num_hands"),
                recorded_at=doc.get("timestamp"),
                payload=doc,
            )
        )
    return rows


def _leg_instant(document: dict[str, Any]) -> Any:
    """When a leg happened, from whichever field its writer used.

    Not one field, because the writers are different programs. The node stamps
    `ts`; `write_observed_record` stamps `observed_at`, because it is the READER
    saying when IT looked, not the node saying when something happened. Falling
    back to Batch's own times last keeps a record that has neither from being
    dropped for want of a clock.
    """
    for field in ("ts", "observed_at", "end_time", "start_time"):
        value = document.get(field)
        if value:
            return value
    return None


def _leg_rows(legs_dir: Path, models: Any) -> list[Any]:
    """Every leg document, through `read_documents`.

    NOT a glob. `compact-legs` bundles sealed records into one file, so a glob
    over `*.json` sees the bundle and misses everything inside it --
    `read_documents` reads both shapes, and the writer uses it too.

    TWO NAME SHAPES, and requiring the first silently dropped 4,591 of 13,440
    documents -- a third of the record, including every one of the 1,823
    `observed` legs, which are the only account of a death the node did not
    survive:

        <task>.<attempt>.start.json      per ATTEMPT -- a retry reuses the id
        <task>.<attempt>.exit.json
        <task>.progress.json             per TASK
        <task>.observed.json             per TASK, written by the READER
    """
    from src.shared.cloudtask import task_log  # noqa: PLC0415

    rows = []
    seen: set[tuple[str, int, str]] = set()
    for name, document in task_log.read_documents(legs_dir).items():
        stem = name[: -len(".json")] if name.endswith(".json") else name
        parts = stem.rsplit(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            task_id, attempt, leg = parts[0], int(parts[1]), parts[2]
        elif len(parts) >= 2:
            task_id, attempt, leg = stem.rsplit(".", 1)[0], task_history.TASK_SCOPED, parts[-1]
        else:
            continue
        key = (task_id, attempt, leg)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            models.Leg(
                task_id=task_id,
                attempt=attempt,
                leg=leg,
                run_id=document.get("run_id") or None,
                at=_leg_instant(document),
                body=document,
            )
        )
    return rows


def run(args: argparse.Namespace) -> BackfillPayload:
    """Read the published record and mirror it into Postgres."""
    dsn = os.environ.get(DSN_ENV)
    if not dsn:
        raise CommandError(
            f"{DSN_ENV} is unset. It carries a password, so it is read from the environment: "
            "export POKER_SOLVER_RECORD_DSN=$(terraform -chdir=infra/store output -raw postgres_dsn)"
        )

    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.dialects.postgresql import insert  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    from src.adapters.postgres import models  # noqa: PLC0415

    engine = sa.create_engine(dsn.replace("postgresql://", "postgresql+psycopg://", 1))
    payload = BackfillPayload(applied=bool(args.apply))

    # Per run, so a divergence names WHICH run rather than only a total. A
    # count that is right in aggregate and wrong per run is the failure this is
    # meant to catch.
    per_run: dict[str, tuple[int, int]] = {}

    with records_root(args) as root, Session(engine) as session:
        wanted = sorted(p for p in root.iterdir() if p.is_dir())
        if args.runs:
            names = set(args.runs)
            wanted = [p for p in wanted if p.name in names]

        for run_dir in wanted:
            built = _rows_for_run(run_dir, models)
            if built is None:
                payload.skipped.append(run_dir.name)
                continue
            run_row, event_rows, checkpoint_rows = built
            eval_rows = _eval_rows(run_dir, models)
            per_run[run_dir.name] = (len(event_rows), len(eval_rows))
            payload.share.runs += 1
            payload.share.events += len(event_rows)
            payload.share.checkpoints += len(checkpoint_rows)
            payload.share.evals += len(eval_rows)
            if not args.apply:
                continue

            # Upsert the run, insert-or-ignore everything else. Re-running must
            # be a no-op rather than a duplicate, which is what makes importing
            # before pruning safe.
            session.execute(
                insert(models.Run)
                .values(_columns(run_row, models.Run))
                .on_conflict_do_update(
                    index_elements=["run_id"],
                    set_={k: v for k, v in _columns(run_row, models.Run).items() if k != "run_id"},
                )
            )
            for table, rows in (
                (models.RunEvent, event_rows),
                (models.Checkpoint, checkpoint_rows),
                (models.Eval, eval_rows),
            ):
                if rows:
                    session.execute(
                        insert(table)
                        .values([_columns(r, table) for r in rows])
                        .on_conflict_do_nothing()
                    )
            session.commit()

    # A separate pass: legs are keyed by TASK and live under `legs/`, a
    # different top-level directory from the run record's `archive/`.
    from src.interfaces.cloud.config import CloudConfig  # noqa: PLC0415
    from src.interfaces.cloud.store import share  # noqa: PLC0415
    from src.interfaces.commands.tasks import download_tasks  # noqa: PLC0415
    from src.shared.cloudtask import task_log  # noqa: PLC0415

    config = CloudConfig.load()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp)
        download_tasks(share.share_client(config), config.share_name, local)
        leg_rows = _leg_rows(task_log.tasks_dir(local), models)
    payload.share.legs = len(leg_rows)
    if args.apply and leg_rows:
        with Session(engine) as session:
            for start in range(0, len(leg_rows), 500):
                chunk = leg_rows[start : start + 500]
                session.execute(
                    insert(models.Leg)
                    .values([_columns(r, models.Leg) for r in chunk])
                    .on_conflict_do_nothing()
                )
            session.commit()

    if args.verify:
        payload.verified = True
        payload.divergences = _divergences(engine, per_run, models)

    with Session(engine) as session:
        payload.database = Counts(
            runs=session.scalar(sa.select(sa.func.count()).select_from(models.Run)) or 0,
            events=session.scalar(sa.select(sa.func.count()).select_from(models.RunEvent)) or 0,
            checkpoints=session.scalar(sa.select(sa.func.count()).select_from(models.Checkpoint))
            or 0,
            evals=session.scalar(sa.select(sa.func.count()).select_from(models.Eval)) or 0,
            legs=session.scalar(sa.select(sa.func.count()).select_from(models.Leg)) or 0,
        )
    return payload


def _divergences(engine: Any, per_run: dict[str, tuple[int, int]], models: Any) -> list[Any]:
    """Where the share and the database disagree, per run.

    Counts rather than checksums, deliberately: the share is the source of
    truth through dual write and the database is allowed to hold MORE for a
    live run -- a node writes its rows the moment they happen, while the share
    sees them at the next publish. So a database ahead is normal and a database
    BEHIND is the bug, and only the second is reported.
    """
    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    with Session(engine) as session:
        events = {
            run: int(n)
            for run, n in session.execute(
                sa.select(models.RunEvent.run_id, sa.func.count()).group_by(models.RunEvent.run_id)
            )
        }
        evals = {
            run: int(n)
            for run, n in session.execute(
                sa.select(models.Eval.run_id, sa.func.count()).group_by(models.Eval.run_id)
            )
        }

    found = []
    for run, (n_events, n_evals) in sorted(per_run.items()):
        for kind, on_share, in_database in (
            ("events", n_events, events.get(run, 0)),
            ("evals", n_evals, evals.get(run, 0)),
        ):
            if in_database < on_share:
                found.append(
                    Divergence(run=run, kind=kind, on_share=on_share, in_database=in_database)
                )
    return found


def _columns(row: Any, table: Any) -> dict[str, Any]:
    """A model instance as a plain dict, with EVERY column present.

    Uniform on purpose: a multi-row `insert().values([...])` requires the same
    keys in every dict, and omitting the Nones made rows disagree about which
    columns they carried -- which SQLAlchemy reports as "explicitly rendered as
    a boundparameter", several layers from the cause.

    Columns whose value is None AND that carry a server default are dropped, so
    the default still applies; a nullable column keeps its explicit None.
    """
    values = {}
    for column in table.__table__.columns:
        value = getattr(row, column.name, None)
        if value is None and column.server_default is not None:
            continue
        values[column.name] = value
    return values


def render(payload: BackfillPayload) -> None:
    print(f"{'':<14}{'share':>10}{'database':>12}")
    for field in ("runs", "events", "checkpoints", "evals", "legs"):
        share = getattr(payload.share, field)
        database = getattr(payload.database, field)
        flag = "" if database >= share else "   <- gap"
        print(f"  {field:<12}{share:>10,}{database:>12,}{flag}")
    if payload.verified:
        if payload.divergences:
            print(f"\nDIVERGENT ({len(payload.divergences)}) -- the database is BEHIND the share:")
            for d in payload.divergences[:20]:
                print(
                    f"  {d.run[:52]:<54} {d.kind:<8} share {d.on_share:>5}  db {d.in_database:>5}"
                )
        else:
            print("\nverified: every run agrees, or the database is ahead (a live run)")
    if payload.skipped:
        print(f"\nunreadable, skipped ({len(payload.skipped)}):")
        for name in payload.skipped[:10]:
            print(f"  {name}")
    if not payload.applied:
        print("\nDRY RUN -- nothing was written. Re-run with --apply to import.")


COMMAND = Command(
    name="backfill-record",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Mirror the published record into Postgres (dry run by default; idempotent).",
)
