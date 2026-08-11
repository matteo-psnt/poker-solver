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
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.interfaces.commands._base import Command, records_root
from src.interfaces.errors import CommandError
from src.shared import run_events
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

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


class BackfillPayload(BaseModel):
    op: Literal["backfill-record"] = "backfill-record"
    applied: bool = False
    share: Counts = Field(default_factory=Counts)
    database: Counts = Field(default_factory=Counts)
    skipped: list[str] = Field(default_factory=list)


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

    created = run_events.head(events) or {}
    run = models.Run(
        run_id=run_dir.name,
        config_name=metadata.config_name or created.get("config_name") or "",
        kernel=created.get("kernel"),
        arm=created.get("arm"),
        experiment_id=created.get("experiment_id"),
        parent_run_id=created.get("parent_run_id"),
        action_config_hash=created.get("action_config_hash"),
        card_abstraction_hash=created.get("card_abstraction_hash"),
        config_hash=created.get("config_hash"),
        # Lossless: `verify_trainer_knobs` compares whole config blocks.
        config=created.get("config") or {},
        git_commit=created.get("git_commit"),
        git_dirty=created.get("git_dirty"),
        git_branch=created.get("git_branch"),
        code_snapshot=created.get("code_snapshot"),
        storage_capacity=created.get("storage_capacity"),
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


def _eval_rows(run_dir: Path, models: Any) -> list[Any]:
    """Every eval document under one run, as rows."""
    rows = []
    evals = run_dir / "evals"
    if not evals.is_dir():
        return rows
    for path in sorted(evals.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
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

    with Session(engine) as session:
        payload.database = Counts(
            runs=session.scalar(sa.select(sa.func.count()).select_from(models.Run)) or 0,
            events=session.scalar(sa.select(sa.func.count()).select_from(models.RunEvent)) or 0,
            checkpoints=session.scalar(sa.select(sa.func.count()).select_from(models.Checkpoint))
            or 0,
            evals=session.scalar(sa.select(sa.func.count()).select_from(models.Eval)) or 0,
        )
    return payload


def _columns(row: Any, table: Any) -> dict[str, Any]:
    """A model instance as a plain dict of set columns, for the insert."""
    return {
        column.name: getattr(row, column.name)
        for column in table.__table__.columns
        if getattr(row, column.name, None) is not None
    }


def render(payload: BackfillPayload) -> None:
    print(f"{'':<14}{'share':>10}{'database':>12}")
    for field in ("runs", "events", "checkpoints", "evals"):
        share = getattr(payload.share, field)
        database = getattr(payload.database, field)
        flag = "" if database >= share else "   <- gap"
        print(f"  {field:<12}{share:>10,}{database:>12,}{flag}")
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
