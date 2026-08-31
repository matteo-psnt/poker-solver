"""The run record as declarative tables -- the source of truth for the schema.

Declarative rather than SQLAlchemy Core, on a measurement rather than a taste:
`ty` resolves `Eval.scor_mbb` to an attribute error and does NOT catch
`evals.c.scor_mbb`, because Core's `.c` is a dynamic ColumnCollection. Core
would cost a dependency and buy none of the safety that justifies it.

These are STORAGE models and deliberately not payload models. The console's
contract is generated from its own pydantic models; fusing the two would make a
column rename a console change.

Analytical queries -- anything with a join, an aggregate or a window -- stay as
`text()` SQL beside the reader that owns them. The paired-within-seed comparison
IS a join, and a builder obscures it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One training run. Columns, not JSONB, for anything a LISTING filters on."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    config_name: Mapped[str] = mapped_column(String)
    kernel: Mapped[str | None] = mapped_column(String)
    arm: Mapped[str | None] = mapped_column(String)
    experiment_id: Mapped[str | None] = mapped_column(String)
    parent_run_id: Mapped[str | None] = mapped_column(String)

    action_config_hash: Mapped[str | None] = mapped_column(String)
    card_abstraction_hash: Mapped[str | None] = mapped_column(String)
    config_hash: Mapped[str | None] = mapped_column(String)

    # Lossless, because `verify_trainer_knobs` compares whole config blocks by
    # EQUALITY. It is also the payload a resume needs and the one thing a blob
    # listing can never reconstruct.
    config: Mapped[dict] = mapped_column(JSONB)

    git_commit: Mapped[str | None] = mapped_column(String)
    git_dirty: Mapped[bool | None] = mapped_column(Boolean)
    git_branch: Mapped[str | None] = mapped_column(String)  # a commit is not an arm
    code_snapshot: Mapped[str | None] = mapped_column(String)

    storage_capacity: Mapped[int | None] = mapped_column(BigInteger)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # `running` by default on purpose: a sweeper that reads absence as terminal
    # deletes a live ladder.
    status: Mapped[str] = mapped_column(String, server_default="running")
    iterations: Mapped[int] = mapped_column(BigInteger, server_default="0")
    num_infosets: Mapped[int | None] = mapped_column(BigInteger)
    runtime_seconds: Mapped[float] = mapped_column(Float, server_default="0")

    __table_args__ = (
        Index(
            "runs_experiment", "experiment_id", postgresql_where=text("experiment_id IS NOT NULL")
        ),
        Index("runs_listing", text("started_at DESC")),
    )


class RunEvent(Base):
    """The event log. `run.jsonl`, with the two jobs of `seq` pulled apart."""

    __tablename__ = "run_events"

    # THE global ordering, and the only thing the SSE cursor reads. Server
    # assigned: a per-run monotone counter cannot also be a global cursor, which
    # the spike proved by needing a second column before the query could be
    # written at all.
    gseq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"))
    # Load-bearing: the run-vs-attempt `status` scoping exists because an
    # unscoped fold read two live runs as `died`.
    attempt: Mapped[int] = mapped_column(Integer, server_default="0")
    event: Mapped[str] = mapped_column(String)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[dict] = mapped_column(JSONB)

    # Idempotent replay ONLY. Two processes write one run's events -- the node
    # wrapper and the trainer subprocess -- so a client-computed monotone seq
    # would collide and a sink specified "must not raise" would swallow it.
    event_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))

    iteration: Mapped[int | None] = mapped_column(
        BigInteger, Computed("(body->>'iteration')::bigint", persisted=True)
    )

    __table_args__ = (
        Index("run_events_replay", "event_uuid", unique=True),
        Index("run_events_fold", "run_id", "attempt", "gseq"),
        Index("run_events_kind", "run_id", "event", text("gseq DESC")),
        Index(
            "run_events_iter", "run_id", "iteration", postgresql_where=text("iteration IS NOT NULL")
        ),
    )


class Checkpoint(Base):
    """A retained rung. A DERIVED CACHE of what Blob holds -- never a claim.

    One rung is one atomically-committed blob, so blob existence IS
    completeness. A `complete` column here would put the advertisement in
    Postgres and the bytes in Blob: two systems that can disagree, and one
    direction of disagreement is the state `require_complete` exists to refuse.
    """

    __tablename__ = "checkpoints"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
    )
    iteration: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    blob_uri: Mapped[str] = mapped_column(String)
    bytes: Mapped[int | None] = mapped_column(BigInteger)

    # WITHOUT THESE TWO, TWO GUARDS GO UN-ARMED: a wrong-tree or rebucketed load
    # does not fail, it silently reinterprets every row as a different infoset.
    fingerprint: Mapped[str] = mapped_column(String)
    abstraction_id: Mapped[str | None] = mapped_column(String)

    is_current: Mapped[bool] = mapped_column(Boolean, server_default="false")
    written_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    __table_args__ = (
        Index("checkpoints_current", "run_id", unique=True, postgresql_where=text("is_current")),
    )


class Eval(Base):
    """One scored evaluation.

    `tier` is NOT a column. It is an ordinal into a ranked query result --
    `--tier 0` means "position 0" and renumbers when a row is added -- so as a
    column it is either meaningless or changes on insert. Pairing goes through
    `tier_digest` over ~30 knobs, computed in Python by the same `tier_key` path
    that has always owned the rule.
    """

    __tablename__ = "evals"

    eval_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"))

    # The curve's x-axis. A row without one is unplaceable rather than merely
    # incomplete, so NULL is a real value here.
    checkpoint_iteration: Mapped[int | None] = mapped_column(BigInteger)

    method: Mapped[str] = mapped_column(String)
    # `base_seed`, NOT `board_seed`: exact_br mirrors one into the other, and a
    # column by the latter name matches no row ever written.
    base_seed: Mapped[int | None] = mapped_column(Integer)

    # Source of truth for the tier. Conditional knobs are written only when
    # non-default and compared only when present on one side, so promoting them
    # to NOT NULL columns would collapse absent into explicit-false and re-pair
    # every legacy row. They stay here, three-valued.
    knobs: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    tier_digest: Mapped[str] = mapped_column(String)

    exploitability_mbb: Mapped[float | None] = mapped_column(Float)
    std_error_mbb: Mapped[float | None] = mapped_column(Float)
    num_hands: Mapped[int | None] = mapped_column(BigInteger)  # a pairing PRECONDITION
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSONB, server_default="{}")

    __table_args__ = (
        Index("evals_pairing", "tier_digest", "run_id", "checkpoint_iteration"),
        Index("evals_run", "run_id", "checkpoint_iteration"),
    )


class Leg(Base):
    """One record of a task's life, from either writer.

    `observed` is the BATCH half of the join -- the only account of a death the
    node did not survive. Without somewhere to put it every OOM-kill and node
    loss stays unresolved forever, which cascades into `cost`, `runinfo` and
    `tasks` re-asking Batch on every poll.

    `cause` is deliberately NOT a column: it is derived at read time from
    (exit.cause, observed.state) by one precedence rule, and materialising it
    would freeze that rule into whichever writer happened to run first.
    """

    __tablename__ = "legs"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    # A Batch retry REUSES the task id, so the attempt is part of the key.
    attempt: Mapped[int] = mapped_column(Integer, primary_key=True, server_default="0")
    leg: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[dict] = mapped_column(JSONB)

    __table_args__ = (
        Index("legs_run", "run_id", postgresql_where=text("run_id IS NOT NULL")),
        # Measured: 7.45ms on a seq scan, 1.59ms with this.
        Index("legs_leg_at", "leg", text("at DESC")),
    )


class Progress(Base):
    """Live progress. Four artifacts with THREE key shapes, so the scope is explicit.

    `train-progress` is per run, `legs/*.progress` per TASK (not per attempt --
    it goes stale rather than absent), and `precompute-progress` is per
    ABSTRACTION and has no run at all.
    """

    __tablename__ = "progress"

    scope: Mapped[str] = mapped_column(String, primary_key=True)  # run|task|abstraction
    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    op: Mapped[str] = mapped_column(String, primary_key=True)
    unit: Mapped[str | None] = mapped_column(String)
    base: Mapped[int | None] = mapped_column(BigInteger)
    done: Mapped[int | None] = mapped_column(BigInteger)
    total: Mapped[int | None] = mapped_column(BigInteger)
    rate: Mapped[float | None] = mapped_column(Float)
    # Without it the ETA is a guess rather than a measured window rate.
    window_seconds: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class Abstraction(Base):
    """A precomputed card abstraction.

    Keyed by ABSTRACTION, not by run -- which is why a run-centric schema had
    nowhere to put it. The resolver raises rather than resolve past this,
    because doing so drops the config_hash a checkpoint is pinned to.
    """

    __tablename__ = "abstractions"

    abstraction_id: Mapped[str] = mapped_column(String, primary_key=True)
    config_hash: Mapped[str] = mapped_column(String)
    shape: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    built_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
