"""Representation-migration registry and applier.

The registry is a linear chain: ``MIGRATIONS[i]`` produces version ``i+1``, and the
top migration's version must equal ``REPRESENTATION_VERSION``. ``migrate_run`` brings
an old run forward into a new directory (functional — the original is never mutated —
and atomic: it plans the whole path first so a barrier/gap fails before any copy, and
rolls back the destination on any failure).

This enforces that a run at any version is either current, migratable, or explicitly
barriered — never silently unloadable.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.pipeline.training.migrations.base import (
    Migration,
    MigrationBarrierError,
    MigrationKind,
)
from src.pipeline.training.migrations.m0001_introduce_versioning import MIGRATION as _M0001
from src.pipeline.training.migrations.m0002_float32_hot_arrays import MIGRATION as _M0002
from src.pipeline.training.migrations.m0003_columnar_key_table import MIGRATION as _M0003
from src.pipeline.training.versioning import (
    REPRESENTATION_VERSION,
    RUN_METADATA_FILE,
    run_representation_version,
)

__all__ = [
    "MIGRATIONS",
    "Migration",
    "MigrationBarrierError",
    "MigrationKind",
    "migrate_run",
    "plan_migration",
    "validate_registry",
]

# Ordered chain of representation migrations.
MIGRATIONS: list[Migration] = [_M0001, _M0002, _M0003]


def validate_registry(migrations: list[Migration] | None = None) -> None:
    """The registry must be a gapless 1..N chain whose top matches REPRESENTATION_VERSION."""
    migrations = MIGRATIONS if migrations is None else migrations
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(migrations) + 1)):
        raise ValueError(f"Migration versions must be contiguous 1..N; got {versions}")
    if migrations and migrations[-1].version != REPRESENTATION_VERSION:
        raise ValueError(
            f"Top migration version {migrations[-1].version} != REPRESENTATION_VERSION "
            f"{REPRESENTATION_VERSION}; bump one to match."
        )


def plan_migration(from_version: int, migrations: list[Migration] | None = None) -> list[Migration]:
    """Ordered steps to bring a run from ``from_version`` to current.

    Raises ``MigrationBarrierError`` if the path crosses a barrier, ``ValueError`` on a
    missing step or a run newer than the code.
    """
    migrations = MIGRATIONS if migrations is None else migrations
    if from_version > REPRESENTATION_VERSION:
        raise ValueError(
            f"Run is at version {from_version}, newer than code ({REPRESENTATION_VERSION})."
        )
    by_version = {m.version: m for m in migrations}
    steps: list[Migration] = []
    for v in range(from_version + 1, REPRESENTATION_VERSION + 1):
        step = by_version.get(v)
        if step is None:
            raise ValueError(f"No migration registered producing version {v}")
        if step.kind is MigrationKind.BARRIER:
            raise MigrationBarrierError(
                f"Cannot migrate across the v{v} barrier: {step.reason}. "
                "Retrain from a current run instead."
            )
        steps.append(step)
    return steps


def _stamp_and_record(run_dir: Path, step: Migration) -> None:
    meta_path = run_dir / RUN_METADATA_FILE
    with open(meta_path) as f:
        data = json.load(f)
    data["representation_version"] = step.version
    history = data.setdefault("migration_history", [])
    history.append(
        {
            "version": step.version,
            "description": step.description,
            "kind": step.kind.value,
            "applied_at": datetime.now().isoformat(),
        }
    )
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


def migrate_run(src: Path, dst: Path, migrations: list[Migration] | None = None) -> Path:
    """Migrate a run to the current version into a NEW directory; original untouched.

    Plans the whole chain first (a barrier/gap fails before any disk write), then
    copies ``src`` → ``dst`` and applies each step on the copy, stamping the version and
    recording history per step. Rolls ``dst`` back on any failure.

    Exception-safe, not kill-safe: a hard kill mid-run can leave a partial ``dst``
    behind. It still fails loud — it carries a stale version stamp, so loaders
    refuse it and a retry hits the exists-check — but it must be deleted by hand.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"Migration destination already exists: {dst}")
    steps = plan_migration(run_representation_version(src), migrations)

    shutil.copytree(src, dst)
    try:
        for step in steps:
            if step.migrate is not None:
                step.migrate(dst)
            _stamp_and_record(dst, step)
            if step.verify is not None:
                step.verify(dst)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    return dst


validate_registry()
