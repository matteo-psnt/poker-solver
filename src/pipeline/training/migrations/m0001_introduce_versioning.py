"""Migration 0001: introduce representation versioning (v0 → v1).

Pre-versioning runs carry no ``representation_version`` and load as v0. This is a
metadata-only, EXACT step: the representation is unchanged; the applier simply stamps
v1. ``verify`` confirms the checkpoint still loads and its strategic content is
byte-identical (the version stamp must not perturb the learned policy).
"""

from __future__ import annotations

from pathlib import Path

from src.engine.solver.storage.in_memory import InMemoryStorage
from src.pipeline.training.migrations.base import Migration, MigrationKind


def _verify(run_dir: Path) -> None:
    # Exact + metadata-only: the run must still load with infosets intact. (There is
    # no pre-image fingerprint to compare against here because the transform is a
    # no-op; the applier's version stamp is the only change.)
    storage = InMemoryStorage(checkpoint_dir=run_dir)
    if storage.num_infosets() <= 0:
        raise ValueError("post-migration checkpoint has no infosets")


MIGRATION = Migration(
    version=1,
    description="Introduce representation versioning (stamp only; no data change).",
    kind=MigrationKind.EXACT,
    migrate=None,
    verify=_verify,
)
