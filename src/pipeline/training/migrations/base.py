"""Migration contract for representation/checkpoint versioning.

A migration produces version ``version`` from ``version - 1`` (the chain is linear).
Migrations are *functional* — the applier runs them on a copy, never the original —
and each declares a ``kind`` that sets how strictly it is verified:

- ``EXACT``: preserves the learned strategy (module renames, storage reformat,
  metadata-only changes). ``verify`` must confirm the checkpoint fingerprint is
  unchanged against a value recorded at authoring time.
- ``APPROXIMATE``: cannot preserve strategy exactly (e.g. an action-size addition
  that widens arrays). ``verify`` only confirms the result loads and is coherent;
  LBR — never used here — measures the warm-start quality separately.
- ``BARRIER``: an intentional break with no transform (e.g. a new abstraction, whose
  bucket→hand information the checkpoint discarded). Carries a ``reason``; the applier
  refuses to cross it and tells the caller to retrain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MigrationKind(Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    BARRIER = "barrier"


class MigrationBarrierError(RuntimeError):
    """Raised when a migration path crosses an intentional barrier (retrain instead)."""


@dataclass(frozen=True)
class Migration:
    """One step in the linear migration chain, producing ``version`` from ``version-1``."""

    version: int
    description: str
    kind: MigrationKind
    # Transform applied in-place on the working copy. None = no data change (the
    # applier still stamps the new version). Must be None for BARRIER.
    migrate: Callable[[Path], None] | None = None
    # Post-migration invariant check on the working copy. Must be None for BARRIER.
    verify: Callable[[Path], None] | None = None
    # Required for BARRIER: why the break was not worth migrating.
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind is MigrationKind.BARRIER:
            if self.migrate is not None or self.verify is not None:
                raise ValueError(
                    f"Barrier migration v{self.version} must not define migrate/verify"
                )
            if not self.reason:
                raise ValueError(f"Barrier migration v{self.version} must carry a reason")
