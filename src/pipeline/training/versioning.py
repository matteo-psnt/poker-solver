"""Representation versioning for training runs.

A run's *representation version* identifies the on-disk checkpoint layout plus the
infoset-encoding format it was produced under. It is the anchor for detecting —
loudly, at build time — when a code change would silently render existing runs
unloadable (the failure mode that orphaned pre-``src.bucketing`` artifacts). A
committed golden run must keep loading under current code, so a format break fails
the test suite instead of orphaning runs; ``migrations`` provides the forward path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.engine.solver.storage.helpers import load_checkpoint_arrays

# Bump when a change alters the on-disk run/checkpoint format or the infoset
# encoding such that an existing run can no longer be loaded/continued as-is.
# A bump is only safe alongside a migration or a documented barrier; the
# golden-run test is what forces that discipline.
REPRESENTATION_VERSION = 2

RUN_METADATA_FILE = ".run.json"


def run_representation_version(run_dir: Path) -> int:
    """Representation version stamped in a run's metadata; 0 for pre-versioning runs."""
    meta_path = Path(run_dir) / RUN_METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"No run metadata at {meta_path}")
    with open(meta_path) as f:
        data = json.load(f)
    return int(data.get("representation_version", 0))


def checkpoint_fingerprint(run_dir: Path) -> str:
    """Content hash of a run's checkpoint arrays (learned policy + regrets).

    Deterministic over the strategic content, independent of key serialization or
    metadata. This is the verification primitive for *exact* migrations (which must
    leave the fingerprint unchanged) and the strong form of the golden tripwire (the
    fixture must load to the *same* strategies, not merely load).
    """
    arrays = load_checkpoint_arrays(Path(run_dir))
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        arr = arrays[name]
        hasher.update(name.encode())
        hasher.update(str(arr.shape).encode())
        hasher.update(str(arr.dtype).encode())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def ensure_current(run_dir: Path) -> None:
    """Raise a clear, actionable error unless the run is at the current version.

    Loader-facing guard: turns "silently loaded a stale/mismatched run" into an
    explicit instruction to migrate (or a hard stop for runs from newer code).
    """
    version = run_representation_version(run_dir)
    if version == REPRESENTATION_VERSION:
        return
    if version > REPRESENTATION_VERSION:
        raise ValueError(
            f"Run {run_dir} is at representation version {version}, newer than this "
            f"code supports ({REPRESENTATION_VERSION}). Update the code."
        )
    raise ValueError(
        f"Run {run_dir} is at representation version {version}; current is "
        f"{REPRESENTATION_VERSION}. Migrate it first: "
        f"migrate_run(src, dst) from src.pipeline.training.migrations."
    )
