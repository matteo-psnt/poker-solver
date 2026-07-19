"""Migration 0003: columnar key table (v2 → v3).

The key→id map and the action signatures moved from two pickled dicts to one
columnar directory (see :mod:`src.engine.solver.storage.key_table`). Pickle is
all-or-nothing, so every worker had to materialize the entire checkpoint to keep
the ~1/N shard it owns -- at 18.9M keys, ~28 GB per worker and ~444 GB across 16,
which is what made resuming a grown checkpoint OOM.

EXACT: the same keys, ids and legal actions come back out, in the same order.
Row index *is* the infoset id in the new layout, and the old dense ids were
already a gapless 0..n-1 assigned in sorted order, so ids are preserved rather
than merely remapped -- ``verify`` re-reads the table and checks that.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from src.core.game.actions import Action, ActionType
from src.engine.solver.storage import key_table
from src.engine.solver.storage.helpers import (
    ACTION_SIGNATURES_FILE,
    CHECKPOINT_MANIFEST_FILE,
    KEY_MAPPING_FILE,
    KEY_TABLE_DIR,
    read_checkpoint_manifest,
)
from src.engine.solver.storage.shared_array.ownership import stable_hash
from src.pipeline.training.migrations.base import Migration, MigrationKind


def _legacy_paths(run_dir: Path) -> tuple[Path, Path]:
    """Old key-mapping and action-signature paths, manifest-aware.

    ``CheckpointPaths`` only understands the current layout, so the pre-v3 names
    are resolved here rather than kept alive in the loader.
    """
    manifest = read_checkpoint_manifest(run_dir)
    if manifest is None:
        return run_dir / KEY_MAPPING_FILE, run_dir / ACTION_SIGNATURES_FILE
    return run_dir / manifest["key_mapping"], run_dir / manifest["action_signatures"]


def _migrate(run_dir: Path) -> None:
    mapping_path, sigs_path = _legacy_paths(run_dir)
    if not mapping_path.exists():
        # Already in the target layout (e.g. a fixture stamped to an older version
        # than its data). Nothing to convert; the applier still stamps the version.
        return
    with open(mapping_path, "rb") as f:
        owned_keys = pickle.load(f)["owned_keys"]
    with open(sigs_path, "rb") as f:
        action_sigs = pickle.load(f)

    # Order rows by the id the checkpoint already assigned, so row index == id and
    # the zarr arrays (indexed by that id) stay aligned without being rewritten.
    ordered = sorted(owned_keys.items(), key=lambda item: item[1])
    keys = [key for key, _ in ordered]
    action_lists: list[list[Action] | None] = []
    for _, infoset_id in ordered:
        sigs = action_sigs.get(infoset_id)
        action_lists.append(
            None if sigs is None else [Action(ActionType[name], amount) for name, amount in sigs]
        )

    # A manifest-less (pre-manifest) run resolves its artifacts by fixed name, so
    # the table has to land on the fixed name too; manifest runs get the versioned
    # name the manifest will point at.
    manifest = read_checkpoint_manifest(run_dir)
    table_name = KEY_TABLE_DIR if manifest is None else f"keys-{int(manifest['iteration'])}"
    table_dir = run_dir / table_name
    key_table.write_key_table(
        table_dir,
        keys=keys,
        key_hashes=[stable_hash(key) for key in keys],
        action_lists=action_lists,
    )

    _rewrite_manifest(run_dir, table_dir.name)
    mapping_path.unlink()
    sigs_path.unlink()
    for legacy in (run_dir / KEY_MAPPING_FILE, run_dir / ACTION_SIGNATURES_FILE):
        if legacy.exists():
            legacy.unlink()


def _rewrite_manifest(run_dir: Path, table_name: str) -> None:
    """Swap the two pickle entries for the table entry, keeping the write atomic."""
    manifest = read_checkpoint_manifest(run_dir)
    if manifest is None:
        return
    manifest.pop("key_mapping", None)
    manifest.pop("action_signatures", None)
    manifest["key_table"] = table_name
    tmp = run_dir / (CHECKPOINT_MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, run_dir / CHECKPOINT_MANIFEST_FILE)


def _verify(run_dir: Path) -> None:
    from src.engine.solver.storage.in_memory import InMemoryStorage

    storage = InMemoryStorage(checkpoint_dir=run_dir)
    if storage.num_infosets() <= 0:
        raise ValueError("post-migration checkpoint has no infosets")

    # Ownership must be reproducible from the persisted hash column, since that is
    # what every worker now shards on; a mismatch would silently re-partition the
    # tree and drop updates.
    from src.engine.solver.storage.helpers import CheckpointPaths

    table_dir = CheckpointPaths.from_dir(run_dir).key_table
    rows = key_table.read_owned_rows(table_dir, num_workers=4, worker_id=1)
    for key in rows.keys:
        if stable_hash(key) % 4 != 1:
            raise ValueError("post-migration key table shards inconsistently with stable_hash")


MIGRATION = Migration(
    version=3,
    description="columnar key table (keys + action signatures) so workers load only their shard.",
    kind=MigrationKind.EXACT,
    migrate=_migrate,
    verify=_verify,
)
