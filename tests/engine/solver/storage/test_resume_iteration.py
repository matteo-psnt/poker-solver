"""Resume must trust the checkpoint manifest, not run metadata.

A checkpoint write is two steps: ``commit_checkpoint_manifest`` publishes the
manifest (carrying its ``iteration``) atomically with the arrays it names, then
``.run.json`` is rewritten separately and non-atomically. A hard kill between them
-- SIGKILL, OOM, the Modal guillotine -- leaves metadata behind the data on disk.
Resuming from the stale metadata re-runs ``[metadata, manifest)``, and because the
deal stream is a pure function of the absolute iteration index, that replays deals
already baked into the loaded arrays and double-counts them.
"""

import json

from src.engine.solver.storage.helpers import (
    CHECKPOINT_MANIFEST_FILE,
    resolve_resume_iteration,
)


def _write_manifest(checkpoint_dir, iteration: int) -> None:
    (checkpoint_dir / CHECKPOINT_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "iteration": iteration,
                "zarr": f"checkpoint-{iteration}.zarr",
                "key_table": f"keys-{iteration}",
            }
        )
    )


def test_manifest_wins_when_metadata_is_stale(tmp_path):
    """The crash window: data committed at 15_640_000, metadata still at 10_000_000."""
    _write_manifest(tmp_path, 15_640_000)
    assert resolve_resume_iteration(tmp_path, metadata_iterations=10_000_000) == 15_640_000


def test_agreeing_sources_resolve_to_that_iteration(tmp_path):
    _write_manifest(tmp_path, 10_000_000)
    assert resolve_resume_iteration(tmp_path, metadata_iterations=10_000_000) == 10_000_000


def test_falls_back_to_metadata_for_pre_manifest_runs(tmp_path):
    """Older runs have no manifest; metadata is then the only source."""
    assert resolve_resume_iteration(tmp_path, metadata_iterations=250_000) == 250_000


def test_no_checkpoint_resolves_to_none(tmp_path):
    assert resolve_resume_iteration(tmp_path, metadata_iterations=0) is None
