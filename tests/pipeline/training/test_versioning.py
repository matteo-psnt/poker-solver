"""Enforcement tripwire for representation versioning.

A committed golden run must keep loading under current code. If a change to the
checkpoint format, the infoset encoding (e.g. moving/renaming ``InfoSetKey``), or
the config schema breaks loading this frozen run, these tests fail — turning a
silent orphaning of existing runs into a loud build failure. See
``src/pipeline/training/versioning.py``.

If a break here is intentional, the fix is a migration or a documented barrier plus
a refreshed golden fixture — never just deleting the tripwire.
"""

import json
from pathlib import Path

from src.engine.solver.storage.in_memory import InMemoryStorage
from src.pipeline.training.run_tracker import RunMetadata
from src.pipeline.training.versioning import (
    REPRESENTATION_VERSION,
    run_representation_version,
)

GOLDEN_RUN = Path(__file__).parents[2] / "fixtures" / "golden_run"


def test_golden_run_metadata_loads_under_current_schema():
    """Metadata (including the inline config) deserializes under the current schema."""
    meta = RunMetadata.load(GOLDEN_RUN / ".run.json")
    assert meta.num_infosets > 0
    assert meta.config.system.config_name == "quick_test"


def test_golden_run_is_at_current_version():
    """The committed golden run is stamped at the current representation version."""
    assert run_representation_version(GOLDEN_RUN) == REPRESENTATION_VERSION


def test_golden_run_checkpoint_still_loads():
    """The checkpoint (zarr arrays + pickled InfoSetKey mapping) still loads.

    This is the fragility tripwire: moving/renaming ``InfoSetKey``, changing the
    zarr layout, or breaking the pickle formats fails here instead of silently
    rendering every existing run unloadable.
    """
    storage = InMemoryStorage(checkpoint_dir=GOLDEN_RUN)
    assert storage.num_infosets() > 0


def test_missing_version_reads_as_legacy_zero(tmp_path):
    """A run without a version stamp (pre-versioning) reads as 0, not the default."""
    (tmp_path / ".run.json").write_text(json.dumps({"run_id": "legacy"}))
    assert run_representation_version(tmp_path) == 0
