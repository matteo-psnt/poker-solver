"""Tests for the representation-migration applier and registry."""

import json
import shutil
from pathlib import Path

import pytest

from src.pipeline.training.migrations import (
    Migration,
    MigrationBarrierError,
    MigrationKind,
    migrate_run,
    plan_migration,
    validate_registry,
)
from src.pipeline.training.versioning import (
    REPRESENTATION_VERSION,
    checkpoint_fingerprint,
    run_representation_version,
)

GOLDEN_RUN = Path(__file__).parents[2] / "fixtures" / "golden_run"


def _v0_copy(dst: Path) -> Path:
    """A copy of the golden run with the version stamp stripped (a legacy v0 run)."""
    shutil.copytree(GOLDEN_RUN, dst)
    meta_path = dst / ".run.json"
    data = json.loads(meta_path.read_text())
    data.pop("representation_version", None)
    meta_path.write_text(json.dumps(data))
    return dst


# --- registry -----------------------------------------------------------------


def test_real_registry_is_valid():
    validate_registry()  # gapless 1..N, top == REPRESENTATION_VERSION


def test_barrier_contract_rejects_transform():
    with pytest.raises(ValueError, match="must not define migrate"):
        Migration(
            version=1,
            description="x",
            kind=MigrationKind.BARRIER,
            reason="r",
            migrate=lambda p: None,
        )
    with pytest.raises(ValueError, match="must carry a reason"):
        Migration(version=1, description="x", kind=MigrationKind.BARRIER)


# --- planning -----------------------------------------------------------------


def test_plan_from_legacy_reaches_current():
    steps = plan_migration(0)
    assert [s.version for s in steps] == list(range(1, REPRESENTATION_VERSION + 1))


def test_plan_from_current_is_empty():
    assert plan_migration(REPRESENTATION_VERSION) == []


def test_plan_rejects_newer_than_code():
    with pytest.raises(ValueError, match="newer than code"):
        plan_migration(REPRESENTATION_VERSION + 1)


def test_plan_raises_on_gap():
    with pytest.raises(ValueError, match="No migration registered"):
        plan_migration(0, migrations=[])  # nothing produces v1


def test_plan_raises_barrier():
    barrier = [
        Migration(
            version=1, description="break", kind=MigrationKind.BARRIER, reason="new abstraction"
        )
    ]
    with pytest.raises(MigrationBarrierError, match="new abstraction"):
        plan_migration(0, migrations=barrier)


# --- applying -----------------------------------------------------------------


def test_migrate_legacy_run_to_current(tmp_path):
    src = _v0_copy(tmp_path / "legacy")
    assert run_representation_version(src) == 0
    before = checkpoint_fingerprint(src)

    dst = migrate_run(src, tmp_path / "migrated")

    assert run_representation_version(dst) == REPRESENTATION_VERSION
    # EXACT migration: strategic content is byte-identical.
    assert checkpoint_fingerprint(dst) == before
    # Original is untouched (functional).
    assert run_representation_version(src) == 0
    # History recorded.
    history = json.loads((dst / ".run.json").read_text())["migration_history"]
    assert [h["version"] for h in history] == [REPRESENTATION_VERSION]
    assert history[-1]["kind"] == "exact"


def test_migrate_current_run_is_noop_copy(tmp_path):
    dst = migrate_run(GOLDEN_RUN, tmp_path / "out")
    assert run_representation_version(dst) == REPRESENTATION_VERSION
    assert checkpoint_fingerprint(dst) == checkpoint_fingerprint(GOLDEN_RUN)


def test_migrate_refuses_existing_dst(tmp_path):
    dst = tmp_path / "out"
    dst.mkdir()
    with pytest.raises(FileExistsError):
        migrate_run(GOLDEN_RUN, dst)


def test_migrate_rolls_back_on_failure(tmp_path):
    def _boom(_run_dir):
        raise RuntimeError("verify failed")

    failing = [Migration(version=1, description="bad", kind=MigrationKind.EXACT, verify=_boom)]
    src = _v0_copy(tmp_path / "legacy")
    dst = tmp_path / "migrated"

    with pytest.raises(RuntimeError, match="verify failed"):
        migrate_run(src, dst, migrations=failing)

    assert not dst.exists()  # rolled back, no partial output
