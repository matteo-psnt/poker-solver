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
    """A copy of the golden run with the version stamp stripped (a legacy v0 run).

    Also drops any recorded migration history: the fixture is itself produced by
    migrating, so leaving its history in place would make the chain assertions
    below compare against entries this test never applied.
    """
    shutil.copytree(GOLDEN_RUN, dst)
    meta_path = dst / ".run.json"
    data = json.loads(meta_path.read_text())
    data.pop("representation_version", None)
    data.pop("migration_history", None)
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
    # The v0 copy already stores float32 arrays (fixture refreshed at v2), so the
    # chain (version stamp + dtype downcast) leaves strategic content byte-identical.
    assert checkpoint_fingerprint(dst) == before
    # Original is untouched (functional).
    assert run_representation_version(src) == 0
    # History recorded, one entry per chain step.
    history = json.loads((dst / ".run.json").read_text())["migration_history"]
    assert [h["version"] for h in history] == list(range(1, REPRESENTATION_VERSION + 1))
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

    failing = [
        Migration(version=1, description="ok", kind=MigrationKind.EXACT),
        Migration(version=2, description="bad", kind=MigrationKind.EXACT, verify=_boom),
        Migration(version=3, description="unreached", kind=MigrationKind.EXACT),
    ]
    src = _v0_copy(tmp_path / "legacy")
    dst = tmp_path / "migrated"

    with pytest.raises(RuntimeError, match="verify failed"):
        migrate_run(src, dst, migrations=failing)

    assert not dst.exists()  # rolled back, no partial output


def _v2_with_manifest(dst: Path) -> Path:
    """A pre-v3 run in the *manifested* layout: two pickles named by CHECKPOINT.json.

    The golden fixture predates manifests, so migrating it only ever exercises the
    fixed-name path. A real production run carries a manifest, and reading one with
    the current schema check fails on the missing 'key_table' -- the exact failure
    this reconstructs.
    """
    import pickle

    from src.engine.solver.storage import key_table

    shutil.copytree(GOLDEN_RUN, dst)
    rows = key_table.read_all_rows(dst / "keys")
    iteration = 4242

    with open(dst / f"key_mapping-{iteration}.pkl", "wb") as f:
        pickle.dump({"owned_keys": {key: i for i, key in enumerate(rows.keys)}}, f)
    with open(dst / f"action_signatures-{iteration}.pkl", "wb") as f:
        pickle.dump(
            {
                i: [(a.type.name, a.amount) for a in actions]
                for i, actions in enumerate(rows.action_lists)
            },
            f,
        )
    (dst / "checkpoint.zarr").rename(dst / f"checkpoint-{iteration}.zarr")
    shutil.rmtree(dst / "keys")
    (dst / "CHECKPOINT.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "zarr": f"checkpoint-{iteration}.zarr",
                "key_mapping": f"key_mapping-{iteration}.pkl",
                "action_signatures": f"action_signatures-{iteration}.pkl",
            }
        )
    )
    meta_path = dst / ".run.json"
    data = json.loads(meta_path.read_text())
    data["representation_version"] = 2
    data.pop("migration_history", None)
    meta_path.write_text(json.dumps(data))
    return dst


def _v1_with_manifest(dst: Path) -> Path:
    """A manifested run still at v1: float64 hot arrays + pickled keys.

    Upcasts the hot arrays back to the float64 that v1 stored; float64→float32
    of float32-derived values is lossless, so the full chain must reproduce the
    golden fingerprint.
    """
    import zarr

    from src.engine.solver.storage.array_specs import ARRAY_SPECS

    _v2_with_manifest(dst)

    store = zarr.DirectoryStore(dst / "checkpoint-4242.zarr")
    root = zarr.open(store, mode="r")
    attrs = dict(root.attrs)
    arrays = {spec.checkpoint_key: root[spec.checkpoint_key][:] for spec in ARRAY_SPECS}
    new_root = zarr.open(store, mode="w")
    for spec in ARRAY_SPECS:
        data = arrays[spec.checkpoint_key]
        if spec.per_action:
            data = data.astype("float64")
        new_root.create_dataset(spec.checkpoint_key, data=data)
    new_root.attrs.update(attrs)

    meta_path = dst / ".run.json"
    data = json.loads(meta_path.read_text())
    data["representation_version"] = 1
    meta_path.write_text(json.dumps(data))
    return dst


def test_migrate_manifested_v1_run(tmp_path):
    """m0002 must read a pre-v3 manifest (which lacks 'key_table'); without that,
    a manifested v1 run is neither current, nor migratable, nor barriered —
    breaking the trichotomy the registry promises."""
    src = _v1_with_manifest(tmp_path / "v1")

    dst = migrate_run(src, tmp_path / "migrated")

    assert run_representation_version(dst) == REPRESENTATION_VERSION
    assert checkpoint_fingerprint(dst) == checkpoint_fingerprint(GOLDEN_RUN)
    manifest = json.loads((dst / "CHECKPOINT.json").read_text())
    assert manifest["key_table"] == "keys-4242"


def test_migrate_v2_run_with_manifest(tmp_path):
    """A manifested v2 run converts, and the manifest is rewritten to point at the table."""
    from src.engine.solver.storage import key_table
    from src.engine.solver.storage.in_memory import InMemoryStorage

    src = _v2_with_manifest(tmp_path / "v2")

    dst = migrate_run(src, tmp_path / "migrated")

    assert run_representation_version(dst) == REPRESENTATION_VERSION
    # Fingerprinting the v2 source is impossible by design -- loaders are
    # current-layout only -- so compare against the run it was derived from:
    # a full golden -> v2 -> v3 round trip must leave the arrays untouched.
    assert checkpoint_fingerprint(dst) == checkpoint_fingerprint(GOLDEN_RUN)

    manifest = json.loads((dst / "CHECKPOINT.json").read_text())
    assert manifest["key_table"] == "keys-4242"
    assert "key_mapping" not in manifest and "action_signatures" not in manifest
    assert not list(dst.glob("*.pkl")), "legacy pickles should be removed"

    # The converted table must load and still carry every row.
    assert (
        key_table.num_rows(dst / "keys-4242") == InMemoryStorage(checkpoint_dir=dst).num_infosets()
    )
