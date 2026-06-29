"""Tests for combo-abstraction resolution across metadata schema drift."""

from __future__ import annotations

import json

import pytest

from src.pipeline.abstraction.resolver import (
    AbstractionMetadataError,
    ComboAbstractionResolver,
    _read_metadata,
)


def _write_abstraction(base_dir, name: str, config: dict) -> None:
    """Create a fake precomputed abstraction directory with given metadata config."""
    path = base_dir / f"buckets-{name}"
    path.mkdir()
    (path / "combo_abstraction.pkl").write_bytes(b"placeholder")
    (path / "metadata.json").write_text(json.dumps({"config": config}))


def test_resolves_abstraction_with_drifted_metadata(tmp_path):
    """A saved abstraction is matched by name even if its config schema drifted.

    Regression: the resolver used to strictly re-parse the saved config and, on
    any drift (e.g. renamed fields), silently skip the directory and report
    'no abstraction found' — even though the abstraction was present.
    """
    abstractions_dir = tmp_path / "combo_abstraction"
    abstractions_dir.mkdir()
    # Metadata shaped like a real pre-refactor snapshot: legacy field names plus
    # an embedded config_hash the current schema would reject.
    _write_abstraction(
        abstractions_dir,
        "quick",
        {
            "config_name": "quick_test",
            "num_board_clusters": {"FLOP": 10, "TURN": 20, "RIVER": 30},
            "num_buckets": {"FLOP": 10, "TURN": 20, "RIVER": 30},
            "config_hash": "deadbeef",
        },
    )

    loaded = object()
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=lambda path: loaded,  # ty: ignore[invalid-argument-type]
    )

    result = resolver.load(abstraction_config="quick_test")
    assert result is loaded


def test_missing_abstraction_still_raises(tmp_path):
    """An unknown abstraction name still fails clearly."""
    abstractions_dir = tmp_path / "combo_abstraction"
    abstractions_dir.mkdir()
    _write_abstraction(abstractions_dir, "other", {"config_name": "something_else"})

    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=lambda path: object(),  # ty: ignore[invalid-argument-type]
    )

    with pytest.raises(FileNotFoundError):
        resolver.load(abstraction_config="quick_test")


def test_corrupt_metadata_is_not_read_as_absent(tmp_path):
    """A truncated metadata.json must raise, not degrade to "no metadata".

    The distinction is a provenance one. `config_hash_for` reads this file to
    learn which buckets a checkpoint was pinned to; when unreadable collapsed
    into absent, the pin was silently dropped and evaluation proceeded against
    buckets it could no longer prove were the trained ones.
    """
    path = tmp_path / "buckets-quick"
    path.mkdir()
    (path / "metadata.json").write_text('{"config": {"config_name": "quick_te')

    with pytest.raises(AbstractionMetadataError, match="could not be read"):
        _read_metadata(path)


def test_absent_metadata_is_still_none(tmp_path):
    """Absence is the one case that stays None."""
    path = tmp_path / "buckets-quick"
    path.mkdir()

    assert _read_metadata(path) is None


def test_one_corrupt_directory_does_not_hide_the_others(tmp_path):
    """Discovery skips a damaged sibling rather than failing the whole scan."""
    abstractions_dir = tmp_path / "combo_abstraction"
    abstractions_dir.mkdir()
    broken = abstractions_dir / "buckets-broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{not json")
    _write_abstraction(abstractions_dir, "quick", {"config_name": "quick_test"})

    loaded = object()
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=lambda path: loaded,  # ty: ignore[invalid-argument-type]
    )

    assert resolver.load(abstraction_config="quick_test") is loaded
