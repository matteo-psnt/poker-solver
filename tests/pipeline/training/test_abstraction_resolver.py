"""Tests for combo-abstraction resolution across metadata schema drift."""

from __future__ import annotations

import json

import pytest

from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver


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
        loader=lambda path: loaded,
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
        loader=lambda path: object(),
    )

    with pytest.raises(FileNotFoundError):
        resolver.load(abstraction_config="quick_test")
