"""Loading a missing or stale abstraction artifact surfaces actionable errors."""

from __future__ import annotations

import json

import pytest

from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


def test_missing_artifact_raises_friendly_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Precompute Combo Abstraction"):
        PostflopPrecomputer.load(tmp_path)


def test_old_storage_version_raises_friendly_error(tmp_path):
    """Artifacts from an older storage layout must ask for regeneration."""
    (tmp_path / "metadata.json").write_text(json.dumps({"config": {}, "statistics": {}}))

    with pytest.raises(RuntimeError, match=r"[Rr]egenerate"):
        PostflopPrecomputer.load(tmp_path)
