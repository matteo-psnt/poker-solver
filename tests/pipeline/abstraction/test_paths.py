"""Tests for deterministic abstraction output paths."""

from pathlib import Path

from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path, output_config_hash


def test_output_path_matches_resolver_dirname():
    """The output dir name must match what the resolver scans for (quick_test)."""
    config = PrecomputeConfig.from_yaml("quick_test")
    out = abstraction_output_path(Path("/repo"), config)

    assert out.parent == Path("/repo/data/combo_abstraction")
    # Encodes buckets, board clusters, equity samples, and the short config hash — the
    # exact scheme the on-disk quick_test abstraction (…-5879c364) uses.
    assert out.name == "buckets-F10T20R30-C10C20C30-s100-5879c364"


def test_output_config_hash_is_short_prefix():
    """The dir hash is an 8-char prefix of the full config hash."""
    config = PrecomputeConfig.from_yaml("quick_test")
    short = output_config_hash(config)

    assert len(short) == 8
    assert config.get_config_hash().startswith(short)
