"""Deterministic filesystem paths for precomputed combo abstractions.

Kept in the pipeline layer (not the interactive CLI) so headless and cloud
precompute paths compute the same directory names the resolver scans for.
"""

from __future__ import annotations

from pathlib import Path

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig

OUTPUT_HASH_LENGTH = 8


def output_config_hash(config: PrecomputeConfig) -> str:
    """Return the normalized short hash used in output directory names."""
    return config.get_config_hash()[:OUTPUT_HASH_LENGTH]


def abstraction_output_path(base_dir: Path, config: PrecomputeConfig) -> Path:
    """Deterministic output directory for an abstraction under ``base_dir``.

    Mirrors ``data/combo_abstraction/<name>`` so the resolver (which scans that
    directory relative to the working directory) finds the result.
    """
    runouts_tag = "exact" if config.flop_runouts is None else str(config.flop_runouts)
    dirname = (
        f"buckets-F{config.num_buckets[Street.FLOP]}T{config.num_buckets[Street.TURN]}"
        f"R{config.num_buckets[Street.RIVER]}-"
        f"r{runouts_tag}-{output_config_hash(config)}"
    )
    return base_dir / "data" / "combo_abstraction" / dirname
