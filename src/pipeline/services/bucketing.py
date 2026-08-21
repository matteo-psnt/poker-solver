"""Producing card abstractions."""

import logging
from pathlib import Path

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.shared import records

PROGRESS_ARTIFACT = "precompute-progress.json"

logger = logging.getLogger(__name__)


def precompute_abstraction(
    abstraction_config: str,
    *,
    num_workers: int | None = None,
    base_dir: Path | None = None,
    overwrite: bool = False,
    progress_file: Path | None = None,
) -> Path:
    """Headless precompute of a combo abstraction; return the output directory.

    Output goes to ``<base_dir>/data/combo_abstraction/<name>`` (``base_dir`` defaults
    to the working directory, matching the resolver's lookup). Skips work if a complete
    abstraction already exists there unless ``overwrite`` is set.
    """
    config = PrecomputeConfig.from_yaml(abstraction_config)
    if num_workers is not None:
        config = config.model_copy(update={"num_workers": num_workers})
    out = abstraction_output_path(base_dir or Path.cwd(), config)
    if not overwrite and (out / "metadata.json").exists():
        return out
    precomputer = PostflopPrecomputer(config)
    precomputer.precompute_all(
        streets=[Street.FLOP, Street.TURN, Street.RIVER],
        # The only thing that reaches the outside before `save()`. That is also
        # why a precompute is never retried, so without a bar a multi-hour build
        # is opaque from the first second.
        on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
    )
    precomputer.save(out)
    return out
