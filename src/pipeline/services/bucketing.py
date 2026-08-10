"""Producing and measuring card abstractions."""

import logging
from collections.abc import Sequence
from pathlib import Path

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.shared import records

PROGRESS_ARTIFACT = "precompute-progress.json"

logger = logging.getLogger(__name__)


def sweep_bucket_counts(
    abstraction_config: str,
    street: Street,
    bucket_counts: Sequence[int],
    *,
    num_workers: int | None = None,
    board_limit: int | None = None,
) -> list[dict]:
    """Quality metrics for one street bucketed at several bucket counts.

    Answers "how much resolution does bucket count k actually buy?" without
    paying for k separate precomputes: the equity pass — which is essentially
    all of the cost — runs once, and each bucket count is then a cheap k-means
    over the same matrices. ``PostflopPrecomputer`` was already factored for
    this; ``compute_street_matrices`` and ``bucket_street``'s ``num_buckets``
    override exist precisely so a sweep can reuse one pass.

    Nothing is written to disk. This measures an abstraction rather than
    producing one, and a sweep that also emitted artifacts would invite loading
    a bucketing that no run's ``card_abstraction_hash`` accounts for.

    Returns one quality dict per entry of ``bucket_counts``, each carrying the
    ``num_buckets`` it was measured at.
    """
    config = PrecomputeConfig.from_yaml(abstraction_config)
    if num_workers is not None:
        config = config.model_copy(update={"num_workers": num_workers})

    precomputer = PostflopPrecomputer(config)
    board_ids, equity, weights, histograms = precomputer.compute_street_matrices(
        street, board_limit=board_limit
    )

    results: list[dict] = []
    for count in bucket_counts:
        quality = precomputer.bucket_street(
            street, board_ids, equity, weights, histograms, num_buckets=count
        )
        results.append({"requested_buckets": int(count), **quality})
        logger.info(
            f"{street.name} @ {count} buckets: "
            f"occupied {quality['occupied_buckets']}/{quality['num_buckets']}, "
            f"variance explained {quality['variance_explained']:.6f}, "
            f"within-bucket std {quality['within_bucket_std']:.6f}"
        )
    return results


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
        # Street completion, which is the only thing that reaches the outside
        # before `save()`. That is also why a precompute is never retried, so
        # without a bar a multi-hour build is opaque from the first second.
        on_street_done=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
    )
    precomputer.save(out)
    return out
