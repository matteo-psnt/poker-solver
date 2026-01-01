"""Storage section editor for CLI config."""

from src.interfaces.cli.flows.config_helpers import try_merge
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_storage_settings(ctx: CliContext, config: Config) -> Config:
    print("Storage Settings")
    print("-" * 40)
    ui.info("These settings affect memory usage and checkpoint I/O performance.")

    initial_capacity = prompts.prompt_int(
        ctx,
        "Initial infoset capacity (grows automatically if exceeded):",
        default=config.storage.initial_capacity,
        min_value=10_000,
    )
    if initial_capacity is None:
        return config

    max_actions = prompts.prompt_int(
        ctx,
        "Max actions stored per infoset:",
        default=config.storage.max_actions,
        min_value=2,
    )
    if max_actions is None:
        return config

    zarr_compression = prompts.prompt_int(
        ctx,
        "Zarr compression level (1=fastest I/O, 9=smallest files):",
        default=config.storage.zarr_compression_level,
        min_value=1,
        max_value=9,
    )
    if zarr_compression is None:
        return config

    zarr_chunk = prompts.prompt_int(
        ctx,
        "Zarr chunk size in infosets (10K-100K typical; larger = faster sequential reads):",
        default=config.storage.zarr_chunk_size,
        min_value=1_000,
    )
    if zarr_chunk is None:
        return config

    return try_merge(
        config,
        {
            "storage": {
                "initial_capacity": initial_capacity,
                "max_actions": max_actions,
                "zarr_compression_level": zarr_compression,
                "zarr_chunk_size": zarr_chunk,
            }
        },
    )
