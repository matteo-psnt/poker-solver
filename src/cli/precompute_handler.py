"""Precomputation handler."""

import multiprocessing as mp
from pathlib import Path

import questionary
import yaml

from src.abstraction.manager import EquityBucketManager
from src.abstraction.metadata import EquityBucketMetadata, compute_config_hash
from src.abstraction.precompute import (
    PrecomputeConfig,
    precompute_equity_bucketing,
    print_summary,
)
from src.game.state import Street


def _check_if_exists(config_dict: dict) -> tuple[bool, str]:
    """Check if equity bucket with this config already exists.

    Returns:
        Tuple of (exists, display_name)
    """
    # Create temporary metadata to compute hash
    # Convert Street enum keys to string keys to match stored metadata
    num_buckets = {}
    num_board_clusters = {}

    for key, value in config_dict["num_buckets"].items():
        if isinstance(key, Street):
            num_buckets[key.name] = value
        else:
            num_buckets[str(key)] = value

    for key, value in config_dict["num_board_clusters"].items():
        if isinstance(key, Street):
            num_board_clusters[key.name] = value
        else:
            num_board_clusters[str(key)] = value

    temp_metadata = EquityBucketMetadata(
        name="temp",
        created_at="",
        abstraction_type="equity_bucketing",
        num_buckets=num_buckets,
        num_board_clusters=num_board_clusters,
        num_equity_samples=config_dict["num_equity_samples"],
        num_samples_per_cluster=config_dict["num_samples_per_cluster"],
        seed=config_dict.get("seed", 42),
    )

    config_hash = compute_config_hash(temp_metadata)

    # Check all existing abstractions for matching hash
    manager = EquityBucketManager()
    abstractions = manager.list_abstractions()

    for name, path, metadata in abstractions:
        if compute_config_hash(metadata) == config_hash:
            return True, config_dict.get("name", "Unknown")

    return False, config_dict.get("name", "Unknown")


def handle_precompute(style):
    """Handle precomputation."""
    # Discover available config files
    config_dir = Path("config/equity_buckets")

    if not config_dir.exists():
        print(f"\nError: Config directory not found: {config_dir}")
        input("\nPress Enter...")
        return

    # Find all YAML config files and check if they exist
    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        print(f"\nError: No config files found in {config_dir}")
        input("\nPress Enter...")
        return

    # Build choices with status indicators
    choices = []
    config_status = {}

    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)

            exists, display_name = _check_if_exists(config_dict)
            config_status[config_file.stem] = exists

            if exists:
                choices.append(f"{config_file.stem} (already precomputed)")
            else:
                choices.append(config_file.stem)
        except Exception:
            choices.append(config_file.stem)
            config_status[config_file.stem] = False

    choices.append("Cancel")

    config_choice = questionary.select(
        "Configuration:",
        choices=choices,
        style=style,
    ).ask()

    if not config_choice or config_choice == "Cancel":
        return

    # Extract config name (remove status suffix)
    config_name = config_choice.replace(" (already precomputed)", "")

    # Load config from YAML file
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"\nError: Config file not found: {config_path}")
        input("\nPress Enter...")
        return

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Check if exists and confirm overwrite
    if config_status.get(config_name, False):
        print("\nThis configuration has already been precomputed.")
        overwrite = questionary.confirm(
            "Do you want to overwrite the existing abstraction?",
            default=False,
            style=style,
        ).ask()

        if not overwrite:
            print("\nCancelled.")
            input("\nPress Enter...")
            return

    # Override num_workers with CPU count
    config_dict["num_workers"] = mp.cpu_count()
    config_dict["config_name"] = config_name

    config = PrecomputeConfig.from_dict(config_dict)

    print_summary(config)

    if questionary.confirm("Proceed?", default=True, style=style).ask():
        try:
            precompute_equity_bucketing(config)
            print("\nDone!")
        except Exception as e:
            print(f"\nError: {e}")

    input("\nPress Enter...")
