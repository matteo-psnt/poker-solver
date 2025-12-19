"""Configuration handling for CLI."""

from pathlib import Path
from typing import Optional

import questionary

from src.utils.config import Config


def select_config(config_dir: Path, custom_style) -> Optional[Config]:
    """
    Select and optionally edit a config file.

    Args:
        config_dir: Base config directory (e.g., "config/")
        custom_style: Questionary style

    Returns:
        Loaded Config object or None if cancelled
    """
    # Look for training configs in config/training/ subdirectory
    training_config_dir = config_dir / "training"
    config_files = sorted(training_config_dir.glob("*.yaml"))

    if not config_files:
        print(f"[ERROR] No config files found in {training_config_dir}/")
        return None

    choices = [f.stem for f in config_files] + ["Cancel"]

    selected = questionary.select(
        "Select configuration:",
        choices=choices,
        style=custom_style,
    ).ask()

    if selected == "Cancel" or selected is None:
        return None

    config_path = training_config_dir / f"{selected}.yaml"
    config = Config.from_file(config_path)

    edit = questionary.confirm(
        "Edit configuration before running?",
        default=False,
        style=custom_style,
    ).ask()

    if edit:
        config = edit_config(config, custom_style)

    return config


def edit_config(config: Config, custom_style) -> Config:
    """
    Interactive config editor.

    Args:
        config: Config to edit
        custom_style: Questionary style

    Returns:
        Modified config
    """
    print("\nEdit Configuration")
    print("-" * 40)

    iterations = questionary.text(
        "Number of iterations:",
        default=str(config.get("training.num_iterations", 1000)),
        style=custom_style,
    ).ask()

    # Card abstraction is now always equity_bucketing
    abstraction_type = "equity_bucketing"

    checkpoint_freq = questionary.text(
        "Checkpoint frequency:",
        default=str(config.get("training.checkpoint_frequency", 100)),
        style=custom_style,
    ).ask()

    config.set("training.num_iterations", int(iterations))
    config.set("card_abstraction.type", abstraction_type)
    config.set("training.checkpoint_frequency", int(checkpoint_freq))

    if abstraction_type == "equity_bucketing":
        default_path = "data/abstractions/equity_buckets.pkl"
        bucketing_path = questionary.text(
            "Equity bucketing file path:",
            default=config.get("card_abstraction.bucketing_path", default_path),
            style=custom_style,
        ).ask()

        config.set("card_abstraction.bucketing_path", bucketing_path)

        if not Path(bucketing_path).exists():
            print(f"\n[!] Warning: Equity bucketing file not found: {bucketing_path}")
            precompute = questionary.confirm(
                "Would you like to precompute equity buckets now?",
                default=True,
                style=custom_style,
            ).ask()

            if precompute:
                from src.cli.precompute_handler import handle_precompute

                handle_precompute(custom_style)

    return config
