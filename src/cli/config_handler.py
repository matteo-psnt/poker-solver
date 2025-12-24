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

    checkpoint_freq = questionary.text(
        "Checkpoint frequency:",
        default=str(config.get("training.checkpoint_frequency", 100)),
        style=custom_style,
    ).ask()

    config.set("training.num_iterations", int(iterations))
    config.set("training.checkpoint_frequency", int(checkpoint_freq))

    # Use combo abstraction config-based approach
    default_config = config.get("card_abstraction.config", "default")

    config_name = questionary.text(
        "Combo abstraction config name:",
        default=default_config,
        style=custom_style,
    ).ask()

    config.set("card_abstraction.config", config_name)

    # Check if abstraction exists
    base_path = Path("data/combo_abstraction")

    abstraction_found = False
    if base_path.exists():
        for path in base_path.iterdir():
            if path.is_dir() and (path / "combo_abstraction.pkl").exists():
                abstraction_found = True
                break

    if not abstraction_found:
        print("\n[!] Warning: No combo abstraction found.")
        precompute = questionary.confirm(
            "Would you like to precompute combo abstraction now?",
            default=True,
            style=custom_style,
        ).ask()

        if precompute:
            from src.cli.combo_handler import handle_combo_precompute

            handle_combo_precompute()

    return config
