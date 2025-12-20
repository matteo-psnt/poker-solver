"""
Configuration management for poker solver.

Loads and validates YAML configuration files for training runs.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Configuration manager for poker solver.

    Loads YAML config files and provides typed access to settings.
    Supports config inheritance and overrides.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Config instance
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(config_dict)

    @classmethod
    def default(cls) -> "Config":
        """
        Get default configuration.

        Returns:
            Config with default settings
        """
        return cls(get_default_config())

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key (supports dot notation).

        Args:
            key: Config key (e.g., 'game.starting_stack')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire config section.

        Args:
            section: Section name (e.g., 'game', 'solver')

        Returns:
            Section dictionary
        """
        return self._config.get(section, {})

    def set(self, key: str, value: Any):
        """
        Set config value by key (supports dot notation).

        Args:
            key: Config key (e.g., 'game.starting_stack')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

    def merge(self, other: "Config"):
        """
        Merge another config into this one (other takes precedence).

        Args:
            other: Config to merge
        """
        self._config = _deep_merge(self._config, other._config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Get config as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ValueError: If config is invalid
        """
        # Validate game config
        game = self.get_section("game")
        if game.get("starting_stack", 0) <= 0:
            raise ValueError("starting_stack must be positive")
        if game.get("small_blind", 0) <= 0:
            raise ValueError("small_blind must be positive")
        if game.get("big_blind", 0) <= 0:
            raise ValueError("big_blind must be positive")

        # Validate training config
        training = self.get_section("training")
        if training.get("num_iterations", 0) <= 0:
            raise ValueError("num_iterations must be positive")
        if training.get("checkpoint_frequency", 0) <= 0:
            raise ValueError("checkpoint_frequency must be positive")

        # Validate storage config
        storage = self.get_section("storage")
        if storage.get("backend") not in ["memory", "disk"]:
            raise ValueError("storage backend must be 'memory' or 'disk'")

    def __str__(self) -> str:
        """String representation."""
        return f"Config({len(self._config)} sections)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Config({self._config})"


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.

    Returns:
        Default configuration
    """
    return {
        "game": {
            "starting_stack": 200,
            "small_blind": 1,
            "big_blind": 2,
        },
        "action_abstraction": {
            "preflop_raises": [2.5, 3.5, 5.0],
            "postflop": {
                "flop": [0.33, 0.66, 1.25],
                "turn": [0.50, 1.0, 1.5],
                "river": [0.50, 1.0, 2.0],
            },
            "all_in_spr_threshold": 2.0,
        },
        "card_abstraction": {
            "type": "equity_bucketing",
            "config": "production",
        },
        "spr_buckets": {
            "thresholds": [4, 13],
        },
        "solver": {
            "type": "mccfr",
            "sampling_method": "external",
        },
        "training": {
            "num_iterations": 1000,
            "checkpoint_frequency": 100,
            "log_frequency": 10,
        },
        "storage": {
            "backend": "disk",
            "cache_size": 100000,
            "flush_frequency": 1000,
        },
        "system": {
            "seed": None,
            "log_level": "INFO",
        },
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: Optional[Path] = None, **overrides) -> Config:
    """
    Load configuration with optional overrides.

    Args:
        config_path: Path to YAML config (uses default if None)
        **overrides: Key-value overrides (dot notation supported)

    Returns:
        Config instance

    Example:
        >>> config = load_config("config.yaml", training__num_iterations=10000)
    """
    if config_path is None:
        config = Config.default()
    else:
        config = Config.from_file(config_path)

    # Apply overrides
    for key, value in overrides.items():
        # Convert __ to . for dot notation
        key = key.replace("__", ".")
        config.set(key, value)

    config.validate()
    return config


def list_training_configs() -> list[str]:
    """
    List all available training configurations.

    Returns:
        List of config names (without .yaml extension)

    Example:
        >>> configs = list_training_configs()
        >>> print(configs)
        ['default', 'production', 'fast_test', 'quick_test']
    """
    config_dir = Path(__file__).parent.parent.parent / "config" / "training"
    if not config_dir.exists():
        return []

    config_files = sorted(config_dir.glob("*.yaml"))
    return [f.stem for f in config_files]


def load_training_config(name: str) -> Config:
    """
    Load a training configuration by name from config/training/.

    Args:
        name: Config name (without .yaml extension)

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If config file not found

    Example:
        >>> config = load_training_config("production")
        >>> config.get("training.num_iterations")
        100000000
    """
    config_dir = Path(__file__).parent.parent.parent / "config" / "training"
    config_path = config_dir / f"{name}.yaml"

    if not config_path.exists():
        available = list_training_configs()
        raise FileNotFoundError(
            f"Training config '{name}' not found.\n"
            f"Available configs: {', '.join(available)}\n"
            f"Place config files in config/training/"
        )

    return Config.from_file(config_path)
