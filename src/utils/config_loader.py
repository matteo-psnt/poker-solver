"""
Configuration loading - YAML overrides applied to dataclass defaults.

Simple, clean, no duplication. All in one file.
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.config import Config, merge_dataclass_config


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """
    Load configuration from YAML file with optional programmatic overrides.

    Args:
        path: Optional path to YAML file
        **overrides: Additional overrides in Python (e.g., training__num_iterations=50000)

    Returns:
        Config with all overrides applied

    Examples:
        >>> # Use all defaults
        >>> cfg = load_config()

        >>> # Load from YAML
        >>> cfg = load_config("config/training/default.yaml")

        >>> # Load from YAML + programmatic override
        >>> cfg = load_config("config/training/default.yaml", training__num_iterations=50000)

        >>> # Pure programmatic (no YAML)
        >>> cfg = load_config(training__num_iterations=50000)
    """
    # Start with defaults
    config = Config()

    # Apply YAML if provided
    if path is not None:
        yaml_overrides = _load_yaml(Path(path))
        config = _merge_config(config, yaml_overrides)

    # Apply programmatic overrides (e.g., from CLI)
    if overrides:
        nested_overrides = _flatten_to_nested(overrides)
        config = _merge_config(config, nested_overrides)

    return config


def _merge_config(base: Any, overrides: dict[str, Any]) -> Any:
    """Strict schema merge."""
    return merge_dataclass_config(base, overrides, strict=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return data or {}


def _flatten_to_nested(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert flat dict with '__' separators to nested dict.

    Example:
        {"training__num_iterations": 50000}
        ->
        {"training": {"num_iterations": 50000}}
    """
    result: dict[str, Any] = {}

    for key, value in flat_dict.items():
        parts = key.split("__")

        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set final value
        current[parts[-1]] = value

    return result
