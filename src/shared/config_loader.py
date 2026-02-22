"""
Configuration loading — YAML overrides applied to Pydantic model defaults.

Single source of truth: defaults live in Python (config.py), YAML files only
specify overrides. YAML files may declare `extends: <filename>` to inherit from
another YAML in the same directory; the current file's values always win.
"""

from pathlib import Path
from typing import Any

import yaml

from src.shared.config import Config
from src.shared.dicts import deep_merge_dicts


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """
    Load configuration from an optional YAML file with optional programmatic overrides.

    Resolution order (last wins):
      1. Python field defaults (always the base)
      2. YAML file (resolved via ``extends`` chain if present)
      3. Programmatic keyword overrides

    Args:
        path: Optional path to a YAML config file.
        **overrides: Programmatic overrides using ``__`` as a nesting separator,
            e.g. ``training__num_iterations=50_000``.

    Returns:
        Validated, frozen :class:`Config` instance.

    Examples:
        >>> cfg = load_config()
        >>> cfg = load_config("config/training/quick_test.yaml")
        >>> cfg = load_config("config/training/quick_test.yaml", training__num_iterations=500)
        >>> cfg = load_config(training__num_iterations=500, game__big_blind=4)
    """
    config = Config.default()

    # Apply YAML overrides (with extends resolution)
    if path is not None:
        yaml_data = _load_yaml(Path(path))
        config = config.merge(yaml_data)

    # Apply programmatic overrides (take precedence over everything)
    if overrides:
        nested_overrides = _flatten_to_nested(overrides)
        config = config.merge(nested_overrides)

    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and recursively resolve any ``extends`` chain.

    A YAML file may declare::

        extends: base.yaml

    to inherit from another YAML in the same directory. Chains are supported
    (A extends B extends C). The current file's values always win over the base.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if "extends" in data:
        base_path = path.parent / data.pop("extends")
        base_data = _load_yaml(base_path)  # recursive — supports chains
        data = deep_merge_dicts(base_data, data)  # current file wins

    return data


def _flatten_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a flat dict with ``__`` separators into a nested dict.

    Example::

        {"training__num_iterations": 50_000}
        →  {"training": {"num_iterations": 50_000}}
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split("__")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
