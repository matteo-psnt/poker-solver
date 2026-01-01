"""Shared helpers for interactive CLI config editing."""

from __future__ import annotations

from pydantic import ValidationError

from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def list_abstraction_configs(ctx: CliContext) -> list[str]:
    """List available card abstraction config names from ``config/abstraction``."""
    abstraction_config_dir = ctx.config_dir / "abstraction"
    if not abstraction_config_dir.exists():
        return []
    return sorted(f.stem for f in abstraction_config_dir.glob("*.yaml") if f.is_file())


def try_merge(config: Config, overrides: dict) -> Config:
    """
    Attempt ``config.merge(overrides)`` and preserve the original on validation errors.
    """
    try:
        return config.merge(overrides)
    except ValidationError as exc:
        ui.error("Invalid configuration - changes not applied:")
        for error in exc.errors():
            field = " -> ".join(str(p) for p in error["loc"])
            print(f"  {field}: {error['msg']}")
        return config


def edit_list_of_floats(ctx: CliContext, prompt: str, default: list[float]) -> list[float] | None:
    """Prompt for a comma-separated list of positive floats."""
    default_str = ", ".join(str(x) for x in default)

    def validate(value: str) -> bool | str:
        try:
            parts = [float(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return "Invalid number - use comma-separated values like 0.33, 0.66, 1.25"
        if not parts:
            return "Enter at least one value"
        if any(x <= 0 for x in parts):
            return "All values must be positive"
        return True

    result = prompts.text(ctx, prompt, default=default_str, validate=validate)
    if result is None:
        return None

    try:
        return [float(x.strip()) for x in result.split(",") if x.strip()]
    except ValueError:
        return None
