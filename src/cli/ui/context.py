"""Shared CLI context and paths."""

from dataclasses import dataclass
from pathlib import Path

from questionary import Style

from src.cli.ui.theme import STYLE


@dataclass(frozen=True)
class CliContext:
    """Shared CLI context for paths and style."""

    base_dir: Path
    config_dir: Path
    runs_dir: Path
    equity_buckets_dir: Path
    style: Style

    @classmethod
    def from_project_root(cls, base_dir: Path | None = None) -> "CliContext":
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[3]

        base_dir = base_dir.resolve()

        return cls(
            base_dir=base_dir,
            config_dir=base_dir / "config",
            runs_dir=base_dir / "data" / "runs",
            equity_buckets_dir=base_dir / "data" / "equity_buckets",
            style=STYLE,
        )
