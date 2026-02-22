"""Shared CLI context and paths."""

from dataclasses import dataclass
from pathlib import Path

from questionary import Style

from src.interfaces.cli.ui.theme import STYLE


@dataclass
class CliContext:
    """Shared CLI context for paths and style."""

    base_dir: Path
    config_dir: Path
    runs_dir: Path
    equity_buckets_dir: Path
    style: Style

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to project root when needed."""
        path_obj = Path(path).expanduser()
        if not path_obj.is_absolute():
            path_obj = self.base_dir / path_obj
        return path_obj.resolve()

    def set_runs_dir(self, runs_dir: str | Path) -> None:
        """Update the active runs directory used by CLI run-related flows."""
        self.runs_dir = self.resolve_path(runs_dir)

    @classmethod
    def from_project_root(cls, base_dir: Path | None = None) -> "CliContext":
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[4]

        base_dir = base_dir.resolve()

        return cls(
            base_dir=base_dir,
            config_dir=base_dir / "config",
            runs_dir=base_dir / "data" / "runs",
            equity_buckets_dir=base_dir / "data" / "equity_buckets",
            style=STYLE,
        )
