"""What a headless subcommand is, and the helpers they share.

A :class:`Command` carries its parser, its handler AND its renderer together.
That is the point of the type: a subcommand used to be assembled from an
``_add_*_parser`` in one module and a ``RENDERERS`` entry in another, so one
could exist without the other -- which is how ``checkpoint-profile`` came to
borrow the evaluate renderer and die on a missing key. Here it cannot be
registered without all three.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.jsonio import json_default


@dataclass(frozen=True)
class Command:
    """One `poker-solver-run` subcommand."""

    name: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], dict[str, Any]]
    render: Callable[[dict[str, Any]], None]
    help: str = ""


def write_result(run_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a per-operation result file (e.g. ``train-static_result.json``)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{payload['op']}_result.json").write_text(
        json.dumps(payload, indent=2, default=json_default)
    )


def resolve_run_dir(run: str, runs_dir: str) -> Path:
    """Resolve a run identifier (name under ``runs_dir``) or an explicit path."""
    as_path = Path(run)
    if as_path.is_dir():
        return as_path
    candidate = Path(runs_dir) / run
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run not found: '{run}' (looked at {as_path} and {candidate})")


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set key__path=value`` into the config loader's override kwargs.

    Values go through JSON so ``1000``/``true``/``null`` arrive as the types the
    strict config models require; anything JSON rejects stays a plain string.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides
