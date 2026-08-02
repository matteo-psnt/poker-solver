"""The `checkpoint-profile` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
)
from src.shared import checkpoint_profile


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run checkpoint-profile`."""
    parser.add_argument("--run", required=True, help="Run id to summarize.")
    parser.add_argument(
        "--runs-dir", default="data/runs", help="Directory containing run directories."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Summarize a run's per-checkpoint phase timings and the Volume commit."""
    run_dir = Path(args.runs_dir) / args.run
    path = run_dir / checkpoint_profile.PROFILE_FILENAME
    if not path.exists():
        raise SystemExit(
            f"No checkpoint profile at {path}. It is written per checkpoint, so the "
            "run must have checkpointed at least once with profiling in place."
        )

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    checkpoints = [r for r in rows if r.get("event") != "volume_commit"]
    commits = [r for r in rows if r.get("event") == "volume_commit"]

    phase_totals: dict[str, float] = {}
    for row in checkpoints:
        for name, secs in row.get("phases", {}).items():
            phase_totals[name] = phase_totals.get(name, 0.0) + secs

    checkpoint_seconds = sum(r["total_seconds"] for r in checkpoints)
    commit_seconds = sum(r["total_seconds"] for r in commits)
    # storage_write wraps the engine-level phases, so counting it alongside them
    # would double-count; collect_keys and storage_write are the top-level split.
    top_level = {k: v for k, v in phase_totals.items() if k in ("collect_keys", "storage_write")}

    return {
        "op": "checkpoint-profile",
        "run": args.run,
        "num_checkpoints": len(checkpoints),
        "checkpoint_seconds": round(checkpoint_seconds, 2),
        "volume_commit_seconds": round(commit_seconds, 2),
        "total_seconds": round(checkpoint_seconds + commit_seconds, 2),
        "commit_share": (
            round(commit_seconds / (checkpoint_seconds + commit_seconds), 3)
            if checkpoint_seconds + commit_seconds > 0
            else None
        ),
        "top_level_phases": {k: round(v, 2) for k, v in sorted(top_level.items())},
        "write_phases": {
            k: round(v, 2)
            for k, v in sorted(phase_totals.items(), key=lambda kv: -kv[1])
            if k not in ("collect_keys", "storage_write")
        },
        "checkpoints": checkpoints,
        "volume_commits": commits,
    }


def render(payload: dict[str, Any]) -> None:
    print(f"Checkpoint profile for {payload['run']}")
    print(f"  Checkpoints: {payload['num_checkpoints']}")
    print(f"  Writing:     {payload['checkpoint_seconds']:.2f}s")
    share = payload["commit_share"]
    commit = f"  Committing:  {payload['volume_commit_seconds']:.2f}s"
    print(commit if share is None else f"{commit}  ({share:.1%} of total)")
    print(f"  Total:       {payload['total_seconds']:.2f}s")
    for label, phases in (
        ("Top-level", payload["top_level_phases"]),
        ("Write phases", payload["write_phases"]),
    ):
        if phases:
            print(f"  {label}:")
            for name, secs in phases.items():
                print(f"    {name:<24} {secs:>8.2f}s")


COMMAND = Command(
    name="checkpoint-profile",
    help="Per-checkpoint phase timings and Volume-commit cost for a run.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
