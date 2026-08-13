"""The `configs` subcommand: what `--config` will accept.

Every dispatching command names a config by STEM -- `submit --config production`,
`submit-precompute --config ochs_gate_ochs` -- and nothing said what the legal
stems were. On a command line that is a mild annoyance (`ls config/training`
answers it). For a second surface it is a wall: the console cannot offer a
picker without either a command to ask, or its own directory read -- and its own
directory read is the thing the whole command seam exists to prevent.

So the answer becomes a command, and both surfaces get it. A stem that this
lists is a stem `submit` accepts: they resolve against the same directories, so
a config added to the tree appears here without anything being registered.

**Names only, deliberately.** Parsing each YAML would make listing the options
cost loading them, and the console calls this to fill a dropdown. `runinfo`
already answers what a run was actually trained with, which is the question
whose answer has to be exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.interfaces.commands._base import Command
from src.shared import repo

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

CONFIG_ROOT = repo.ROOT / "config"

"""The two kinds, and the flag each one feeds.

Kept as data rather than two code paths because the console renders them the
same way and only the destination differs.
"""
KINDS: dict[str, str] = {
    "training": "submit --config",
    "abstraction": "submit-precompute --config",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver configs`."""
    parser.add_argument(
        "--kind",
        default="",
        choices=("", *KINDS),
        help="Show only training or only abstraction configs (default: both).",
    )


def _stems(directory: Path) -> list[str]:
    """Config stems under ``directory``, sorted.

    Missing rather than empty is not distinguished: a checkout without
    `config/abstraction` and one with an empty `config/abstraction` are the same
    answer to "what can I pass", and the caller has no different action to take.
    """
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """List the config stems each dispatching command will accept."""
    wanted = [args.kind] if args.kind else list(KINDS)
    return {
        "op": "configs",
        "root": str(CONFIG_ROOT),
        "kinds": [
            {"kind": kind, "flag": KINDS[kind], "names": _stems(CONFIG_ROOT / kind)}
            for kind in wanted
        ],
    }


def render(payload: dict[str, Any]) -> None:
    for group in payload["kinds"]:
        names = group["names"]
        print(f"{group['kind']} ({group['flag']}) — {len(names)} config(s)")
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (none)")


COMMAND = Command(
    name="configs",
    help="List the config stems `submit` and `submit-precompute` accept.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
