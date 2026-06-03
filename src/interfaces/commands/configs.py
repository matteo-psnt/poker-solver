"""The `configs` subcommand: what `--config` will accept.

Every dispatching command names a config by STEM -- `submit --config production`,
`submit-precompute --config production` -- and nothing said what the legal
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

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command
from src.shared import repo

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

CONFIG_ROOT = repo.ROOT / "config"

# Data rather than two code paths: the console renders them identically and only
# the destination differs.
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


class ConfigKind(BaseModel):
    kind: str
    """The flag these feed, verbatim -- so a picker can say what it sets."""
    flag: str
    names: list[str] = []


class ConfigsPayload(BaseModel):
    """The config stems each dispatching command will accept."""

    op: Literal["configs"] = "configs"
    root: str
    kinds: list[ConfigKind] = []


def run(args: argparse.Namespace) -> ConfigsPayload:
    """List the config stems each dispatching command will accept."""
    wanted = [args.kind] if args.kind else list(KINDS)
    return ConfigsPayload(
        root=str(CONFIG_ROOT),
        kinds=[
            ConfigKind(kind=kind, flag=KINDS[kind], names=_stems(CONFIG_ROOT / kind))
            for kind in wanted
        ],
    )


def render(payload: ConfigsPayload) -> None:
    for group in payload.kinds:
        print(f"{group.kind} ({group.flag}) — {len(group.names)} config(s)")
        for name in group.names:
            print(f"  {name}")
        if not group.names:
            print("  (none)")


COMMAND = Command(
    name="configs",
    help="List the config stems `submit` and `submit-precompute` accept.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
