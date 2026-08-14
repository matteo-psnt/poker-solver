"""The `push-code` subcommand: publish an immutable snapshot of the tree.

Rarely needed directly. ``submit`` and ``score`` each
snapshot the tree themselves, because pinning per submission is not optional --
a push while a job is running must not change what that job is executing. This
command exists for staging a snapshot deliberately, and for confirming what a
snapshot would contain.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.cloud.tasks import spec
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver push-code`."""
    parser.add_argument("--root", default=".", help="Tree to snapshot.")


class PushedCodePayload(BaseModel):
    """The snapshot that was sealed. Its id is what a task executes."""

    op: Literal["push-code"] = "push-code"
    code_snapshot: str


def run(args: argparse.Namespace) -> PushedCodePayload:
    """Build and upload one snapshot; return its id."""
    config = CloudConfig.load()
    snapshot = share.publish_code_snapshot(
        share.share_client(config), config.share_name, Path(args.root), spec.utcnow()
    )
    return PushedCodePayload(code_snapshot=snapshot)


def render(payload: PushedCodePayload) -> None:
    print(payload.code_snapshot)


COMMAND = Command(
    name="push-code",
    help="Publish an immutable snapshot of the working tree; echoes its id.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
