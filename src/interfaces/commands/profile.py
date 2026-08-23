"""The `profile` subcommand: ask a RUNNING task for a stack sample, and read it.

The other half of :mod:`src.shared.cloudtask.node.profile`, which explains why
the trigger is a file on the share rather than a submit flag. This is the
operator's end of that contract: it writes the request, waits for the node to
serve it, and brings the speedscope document down.

The wait is the reason this is a command and not a documented `touch`. The
node polls, records for the duration asked, then uploads -- so the profile
appears a minute or two after the request, under a name the caller cannot
construct: the attempt number and the per-task counter are the node's. Finding
it means listing and taking what is NEW, which is a loop nobody should be asked
to run by hand at the moment they are already debugging something else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from azure.core.exceptions import ResourceNotFoundError
from pydantic import BaseModel

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import blob
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node import profile as node_profile

if TYPE_CHECKING:
    import argparse


# A poll on the node, plus the recording, plus the upload of a few MB over SMB.
# Generous because the cost of being early is reporting "nothing landed" about a
# profile that lands ten seconds later.
SETTLE_SECONDS = 240
POLL_SECONDS = 10


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver profile`."""
    parser.add_argument("--task", default=None, help="Task id to sample. It must be RUNNING.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=node_profile.DEFAULT_SECONDS,
        help=f"Sampling duration, clamped on the node to {node_profile.MAX_SECONDS}s.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the profiles already on the share."
    )
    parser.add_argument(
        "--get", default=None, help="Download this profile by name instead of asking for a new one."
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Directory to download into. The name on the share is kept.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Write the request and return. The profile is then read with --list/--get.",
    )


class ProfilePayload(BaseModel):
    """What was asked for, and what came back.

    `landed` is null for `--no-wait` and for a wait that timed out; the two are
    told apart by `waited`, because a request that was served slowly and one
    that was never served are different problems.
    """

    op: Literal["profile"] = "profile"
    task: str | None = None
    seconds: int | None = None
    waited: int | None = None
    landed: str | None = None
    downloaded: str | None = None
    available: list[str] | None = None


def _profiles(config: Any) -> list[str]:
    """Every profile document in the container, newest name last."""
    return blob.diagnostic_names(config, node_profile.PROFILE_SUFFIX)


def _download(config: Any, name: str, out: str) -> str:
    destination = Path(out).expanduser() / name
    blob.download_diagnostic(config, name, destination)
    return str(destination)


def run(args: argparse.Namespace) -> ProfilePayload:
    """Ask for a profile, or read one that already exists."""
    if not (args.list or args.get or args.task):
        raise CommandError("profile: --task is required unless --list or --get is given.")

    config = CloudConfig.load()

    if args.list:
        return ProfilePayload(available=_profiles(config))

    if args.get:
        return ProfilePayload(
            landed=args.get,
            downloaded=_download(config, args.get, args.out),
        )

    # Taken BEFORE the request so the wait can tell a profile this call produced
    # from one an earlier call left behind -- the node numbers them per task and
    # per attempt, so a retried task starts counting again.
    before = set(_profiles(config))
    blob.write_diagnostic(
        config,
        f"{args.task}{node_profile.REQUEST_SUFFIX}",
        # A blob is committed whole, so the trailing newline the SMB write
        # needed as a completeness marker is now belt and braces rather than
        # load-bearing: the node cannot see this object before its bytes.
        f"{args.seconds}\n",
    )
    if args.no_wait:
        return ProfilePayload(task=args.task, seconds=args.seconds)

    deadline = time.monotonic() + args.seconds + SETTLE_SECONDS
    started = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        fresh = [
            name
            for name in _profiles(config)
            if name not in before and name.startswith(f"{args.task}.")
        ]
        if fresh:
            name = sorted(fresh)[-1]
            try:
                downloaded = _download(config, name, args.out)
            except ResourceNotFoundError:
                # LISTED is not READABLE -- kept because a listing and a read
                # are still two calls, and the one that raised here came back
                # as raw Azure XML. The next poll gets it.
                continue
            return ProfilePayload(
                task=args.task,
                seconds=args.seconds,
                waited=int(time.monotonic() - started),
                landed=name,
                downloaded=downloaded,
            )
    return ProfilePayload(
        task=args.task, seconds=args.seconds, waited=int(time.monotonic() - started)
    )


def render(payload: ProfilePayload) -> None:
    if payload.available is not None:
        for name in payload.available:
            print(f"  {name}")
        if not payload.available:
            print("  no profiles on the share yet")
        return
    if payload.downloaded:
        print(f"  {payload.landed} -> {payload.downloaded}")
        print("  open it at https://speedscope.app")
        return
    if payload.waited is None:
        print(f"  asked {payload.task} for {payload.seconds}s")
        print("  read it with: poker-solver profile --list")
        return
    print(f"  nothing landed for {payload.task} in {payload.waited}s.")
    print("  The task must be RUNNING, and its log says whether py-spy could attach:")
    print(f"    poker-solver logs --task {payload.task} | grep profile")


COMMAND = Command(
    name="profile",
    help="Sample a running task's stacks with py-spy and download the result.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
