"""The `record-migrate` subcommand: bring the record's schema up to the code's."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.adapters.postgres import schema
from src.interfaces.cloud import config as cloud_config
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver record-migrate`."""
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report the two revisions without applying anything.",
    )


class MigratedPayload(BaseModel):
    """The revision the server held, and the one it holds now."""

    op: Literal["record-migrate"] = "record-migrate"
    before: str
    after: str
    applied: bool


def run(args: argparse.Namespace) -> MigratedPayload:
    """Apply, or only compare under `--check`."""
    dsn = cloud_config.record_dsn()
    if args.check:
        from src.adapters.postgres import connect  # noqa: PLC0415 -- one engine, here

        have = schema.current_revision(connect.engine_for(dsn))
        return MigratedPayload(before=have, after=schema.head_revision(), applied=False)
    before, after = schema.upgrade(dsn)
    return MigratedPayload(before=before, after=after, applied=True)


def render(payload: MigratedPayload) -> None:
    have = payload.before or "unversioned"
    if payload.applied:
        print(f"record schema: {have} -> {payload.after}")
    elif payload.before == payload.after:
        print(f"record schema is current at {payload.after}")
    else:
        print(f"record schema is {have}; the code expects {payload.after} -- run without --check")


COMMAND = Command(
    name="record-migrate",
    help="Apply pending schema migrations to the record database (--check to only compare).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
