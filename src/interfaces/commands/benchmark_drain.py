"""The `benchmark-drain` subcommand: give abandoned hand slots back.

Their API allows twenty hands in progress at once, per ACCOUNT, and a hand that
is started and never finished holds its slot indefinitely -- across processes,
across days. Eleven had leaked when this was written, so a run asking for twelve
concurrent hands was fighting for nine.

Draining folds each one, which SCORES it against the public record. That is the
cheap side of the trade: eleven small folds once, against every future run
losing half its concurrency to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command
from src.interfaces.gtowizard import session
from src.interfaces.gtowizard.client import BenchmarkClient, key_from_environment
from src.interfaces.gtowizard.protocol import GAME_NAME

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver benchmark-drain`."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the open hands without folding any.",
    )
    parser.add_argument("--game", default=GAME_NAME, help="Game name.")


class DrainPayload(BaseModel):
    """What was open, and what gave its slot back."""

    op: Literal["benchmark-drain"] = "benchmark-drain"
    game: str
    open_hands: list[int] = []
    released: list[int] = []
    dry_run: bool = False


def run(args: argparse.Namespace) -> DrainPayload:
    with BenchmarkClient(key_from_environment()) as client:
        # One listing, shared: what is reported open and what is acted on have
        # to be the same hands, or the report describes a table it never saw.
        open_frames = client.in_progress(args.game)
        released = [] if args.dry_run else session.drain(client, open_frames)
    return DrainPayload(
        game=args.game,
        open_hands=[frame.hand_id for frame in open_frames],
        released=released,
        dry_run=args.dry_run,
    )


def render(payload: DrainPayload) -> None:
    print(f"{payload.game} — {len(payload.open_hands)} of 20 hand slots were open")
    for hand_id in payload.open_hands:
        mark = "released" if hand_id in payload.released else "still held"
        print(f"  {hand_id}  {'listed' if payload.dry_run else mark}")
    if not payload.open_hands:
        print("  (none — every slot is free)")
    elif payload.dry_run:
        print("Nothing was folded; drop --dry-run to give these slots back.")


COMMAND = Command(
    name="benchmark-drain",
    help="Fold hands left open against GTO Wizard, freeing their concurrency slots.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
