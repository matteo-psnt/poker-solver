"""The `benchmark-board` subcommand: the public GTO Wizard leaderboard.

Needs NO API key -- `/leaderboard` and `/winnings` are open -- which is what
makes the field, and the two reference floors we probe against, readable before
a key is approved.

Ranked on ``aivat_score_bb_per_100``, which is the benchmark's metric.
``bb_per_100`` is the RAW chip result and is dominated by luck: Roman_SL reads
+0.51 raw and -9.76 adjusted. Both are shown so the gap is visible rather than
inviting the wrong one to be quoted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command
from src.interfaces.gtowizard.client import DEFAULT_VERSION, BenchmarkClient
from src.interfaces.gtowizard.protocol import GAME_NAME

if TYPE_CHECKING:
    import argparse

# Their endpoint's own ceiling.
MAX_ROWS = 100


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver benchmark-board`."""
    parser.add_argument("--limit", type=int, default=25, help="Rows to show.")
    parser.add_argument(
        "--min-hands",
        type=int,
        default=1000,
        help="Hide entries below this volume (default 1000, their own default).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help=f"GTO Wizard AI engine version (default {DEFAULT_VERSION}). Versions are "
        "DIFFERENT OPPONENTS: Claude Opus 4.6 reads -13.62 on v1 and -19.74 on v2.",
    )
    parser.add_argument("--game", default=GAME_NAME, help="Game name.")


class BoardRow(BaseModel):
    bot_name: str
    organization: str
    version: int | None = None
    hands: int
    aivat_bb_per_100: float
    aivat_std_bb_per_100: float
    raw_bb_per_100: float


class BoardPayload(BaseModel):
    """The public leaderboard, ranked by the metric that counts."""

    op: Literal["benchmark-board"] = "benchmark-board"
    game: str
    version: int | None = None
    rows: list[BoardRow] = []


def run(args: argparse.Namespace) -> BoardPayload:
    """Read the public leaderboard.

    Always asks for their full page and truncates HERE: their ``limit`` cuts by
    their own ordering, which is not the AIVAT ranking, so asking for eight rows
    returns eight arbitrary ones and drops entries that belong in the top eight.
    """
    with BenchmarkClient() as client:
        entries = client.leaderboard(
            args.game, limit=MAX_ROWS, min_hands=args.min_hands, version=args.version
        )
    rows = [
        BoardRow(
            bot_name=str(entry["bot_name"]),
            organization=str(entry.get("organization", "")),
            version=entry.get("gto_wizard_version"),
            hands=int(entry["total_hands"]),
            aivat_bb_per_100=float(entry["aivat_score_bb_per_100"]),
            aivat_std_bb_per_100=float(entry["aivat_std_bb_per_100"]),
            raw_bb_per_100=float(entry["bb_per_100"]),
        )
        for entry in entries
    ]
    rows.sort(key=lambda row: -row.aivat_bb_per_100)
    return BoardPayload(game=args.game, version=args.version, rows=rows[: args.limit])


def render(payload: BoardPayload) -> None:
    print(f"{payload.game} — GTO Wizard AI v{payload.version}")
    print(f"{'AIVAT bb/100':>13} {'±':>6} {'raw':>8} {'hands':>8}  bot")
    for row in payload.rows:
        who = f"{row.bot_name} ({row.organization})" if row.organization else row.bot_name
        print(
            f"{row.aivat_bb_per_100:13.2f} {row.aivat_std_bb_per_100:6.2f} "
            f"{row.raw_bb_per_100:8.2f} {row.hands:8d}  {who}"
        )
    if not payload.rows:
        print("  (none)")


COMMAND = Command(
    name="benchmark-board",
    help="The public GTO Wizard leaderboard, ranked by AIVAT bb/100 (no API key needed).",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
