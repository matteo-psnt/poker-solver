"""The `seat-plan` subcommand: what the chipzen seat fields, resolved against the store.

`infra/serve/seat.toml` names the runs and rungs. "latest" there means the newest
rung the checkpoints container HOLDS -- not what a manifest names, which includes
rungs still uploading and rungs pruned long ago. The deploy runs this before it
touches the box, so a typo'd run or a missing rung fails there and not on the box.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, computed_field

from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import blob
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import repo

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

SEAT_FILE = repo.ROOT / "infra" / "serve" / "seat.toml"
RUNG = re.compile(r"static-(\d+)\.ckpt\.zst")
LATEST = "latest"


class Seated(BaseModel):
    """One blueprint as the seat takes it: `run:at:threshold`."""

    run: str
    at: int
    threshold: float

    @computed_field
    @property
    def spec(self) -> str:
        return f"{self.run}:{self.at}:{self.threshold:.2f}"


class SeatPlanPayload(BaseModel):
    """The deepest blueprint, the shallower rungs, and a digest that changes
    exactly when what is fielded does -- the deploy's marker for "redeploy"."""

    op: Literal["seat-plan"] = "seat-plan"
    base: Seated
    rungs: list[Seated]
    digest: str


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver seat-plan`."""
    parser.add_argument("--file", default=str(SEAT_FILE), help="The seat file to resolve.")


def _resolve(run: str, at: str, threshold: float, held: Callable[[str], set[str]]) -> Seated:
    rungs = sorted(int(m[1]) for name in held(run) if (m := RUNG.fullmatch(name)))
    if not rungs:
        raise CommandError(f"{run} holds no rung in the checkpoints container")
    if at == LATEST:
        return Seated(run=run, at=rungs[-1], threshold=threshold)
    if not at.isdigit() or int(at) not in rungs:
        raise CommandError(f"{run} has no rung {at!r}; the container holds {rungs[0]}..{rungs[-1]}")
    return Seated(run=run, at=int(at), threshold=threshold)


def plan(spec: dict, held: Callable[[str], set[str]]) -> SeatPlanPayload:
    """Resolve a parsed seat file against `held(run)`, the rung objects a run holds."""
    base = _resolve(spec["run"], str(spec["at"]), float(spec["threshold"]), held)
    rungs = []
    for entry in spec.get("rungs", []):
        run, at, threshold = entry.split(":")
        rungs.append(_resolve(run, at, float(threshold), held))
    fielded = " ".join(seat.spec for seat in (base, *rungs))
    digest = hashlib.sha256(fielded.encode()).hexdigest()[:12]
    return SeatPlanPayload(base=base, rungs=rungs, digest=digest)


def run(args: argparse.Namespace) -> SeatPlanPayload:
    """Resolve the seat file against the checkpoints container."""
    config = CloudConfig.load()
    spec = tomllib.loads(Path(args.file).read_text())
    return plan(spec, lambda run_id: blob.rungs_for(config, run_id))


def render(payload: SeatPlanPayload) -> None:
    print(f"base  {payload.base.spec}")
    for seat in payload.rungs:
        print(f"rung  {seat.spec}")
    print(f"digest {payload.digest}")


COMMAND = Command(
    name="seat-plan",
    help="Resolve infra/serve/seat.toml: the rungs the seat would field.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
