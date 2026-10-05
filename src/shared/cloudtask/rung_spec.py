"""``run[:at[:threshold]]``: one rung of a depth ladder, as `deploy.sh` spells it.

Here, stdlib only, because three ends read it -- `score` resolves the runs, the
node fetches them before `uv sync`, `evaluate` loads them -- and must agree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RungSpec:
    """A run, the rung of it that plays (None = its latest), and the cut it serves at."""

    run: str
    at: int | None = None
    threshold: float = 0.0

    @classmethod
    def parse(cls, spec: str) -> RungSpec:
        # Split on ALL THREE fields: `deploy.sh` once read "12000:0.20" as the
        # iteration and staged a rung named "0.20".
        run, _, rest = spec.strip().partition(":")
        at, _, threshold = rest.partition(":")
        if not run:
            raise ValueError(f"rung spec {spec!r} names no run")
        if at and not at.isdigit():
            raise ValueError(f"rung spec {spec!r}: {at!r} is not an iteration")
        return cls(run, int(at) if at else None, float(threshold) if threshold else 0.0)

    def __str__(self) -> str:
        at = "" if self.at is None else str(self.at)
        return f"{self.run}:{at}:{self.threshold:g}"


def parse_ladder(text: str) -> list[RungSpec]:
    """A comma-separated ladder. Refuses an empty one, which would play nothing."""
    specs = [RungSpec.parse(part) for part in text.split(",") if part.strip()]
    if not specs:
        raise ValueError(f"ladder {text!r} names no rung")
    return specs


def format_ladder(specs: list[RungSpec]) -> str:
    return ",".join(str(spec) for spec in specs)
