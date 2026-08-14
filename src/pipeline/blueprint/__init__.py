"""A blueprint: how one is built, and how one is read.

    construction.py  config (+ checkpoint) -> the objects a blueprint is made of
    grid.py          the strategy at a node, for every hand a player can hold
    paths.py         naming a spot in the game, and replaying it into a state

**This sits beside `training/` and `evaluation/`, not inside either.** Its
consumers are training, evaluation and anything that serves a run for reading,
and reading outnumbers training -- so filing it under one consumer makes the
others reach across the `training`/`evaluation` independence contract to get at
it. `.importlinter` pins it: nothing here may import training, evaluation,
services or interfaces.

Building and reading are one subject. `grid.py` needs the bucketing the
constructor assembled and `paths.py` needs the betting tree; neither is a
CONSUMER of a blueprint in the sense the contract means, they are how you look
at one.

Not to be confused with `src.interfaces.blueprint`, the HTTP server that exposes
these over a socket and holds no logic of its own.
"""

from src.pipeline.blueprint.construction import (
    build_card_abstraction,
    build_static_evaluation_solver,
    resolve_card_abstraction_hash,
)

__all__ = [
    "build_card_abstraction",
    "build_static_evaluation_solver",
    "resolve_card_abstraction_hash",
]
