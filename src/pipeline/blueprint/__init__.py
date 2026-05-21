"""A blueprint: how one is built, and how one is read.

    construction.py  config (+ checkpoint) -> the objects a blueprint is made of
    grid.py          the strategy at a node, for every hand a player can hold
    paths.py         naming a spot in the game, and replaying it into a state

**This sits beside `training/` and `evaluation/`, not inside either.** It lived
under `training` while training was the only thing that built a solver, but the
consumers are training, evaluation and anything that serves a run for reading --
and reading outnumbers training. Filing it under one consumer made the others
reach across the `training`/`evaluation` independence contract to get at it,
which is the coupling that contract exists to prevent. `.importlinter` pins it:
nothing here may import training, evaluation, services or interfaces.

Reading was a separate package (`pipeline/analysis/`) until it was clear that
building and reading are the same subject. `grid.py` needs the bucketing the
constructor assembled -- it reads one strategy per BUCKET, not per combo,
because that is what the solver actually distinguishes -- and `paths.py` needs
the betting tree. Neither was ever a consumer of a blueprint in the sense the
contract means; they are how you look at one.

Not to be confused with `src.interfaces.blueprint`, which is the HTTP server
that exposes these over a socket and, by its own docstring, holds no logic of
its own.
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
