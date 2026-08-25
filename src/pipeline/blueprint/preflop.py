"""The preflop strategy as a 13x13 grid, the way a poker player reads one.

A blueprint is 45 million rows and unreadable. Its preflop strategy is 169 hand
classes, and a person can tell at a glance whether those look like poker --
whether the raises live in the top-left, whether the folds are bottom-right,
whether suited hands play more than their offsuit twins. That judgement is worth
more than any single number for the question "did this train into something
sane", which is why this exists next to the estimators rather than inside them.

Built on :func:`~src.pipeline.blueprint.paths.replay` and
:func:`~src.pipeline.blueprint.grid.strategy_grid` rather than on hand-built
query states. An earlier renderer assembled its own infoset keys with an assumed
SPR bucket and rendered blank charts whenever anything about key encoding
drifted; going through the maintained read path means the chart follows the
solver instead of guessing at it.

Untrained is carried through, never smoothed. A class the blueprint never
visited is absent, because a uniform row would look like a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.engine.search.range_inference import ALL_COMBOS
from src.pipeline.blueprint.grid import strategy_grid
from src.pipeline.blueprint.paths import replay

if TYPE_CHECKING:
    from src.core.game.state import Card
    from src.engine.solver.policy.source import ScorableBlueprint

#: High-to-low, the order a chart is drawn in.
RANKS = "AKQJT98765432"

_RANK_ORDER = {rank: index for index, rank in enumerate(RANKS)}


#: eval7 ranks the deuce 0 and the ace 12, so this maps a rank to its letter.
_ACE = 12


@dataclass(frozen=True)
class ClassStrategy:
    """One hand class's strategy, and how much training stands behind it.

    ``reach_count`` is the number of times training visited the infoset. A row
    with a handful of visits is a guess wearing a probability, and the chart says
    so rather than letting the colour imply confidence.
    """

    label: str
    actions: tuple[str, ...]
    strategy: tuple[float, ...]
    reach_count: int

    def weight_of(self, *tokens: str) -> float:
        """Total probability on actions whose token starts with any of ``tokens``.

        The tokens are :func:`~src.pipeline.blueprint.paths.encode_action`'s --
        ``f`` fold, ``x`` check, ``c`` call, ``b``/``r`` sized, ``A`` all-in --
        because that is what the grid reports and inventing a second vocabulary
        here is how a renderer starts disagreeing with the tree.
        """
        return sum(
            probability
            for action, probability in zip(self.actions, self.strategy, strict=True)
            if action.startswith(tokens)
        )

    @property
    def aggression(self) -> float:
        """Probability of putting chips in beyond a call."""
        return self.weight_of("b", "r", "A")

    @property
    def fold(self) -> float:
        return self.weight_of("f")

    @property
    def passive(self) -> float:
        """Check or call -- staying in without raising."""
        return self.weight_of("x", "c")


@dataclass(frozen=True)
class PreflopChart:
    """Every trained preflop class at one node, plus what was not trained."""

    path: str
    actor: int
    actions: tuple[str, ...]
    classes: dict[str, ClassStrategy]
    untrained: tuple[str, ...]

    def row(self, high: str, low: str) -> ClassStrategy | None:
        """The cell at grid position ``(high, low)``.

        Above the diagonal is suited, below is offsuit -- the convention every
        poker chart uses, so a reader does not have to be told.
        """
        return self.classes.get(class_label(high, low))


def class_label(high: str, low: str) -> str:
    """The class at grid position ``(high, low)``: pair, suited, or offsuit."""
    if high == low:
        return f"{high}{high}"
    if _RANK_ORDER[high] < _RANK_ORDER[low]:
        return f"{high}{low}s"
    return f"{low}{high}o"


def preflop_chart(blueprint: ScorableBlueprint, path: str = "") -> PreflopChart:
    """Read the blueprint's preflop strategy for every hand class at ``path``.

    ``path`` is empty for the opening spot -- the button first to act. Any
    preflop line the action model offers works: ``"c"`` is the big blind's
    option behind a limp, and a raise token puts the caller in the spot facing
    it. The path is the identifier, so a chart names the spot it describes.
    """
    node = replay(blueprint, path)
    grid = strategy_grid(blueprint, node)

    classes: dict[str, ClassStrategy] = {}
    untrained: set[str] = set()
    for index, combo in enumerate(ALL_COMBOS):
        row = grid.for_combo(index)
        if row is None:
            continue
        label = _class_of(combo)
        if not row.trained or row.strategy is None:
            if label not in classes:
                untrained.add(label)
            continue
        # Every combo of a class shares a preflop bucket, so the first trained
        # one settles the row; the rest are the same by construction.
        untrained.discard(label)
        classes.setdefault(
            label,
            ClassStrategy(
                label=label,
                actions=grid.actions,
                strategy=row.strategy,
                reach_count=row.reach_count,
            ),
        )

    return PreflopChart(
        path=path,
        actor=node.actor if node.actor is not None else -1,
        actions=grid.actions,
        classes=classes,
        untrained=tuple(sorted(untrained)),
    )


def _class_of(combo: tuple[Card, Card]) -> str:
    """The canonical class string for one concrete combo."""
    first, second = combo
    high, low = sorted((_rank_char(first), _rank_char(second)), key=_RANK_ORDER.__getitem__)
    if high == low:
        return f"{high}{high}"
    suited = first.suit_eval7() == second.suit_eval7()
    return f"{high}{low}{'s' if suited else 'o'}"


def _rank_char(card: Card) -> str:
    """The chart's rank letter for a card."""
    return RANKS[_ACE - card.rank_eval7()]
