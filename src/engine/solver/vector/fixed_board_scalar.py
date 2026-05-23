"""The production scalar kernel, restricted to a fixed set of boards.

The vector kernels solve a game whose chance layer is "one of these K runouts,
uniformly". The shipped scalar kernel
(:class:`~src.engine.solver.mccfr.static_solver.StaticTreeSolver`) samples a
fresh runout from the whole deck every iteration, so the two do not solve the
same game and cannot be compared as they stand.

This pins the scalar kernel to the same K runouts. Then both are solving one
identical game, both write the same ``(node, bucket, action)`` table, and the
same exact best response scores either — which is the only way to ask which
kernel gets further per unit of compute rather than which one had the easier
problem.

What is deliberately NOT changed: the regret math, the external sampling, the
averaging, the storage. Only where the cards come from. A comparison whose
"scalar" arm was a reimplementation would measure the reimplementation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.solver.mccfr.static_solver import StaticTreeSolver

# A runout is the community cards in order; the prefix of length
# ``street.board_card_count`` is what is public on each street. Typed as a plain
# tuple rather than a fixed five: callers build these by comprehension from a
# sampled board, and pinning the arity here only moves the complaint to them
# without making anything safer -- the length is a property of the board.
Runout = tuple[Card, ...]


class FixedBoardStaticSolver(StaticTreeSolver):
    """``StaticTreeSolver`` whose chance layer draws from a fixed runout set.

    One runout is chosen per iteration and then *held*: every street of that
    hand reveals the next cards of the same board. Sampling independently per
    street would deal boards that are not in the set at all, which is a
    different — and much larger — chance layer than the one being compared.
    """

    def __init__(self, *args, runouts: Sequence[Runout], **kwargs):
        super().__init__(*args, **kwargs)
        if not runouts:
            raise ValueError("A fixed-board solver needs at least one runout.")
        self._runouts = list(runouts)
        self._current: Runout = self._runouts[0]

    def deal_initial_state(self) -> GameState:
        """Choose this iteration's runout, then hole cards that avoid it."""
        self._current = self._runouts[random.randrange(len(self._runouts))]
        blocked = set(self._current)
        available = [card for card in FULL_DECK if card not in blocked]
        cards = random.sample(available, 4)
        # The TREE's stack, not the config's. The tree is enumerated for one
        # starting stack and every infoset id derives from it, so dealing the
        # config's stack into a tree built for another produces states the tree
        # has no node for -- which surfaces as an illegal action deep in a
        # traversal rather than as a mismatch anyone can see.
        return self.rules.create_initial_state(
            starting_stack=self.tree.starting_stack,
            hole_cards=((cards[0], cards[1]), (cards[2], cards[3])),
            button=(self.iteration // 2) % 2,
        )

    def sample_chance_outcome(self, state: GameState) -> GameState:
        """Reveal the next cards of *this hand's* runout, not fresh ones."""
        return self._with_board(state)

    def deal_remaining_cards(self, state: GameState) -> GameState:
        """Complete the board at an all-in showdown, from the same runout."""
        if len(state.board) >= 5:
            return state
        return state.replace(board=self._current, validate=False)

    def _with_board(self, state: GameState) -> GameState:
        needed = state.street.board_card_count
        if len(state.board) >= needed:
            return state
        return state.replace(board=self._current[:needed], validate=False)


__all__ = ("FixedBoardStaticSolver", "Runout")
