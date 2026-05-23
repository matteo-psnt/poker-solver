"""CFR over a uniform mixture of boards — a game with a chance layer.

:class:`~src.engine.solver.vector.kernel.VectorCFR` solves one board. That game
has no chance node anywhere in it: both players effectively see the river from
the first decision. It is the right object for validating the kernel's
arithmetic and the wrong one for asking how many iterations real poker needs,
because the whole difficulty of the real game is the chance layer.

This module supplies the missing layer in its smallest honest form: chance deals
one of ``K`` fixed boards uniformly, then play proceeds. The information set is
still ``(node, bucket)`` — a player sees its bucket, not which board produced it
— so exact best response is still computable, and ``K`` is a dial on how much
chance the game contains.

What that dial measures: how iterations-to-convergence scales with the number of
boards. It is the quantity that decides whether a full-tree vector iteration on
one sampled board is a step toward a converged blueprint or merely a fast step
sideways, and it cannot be read off a single-board run at any depth.

Two things have to be joint across boards rather than per board:

    regrets     one stored row is reached through every board, so the boards'
                increments sum before the CFR+ floor applies. Flooring each
                board's contribution separately is a different algorithm.
    best        a responder picks one action per ``(node, bucket)`` for the whole
                mixture, so the maximisation runs on values summed across
                boards. Choosing per board would let it act on the board
                identity, which its information set does not contain.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.engine.solver.vector.compiled_tree import CompiledTree
from src.engine.solver.vector.hand_context import HandContext
from src.engine.solver.vector.kernel import DTYPE, VectorCFR


class BoardMixtureCFR:
    """Vector CFR+ over ``K`` boards sharing one regret table.

    Each board gets its own :class:`VectorCFR` for its per-board scratch — the
    ranges, values and board-specific card matrices — but ``regrets`` and
    ``strategy_sum`` are one shared pair of arrays, because one abstraction row
    is the same row whatever board produced it.
    """

    def __init__(
        self,
        compiled: CompiledTree,
        contexts: Sequence[HandContext],
        *,
        cfr_plus: bool = True,
    ):
        if not contexts:
            raise ValueError("A mixture needs at least one board.")

        self.compiled = compiled
        self.cfr_plus = cfr_plus
        self.iteration = 0

        self.boards = [VectorCFR(compiled, context, cfr_plus=False) for context in contexts]
        self.regrets = self.boards[0].regrets
        self.strategy_sum = self.boards[0].strategy_sum
        for board in self.boards[1:]:
            board.regrets = self.regrets
            board.strategy_sum = self.strategy_sum

        self._delta = np.zeros(compiled.tree.num_slots, dtype=DTYPE)

    @property
    def num_boards(self) -> int:
        return len(self.boards)

    def iterate(self, initial_range: np.ndarray) -> None:
        """One iteration of the mixture game.

        Every board reads the *same* strategy — the one regret matching gives at
        the start of the iteration — and writes its increment to a buffer. If
        boards updated the table in sequence instead, later boards would respond
        to earlier boards' updates and this would be a different algorithm whose
        iteration count is not the one being measured.
        """
        self.iteration += 1
        self._delta[:] = 0.0

        for board in self.boards:
            board.regret_target = self._delta
            board.iterate(initial_range)
            board.regret_target = None

        self.regrets += self._delta
        if self.cfr_plus:
            np.maximum(self.regrets, 0.0, out=self.regrets)

    def best_response_value(self, br_player: int, initial_range: np.ndarray) -> float:
        """Root value of ``br_player``'s best response across the whole mixture.

        The backward pass is interleaved rather than run per board: at each of
        the responder's node groups, every board's child values are collapsed to
        buckets and summed *before* the argmax, so one action is chosen per
        ``(node, bucket)`` for the mixture as a whole. Running each board to
        completion separately and averaging would give a responder that sees the
        board, which overstates exploitability.
        """
        for board in self.boards:
            board.forward(initial_range, use_average=True)
            board.evaluate_terminals()
            board.value[:] = 0.0

        for group in reversed(self.boards[0].groups):
            for chunk in reversed(self.boards[0].chunks(group)):
                nodes = group.node_ids[chunk]
                children = []
                for board in self.boards:
                    targets, is_terminal = board.child_targets(group, chunk)
                    children.append(board.gather_children(targets, is_terminal, br_player))

                if group.actor == br_player:
                    chosen = self._joint_argmax(group, children)
                    for board, child, pick in zip(self.boards, children, chosen, strict=True):
                        board.value[br_player, nodes] = np.take_along_axis(
                            child, pick[:, :, None], axis=2
                        )[:, :, 0]
                else:
                    for board, child in zip(self.boards, children, strict=True):
                        board.value[br_player, nodes] = child.sum(axis=-1)

        total = sum(float(board.value[br_player, 0].sum()) for board in self.boards)
        return total / self.num_boards

    def _joint_argmax(self, group, children: list[np.ndarray]) -> list[np.ndarray]:
        """Per-board, per-hand action indices from one choice per bucket.

        Each board contributes its counterfactual values collapsed onto the
        shared bucket rows; the sum over boards is what the responder can
        actually see, and the argmax over that sum is expanded back through each
        board's own hand→bucket map.
        """
        num_buckets = self.compiled.tree.num_buckets(group.street)
        totals = np.zeros((children[0].shape[0], num_buckets, group.num_actions), dtype=DTYPE)

        for board, child in zip(self.boards, children, strict=True):
            segments = board.segments[group.street]
            collapsed = np.add.reduceat(
                child[:, segments.hand_order, :], segments.segment_start, axis=1
            )
            totals[:, segments.segment_bucket, :] += collapsed

        best = totals.argmax(axis=-1)
        return [best[:, board.context.buckets_for(group.street)] for board in self.boards]

    def exploitability(self, initial_range: np.ndarray, compatible_pairs: float) -> float:
        """Mean of both players' best-response gains, in chips per hand."""
        gains = [
            self.best_response_value(player, initial_range) / compatible_pairs for player in (0, 1)
        ]
        return (gains[0] + gains[1]) / 2.0


__all__: Sequence[str] = ("BoardMixtureCFR",)
