"""CFR over a uniform mixture of boards -- a game with a chance layer.

:class:`~src.engine.solver.vector.kernel.VectorCFR` solves one board, a game
with no chance node anywhere in it: both players effectively see the river from
the first decision. That is the right object for validating the kernel's
arithmetic and the wrong one for asking how many iterations real poker needs.

Here chance deals one of ``K`` fixed boards uniformly, then play proceeds. The
information set is still ``(node, bucket)`` -- a player sees its bucket, not
which board produced it -- so exact best response is still computable, and ``K``
is a dial on how much chance the game contains. What it measures is how
iterations-to-convergence scales with the number of boards, which is what
decides whether a full-tree vector iteration on one sampled board is a step
toward a converged blueprint or a fast step sideways.

Two things must be JOINT across boards rather than per board:

    regrets     one stored row is reached through every board, so the boards'
                increments sum before the CFR+ floor applies. Flooring each
                board's contribution separately is a different algorithm.
    best        a responder picks one action per ``(node, bucket)`` for the whole
                mixture, so the maximisation runs on values summed across
                boards. Choosing per board would let it act on the board
                identity, which its information set does not contain.

That best response measures exploitability *inside* the abstraction: the
opponent is only as sharp as our own bucketing, so the number falls as the
solver improves and says nothing about what the bucketing costs.
``best_response_value(unconstrained=True)`` measures the other one -- the
opponent is told its exact two cards -- and the gap between them is what the
abstraction gives away. That gap does not shrink with more iterations.

Being told your cards is not being told the future, and the difference is easy
to get wrong in the flattering direction. A street's bucket is a function of the
hand and the cards face up on that street
(``vector_universe.POSTFLOP_PREFIX``), so two runouts sharing that prefix are
ONE observation and must be maximised jointly. Maximising per board prices in
clairvoyance and reads as a larger abstraction cost than is real.
``_visible_partition`` is what keeps the two apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector.kernel import DTYPE, VectorCFR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.engine.solver.vector.compiled_tree import CompiledTree
    from src.engine.solver.vector.hand_context import HandContext

# How many board cards are face up when a street's decisions are taken.
VISIBLE_CARDS: dict[Street, int] = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}

# Width of the global two-card index, ``low * 52 + high``. Sparse — only 1,326
# of the slots are reachable — but it lets boards with different live-hand sets
# be summed on one axis without building a per-partition mapping.
GLOBAL_HANDS = 52 * 52


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
        boards: Sequence[Sequence[int] | np.ndarray] | None = None,
    ):
        if not contexts:
            raise ValueError("A mixture needs at least one board.")
        if boards is not None and len(boards) != len(contexts):
            raise ValueError("One board per context, in the same order.")

        self.compiled = compiled
        self.cfr_plus = cfr_plus
        self.iteration = 0
        self.boards_cards = None if boards is None else [tuple(int(c) for c in b) for b in boards]

        self.boards = [VectorCFR(compiled, context, cfr_plus=False) for context in contexts]
        self.regrets = self.boards[0].regrets
        self.strategy_sum = self.boards[0].strategy_sum
        for board in self.boards[1:]:
            board.regrets = self.regrets
            board.strategy_sum = self.strategy_sum

        self._delta = np.zeros(compiled.tree.num_slots, dtype=DTYPE)
        self._global_hand_id = [
            context.hand_cards[:, 0].astype(np.int64) * 52 + context.hand_cards[:, 1]
            for context in contexts
        ]

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

    def best_response_value(
        self,
        br_player: int,
        initial_range: np.ndarray,
        *,
        unconstrained: bool = False,
        per_board: bool = False,
    ) -> float | np.ndarray:
        """Root value of ``br_player``'s best response across the whole mixture.

        The backward pass is interleaved rather than run per board: at each of
        the responder's node groups, every board's child values are collapsed to
        buckets and summed *before* the argmax, so one action is chosen per
        ``(node, bucket)`` for the mixture as a whole.

        ``unconstrained`` lifts the *card* abstraction and nothing else. The
        constrained responder sees what the abstraction shows: one action per
        bucket, chosen jointly across boards, because a bucket cannot tell you
        which board you are on. The unconstrained one is told its exact holding
        — but it is still not told the future, so the maximisation stays joint
        across boards it cannot yet distinguish. See :meth:`_per_hand_max`.

        So the constrained figure is what the solver is trying to minimise and
        the unconstrained one is what an opponent could actually take. Their
        difference is the abstraction's own cost, and it is the number that does
        not go away no matter how long the solver runs.

        ``per_board`` returns each board's own contribution instead of their
        mean. Read them as PAIRED samples, not independent ones: the responder
        picks one action per ``(node, bucket)`` across the whole mixture, so a
        board's value is its share of a jointly chosen strategy rather than a
        best response to that board alone. That is exactly what makes them the
        right thing to difference between two strategies scored on the SAME
        boards — and the wrong thing to treat as i.i.d. draws when quoting a
        single arm's absolute error bar.

        Raises:
            ValueError: if ``unconstrained`` and the mixture was built without
                ``boards``, which is what says who may be distinguished when.
        """
        if unconstrained and self.boards_cards is None:
            raise ValueError(
                "An unconstrained best response needs the board cards: without them "
                "the mixture cannot tell which runouts a responder is allowed to "
                "distinguish, and maximising per board would let it act on cards "
                "that have not been dealt. Pass boards= to the constructor."
            )
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

                if group.actor == br_player and unconstrained:
                    self._per_hand_max(group, children, br_player, nodes)
                elif group.actor == br_player:
                    chosen = self._joint_argmax(group, children)
                    for board, child, pick in zip(self.boards, children, chosen, strict=True):
                        board.value[br_player, nodes] = np.take_along_axis(
                            child, pick[:, :, None], axis=2
                        )[:, :, 0]
                else:
                    for board, child in zip(self.boards, children, strict=True):
                        board.value[br_player, nodes] = child.sum(axis=-1)

        # float64 regardless of the kernel's dtype: this is a handful of numbers,
        # and rounding them to float32 makes the decomposition disagree with the
        # aggregate in the 8th figure -- enough to look like a real discrepancy
        # to anyone checking that the parts sum to the whole.
        contributions = np.array(
            [float(board.value[br_player, 0].sum()) for board in self.boards], dtype=np.float64
        )
        if per_board:
            return contributions
        return float(contributions.sum()) / self.num_boards

    def _visible_partition(self, street: Street) -> list[list[int]]:
        """Board indices grouped by what is face up when ``street`` is acted on.

        Two runouts that agree on every visible card are the *same* observation
        to a player standing on that street, however different their futures
        are. Preflop nothing is face up, so every board falls in one group; by
        the river the prefix is the whole board and each is alone.

        This is the whole difference between measuring an abstraction's cost and
        measuring clairvoyance. Maximising per board would let a responder pick
        its preflop action already knowing the river — an opponent no strategy
        can defend against, and not one that exists.
        """
        assert self.boards_cards is not None
        visible = VISIBLE_CARDS[street]
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, board in enumerate(self.boards_cards):
            groups.setdefault(tuple(sorted(board[:visible])), []).append(index)
        return list(groups.values())

    def _per_hand_max(self, group, children: list[np.ndarray], br_player: int, nodes) -> None:
        """Maximise per hand, jointly over boards sharing this street's prefix.

        Hands are summed on a global two-card axis rather than each board's own,
        because boards remove different cards and so index their live hands
        differently. A hand blocked by one board simply contributes nothing to
        that board's term, which is right: its value is the sum over exactly the
        runouts it could still be held on.
        """
        for members in self._visible_partition(group.street):
            if len(members) == 1:
                only = members[0]
                self.boards[only].value[br_player, nodes] = children[only].max(axis=-1)
                continue

            totals = np.zeros((children[0].shape[0], GLOBAL_HANDS, group.num_actions), dtype=DTYPE)
            for index in members:
                # A board lists each live hand once, so these indices are unique
                # and ``+=`` needs no ``np.add.at``.
                totals[:, self._global_hand_id[index], :] += children[index]

            best = totals.argmax(axis=-1)
            for index in members:
                pick = best[:, self._global_hand_id[index]]
                self.boards[index].value[br_player, nodes] = np.take_along_axis(
                    children[index], pick[:, :, None], axis=2
                )[:, :, 0]

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

    def exploitability(
        self, initial_range: np.ndarray, compatible_pairs: float, *, unconstrained: bool = False
    ) -> float:
        """Mean of both players' best-response gains, in chips per hand."""
        gains = [
            self.best_response_value(player, initial_range, unconstrained=unconstrained)
            / compatible_pairs
            for player in (0, 1)
        ]
        return (gains[0] + gains[1]) / 2.0

    def exploitability_per_board(
        self, initial_range: np.ndarray, compatible_pairs: float, *, unconstrained: bool = False
    ) -> np.ndarray:
        """The same number, decomposed onto the boards that produced it.

        The aggregate is these times ``num_boards``, so nothing here is a second
        estimate of the same quantity -- it is the one estimate, un-collapsed.
        Keeping it is what lets a comparison carry an interval: two arms scored
        on one board set difference board-by-board, and the spread of THAT is
        the noise a verdict has to clear. Collapsing first throws the only
        evidence about its own precision away, which is how a 3.3% effect came
        to be quoted against a 7-14% spread.
        """
        gains = [
            np.asarray(
                self.best_response_value(
                    player, initial_range, unconstrained=unconstrained, per_board=True
                )
            )
            / compatible_pairs
            for player in (0, 1)
        ]
        return (gains[0] + gains[1]) / 2.0


__all__: Sequence[str] = ("BoardMixtureCFR",)
