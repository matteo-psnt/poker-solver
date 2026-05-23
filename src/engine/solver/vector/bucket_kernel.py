"""Vector CFR+ with no board dimension at all.

:class:`~src.engine.solver.vector.kernel.VectorCFR` carries a range over private
*hands* and needs one full pass per board. There are 134,459 suit-isomorphic
river boards and a realistic budget affords a few hundred, so that design cannot
train a blueprint however fast the kernel is — the gap is 400x and structural.

This kernel carries a range over *buckets* instead. Chance is no longer a card
deal but a bucket-to-bucket transition applied on edges that cross a street
boundary, so a single pass covers the entire board space at once. The abstraction
did the integration over boards when it was built; this just stops re-doing it.

What changes from the hand-space kernel, and nothing else does:

    width       vectors are ``max(buckets)`` wide, not ``live hands`` wide, and
                the meaningful prefix changes per street
    chance      a street-crossing edge multiplies the range by ``P(next | this)``
                going down, and by its transpose coming back up
    terminals   fold mass and showdown values come from bucket-pair matrices
                rather than exact per-hand blocking and ranks

The regret layout, the node grouping, the level order and the CFR+ math are the
same, and deliberately so: ``regrets`` and ``strategy_sum`` are the identical
flat ``(node, bucket, action)`` arrays, which is what lets a strategy trained
here be scored by the hand-space best response with no translation. That
cross-evaluation is the whole point — it measures what the averaging in
``bucket_game`` costs, in the game where card removal and showdowns are exact.

There is no hand→bucket collapse here because there are no hands: the segment sum
the hand-space kernel needs is the identity in bucket space.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector.bucket_game import BucketGame
from src.engine.solver.vector.compiled_tree import EDGE_TO_TERMINAL, CompiledTree, TerminalKind
from src.engine.solver.vector.kernel import NodeGroup, build_groups

# float32, and the choice is measured rather than inherited.
#
# float64 is genuinely better for *diagnostics*: zero-sum at the root is an
# identity here, and the root value is a near-total cancellation of intermediates
# four orders larger, so float32 leaves a ~1e-3 relative residual where float64
# leaves ~1e-13. Four significant digits.
#
# It buys nothing for the *strategy*. Six paired runs on the real abstraction
# (three derivation seeds x 32 and 2,000 boards), scoring the trained strategy by
# the hand-space kernel's exact best response, put the float64-minus-float32
# difference at |delta| <= 0.001 with the sign flipping -- against a
# seed-to-seed spread of 0.067 (32 boards) and 0.012 (2,000). That is no effect.
#
# And it is not free: ~1.8x wall-clock and 2x memory at production shapes
# (4.2-4.8 s -> 7.4-8.6 s per iteration; 2.4 GB -> 3.6 GB peak). So the default
# is the cheap one, and float64 stays available for when a residual matters.
DEFAULT_DTYPE = np.float32

# Street order, so a transition can be looked up from a pair of streets.
_NEXT_STREET = {
    Street.PREFLOP: Street.FLOP,
    Street.FLOP: Street.TURN,
    Street.TURN: Street.RIVER,
}


def _as_dtype(game: BucketGame, dtype: type[np.floating]) -> BucketGame:
    """The same game with every matrix in one precision."""
    return BucketGame(
        buckets_per_street=game.buckets_per_street,
        transitions={step: m.astype(dtype, copy=False) for step, m in game.transitions.items()},
        compatible={s: m.astype(dtype, copy=False) for s, m in game.compatible.items()},
        showdown=game.showdown.astype(dtype, copy=False),
    )


class BucketVectorCFR:
    """Full-tree vector CFR+ over the board-free abstract game.

    One iteration covers every infoset at every node for the whole board space,
    not for one board. Ranges are stored at a uniform ``max_buckets`` width with
    only the current street's prefix meaningful; the arrays are small enough
    (169-600 wide against 1,081 hands) that padding costs less than the
    bookkeeping to avoid it.

    Known limit, deliberately not fixed:
        Unlike :class:`~src.engine.solver.vector.kernel.VectorCFR`, this does not
        chunk a node group before working on it — the widest level at production
        settings builds one ``(14,737 nodes, 600 buckets, 6 actions)`` block,
        about 400 MB in float64. That is comfortable at 100/300/600 and measured
        (2.4 GB peak in float32, 3.6 GB in float64), but it scales with the
        bucket count, so a substantially finer abstraction would want the
        ``chunks`` treatment the hand-space kernel already has.
    """

    def __init__(
        self,
        compiled: CompiledTree,
        game: BucketGame,
        *,
        cfr_plus: bool = True,
        dtype: type[np.floating] = DEFAULT_DTYPE,
    ):
        self.compiled = compiled
        self.cfr_plus = cfr_plus
        self.dtype = dtype
        self.iteration = 0

        # The game is derived in float64 -- these are the constants every value
        # is built from, and rounding them is the one economy with no upside.
        # Carrying that into a float32 kernel would upcast every matmul and
        # silently pay the float64 cost anyway, so it is cast once, here.
        self.game = _as_dtype(game, dtype)

        tree = compiled.tree
        self.width = game.max_buckets
        self.regrets = np.zeros(tree.num_slots, dtype=self.dtype)
        self.strategy_sum = np.zeros(tree.num_slots, dtype=self.dtype)
        self.groups = build_groups(compiled)

        shape = (2, compiled.num_nodes, self.width)
        self.reach = np.zeros(shape, dtype=self.dtype)
        self.value = np.zeros(shape, dtype=self.dtype)
        self.terminal_reach = np.zeros((2, compiled.num_terminals, self.width), dtype=self.dtype)
        self.terminal_value = np.zeros((2, compiled.num_terminals, self.width), dtype=self.dtype)

        self.strategy_cache: list[np.ndarray] = []
        self._node_street = np.array([node.street.value for node in tree.nodes], dtype=np.int8)

    # ---- layout helpers --------------------------------------------------

    def _slot_index(self, group: NodeGroup, chunk: slice) -> np.ndarray:
        num_buckets = self.compiled.tree.num_buckets(group.street)
        span = np.arange(num_buckets * group.num_actions, dtype=np.int64)
        return group.slot_base[chunk][:, None] + span[None, :]

    def _strategy_block(
        self, group: NodeGroup, chunk: slice, *, use_average: bool = False
    ) -> np.ndarray:
        num_actions = group.num_actions
        num_buckets = self.compiled.tree.num_buckets(group.street)
        source = self.strategy_sum if use_average else self.regrets
        block = source[self._slot_index(group, chunk)].reshape(-1, num_buckets, num_actions)

        positive = np.maximum(block, 0.0)
        total = positive.sum(axis=-1, keepdims=True)
        uniform = self.dtype(1.0 / num_actions)
        return np.where(total > 0, positive / np.where(total > 0, total, 1.0), uniform)

    def _child_targets(self, group: NodeGroup, chunk: slice) -> tuple[np.ndarray, np.ndarray]:
        base = group.edge_base[chunk]
        edges = base[:, None] + np.arange(group.num_actions, dtype=np.int64)[None, :]
        return self.compiled.edge_target[edges], self.compiled.edge_kind[edges] == EDGE_TO_TERMINAL

    def _street_of(self, target: np.ndarray, terminal: bool) -> np.ndarray:
        source = self.compiled.terminal_street if terminal else self._node_street
        return source[target]

    def _advance(self, ranges: np.ndarray, street: Street, target_street: Street) -> np.ndarray:
        """Carry a range across zero or more street boundaries.

        A single edge can only cross one boundary in this tree, but going through
        the loop rather than asserting that keeps the kernel correct if an
        all-in ever skips a street — a case the betting tree does produce, since
        a showdown terminal can sit on an earlier street than the river.
        """
        current = street
        out = ranges
        while current != target_street:
            step = (current, _NEXT_STREET[current])
            out = out @ self.game.transitions[step]
            current = _NEXT_STREET[current]
        return out

    def _retreat(self, values: np.ndarray, street: Street, target_street: Street) -> np.ndarray:
        """Carry counterfactual values back across the same boundaries.

        The transpose, not the inverse: a value at a parent bucket is the
        transition-weighted average of the child buckets it can become, which is
        exactly what right-multiplying by ``P.T`` computes.
        """
        chain = []
        current = street
        while current != target_street:
            chain.append((current, _NEXT_STREET[current]))
            current = _NEXT_STREET[current]

        out = values
        for step in reversed(chain):
            out = out @ self.game.transitions[step].T
        return out

    # ---- the three passes ------------------------------------------------

    def forward(self, initial_range: np.ndarray, *, use_average: bool = False) -> None:
        self.reach[:] = 0.0
        self.terminal_reach[:] = 0.0
        preflop = self.compiled.tree.num_buckets(Street.PREFLOP)
        self.reach[0, 0, :preflop] = initial_range
        self.reach[1, 0, :preflop] = initial_range
        self.strategy_cache = []

        for group in self.groups:
            actor, other = group.actor, 1 - group.actor
            num_buckets = self.compiled.tree.num_buckets(group.street)
            nodes = group.node_ids
            chunk = slice(0, nodes.shape[0])

            bucket_strategy = self._strategy_block(group, chunk, use_average=use_average)
            self.strategy_cache.append(bucket_strategy)
            targets, is_terminal = self._child_targets(group, chunk)

            actor_reach = self.reach[actor, nodes][:, :num_buckets, None] * bucket_strategy
            other_reach = self.reach[other, nodes][:, :num_buckets]

            if not use_average:
                self._accumulate_strategy_sum(group, chunk, bucket_strategy)

            for action in range(group.num_actions):
                for terminal in (True, False):
                    mask = is_terminal[:, action] == terminal
                    if not mask.any():
                        continue
                    store = self.terminal_reach if terminal else self.reach
                    target = targets[mask, action]
                    self._deposit(
                        store,
                        target,
                        actor,
                        other,
                        actor_reach[mask, :, action],
                        other_reach[mask],
                        group.street,
                        self._street_of(target, terminal),
                    )

    def _deposit(
        self,
        store: np.ndarray,
        target: np.ndarray,
        actor: int,
        other: int,
        actor_reach: np.ndarray,
        other_reach: np.ndarray,
        street: Street,
        child_streets: np.ndarray,
    ) -> None:
        """Write child ranges, applying a transition wherever the street changes."""
        for street_value in np.unique(child_streets):
            same = child_streets == street_value
            child_street = Street(int(street_value))
            width = self.compiled.tree.num_buckets(child_street)
            moved_actor = self._advance(actor_reach[same], street, child_street)
            moved_other = self._advance(other_reach[same], street, child_street)
            store[actor, target[same], :width] = moved_actor
            store[other, target[same], :width] = moved_other

    def evaluate_terminals(self) -> None:
        """Value every terminal from bucket-pair matrices.

        Folds use the compatibility rate for the street the hand ended on;
        showdowns use the river sign matrix, since a showdown always resolves on
        a complete board however early the betting stopped.
        """
        kind = self.compiled.terminal_kind
        magnitude = self.compiled.terminal_value.astype(self.dtype)
        # Rows are written at their own street's width, so stale entries beyond
        # it would survive from a previous iteration and be read back by a
        # parent on a wider street.
        self.terminal_value[:] = 0.0

        for street_value in np.unique(self.compiled.terminal_street):
            street = Street(int(street_value))
            width = self.compiled.tree.num_buckets(street)
            on_street = self.compiled.terminal_street == street_value

            folds = np.flatnonzero(on_street & (kind == TerminalKind.FOLD))
            if folds.shape[0]:
                rate = self.game.compatible[street]
                for player in (0, 1):
                    sign = 1.0 if player == 0 else -1.0
                    mass = self.terminal_reach[1 - player, folds, :width] @ rate
                    self.terminal_value[player, folds, :width] = (
                        sign * magnitude[folds][:, None] * mass
                    )

            showdowns = np.flatnonzero(on_street & (kind == TerminalKind.SHOWDOWN))
            if showdowns.shape[0]:
                scale = magnitude[showdowns][:, None]
                for player in (0, 1):
                    # An all-in before the river still runs the board out, so the
                    # showdown resolves in river buckets whatever street the
                    # betting stopped on: carry the opponent's range forward,
                    # settle it there, and carry the value back to this street.
                    reach = self.terminal_reach[1 - player, showdowns, :width]
                    at_river = self._advance(reach, street, Street.RIVER)
                    beaten = scale * (at_river @ self.game.showdown.T)
                    self.terminal_value[player, showdowns, :width] = self._retreat(
                        beaten, street, Street.RIVER
                    )

    def backward(self) -> None:
        self.value[:] = 0.0
        cached = iter(reversed(self.strategy_cache))

        for group in reversed(self.groups):
            actor, other = group.actor, 1 - group.actor
            num_buckets = self.compiled.tree.num_buckets(group.street)
            nodes = group.node_ids
            chunk = slice(0, nodes.shape[0])

            strategy = next(cached)
            targets, is_terminal = self._child_targets(group, chunk)

            actor_children = self._gather_children(group, targets, is_terminal, actor)
            other_children = self._gather_children(group, targets, is_terminal, other)

            actor_value = (strategy * actor_children).sum(axis=-1)
            self.value[actor, nodes, :num_buckets] = actor_value
            self.value[other, nodes, :num_buckets] = other_children.sum(axis=-1)

            self._apply_regrets(group, chunk, actor_children, actor_value)

    def _gather_children(
        self, group: NodeGroup, targets: np.ndarray, is_terminal: np.ndarray, player: int
    ) -> np.ndarray:
        """``(nodes, buckets, actions)`` child values, pulled back to this street."""
        num_buckets = self.compiled.tree.num_buckets(group.street)
        out = np.zeros((targets.shape[0], num_buckets, targets.shape[1]), dtype=self.dtype)

        for action in range(targets.shape[1]):
            for terminal in (True, False):
                mask = is_terminal[:, action] == terminal
                if not mask.any():
                    continue
                store = self.terminal_value if terminal else self.value
                target = targets[mask, action]
                for street_value in np.unique(self._street_of(target, terminal)):
                    child_street = Street(int(street_value))
                    same = self._street_of(target, terminal) == street_value
                    rows = target[same]
                    width = self.compiled.tree.num_buckets(child_street)
                    values = store[player, rows, :width]
                    pulled = self._retreat(values, group.street, child_street)
                    index = np.flatnonzero(mask)[same]
                    out[index, :, action] = pulled
        return out

    def _apply_regrets(
        self,
        group: NodeGroup,
        chunk: slice,
        actor_children: np.ndarray,
        actor_value: np.ndarray,
    ) -> None:
        block = actor_children - actor_value[:, :, None]
        index = self._slot_index(group, chunk)
        updated = self.regrets[index] + block.reshape(block.shape[0], -1)
        if self.cfr_plus:
            np.maximum(updated, 0.0, out=updated)
        self.regrets[index] = updated

    def _accumulate_strategy_sum(
        self, group: NodeGroup, chunk: slice, bucket_strategy: np.ndarray
    ) -> None:
        num_buckets = self.compiled.tree.num_buckets(group.street)
        mass = self.reach[group.actor, group.node_ids][:, :num_buckets]
        index = self._slot_index(group, chunk)
        self.strategy_sum[index] += (mass[:, :, None] * bucket_strategy).reshape(mass.shape[0], -1)

    def iterate(self, initial_range: np.ndarray) -> None:
        """One iteration covering every infoset over the whole board space."""
        self.iteration += 1
        self.forward(initial_range)
        self.evaluate_terminals()
        self.backward()

    # ---- scoring ---------------------------------------------------------

    def best_response_value(self, br_player: int, initial_range: np.ndarray) -> np.ndarray:
        """Root counterfactual value of ``br_player``'s best response.

        Exploitability *inside this abstract game* — the quantity that must fall
        if the kernel is minimising regret. It is not real-game exploitability,
        which the averaging in ``bucket_game`` floors well above zero; measuring
        that needs the hand-space kernel and a board the size of a real one.

        What this buys is scale. Cross-evaluating against hand space costs one
        pass per board and is unaffordable past a few dozen; this costs two more
        passes over the same board-free tree, so a within-run exploitability
        curve is computable at full production size in minutes.

        The responder maximises per bucket, which is all its information set
        contains, and no hand→bucket collapse is needed because there are no
        hands. Only the responder's value chain is computed: the opponent's would
        be a sum over the responder's actions, which presumes it mixes.
        """
        self.forward(initial_range, use_average=True)
        self.evaluate_terminals()
        self.value[:] = 0.0

        for group in reversed(self.groups):
            num_buckets = self.compiled.tree.num_buckets(group.street)
            nodes = group.node_ids
            chunk = slice(0, nodes.shape[0])
            targets, is_terminal = self._child_targets(group, chunk)
            children = self._gather_children(group, targets, is_terminal, br_player)

            if group.actor == br_player:
                self.value[br_player, nodes, :num_buckets] = children.max(axis=-1)
            else:
                # The actor's own probabilities already rode down in the reach
                # vector, so the responder's value sums over actions.
                self.value[br_player, nodes, :num_buckets] = children.sum(axis=-1)

        return self.value[br_player, 0, : self.compiled.tree.num_buckets(Street.PREFLOP)]

    def exploitability(self, initial_range: np.ndarray) -> float:
        """Mean best-response gain of both players, in chips per dealt hand.

        Two normalisations are needed and both are easy to get wrong.

        The root value is a *double* sum: ``value[b]`` already carries the
        opponent's whole range, so summing it over ``b`` sums over every ordered
        pair of holdings. Dividing by the root mass once — as if it were a single
        sum — inflates the figure by roughly the number of hands.

        And ``value[b]`` is the value *per hand* in bucket ``b``, not for the
        bucket as a whole, so the outer sum must be mass-weighted rather than
        flat. A bucket holding twice the combos contributes twice.

        Dividing by the compatible-pair mass then puts this in the same units as
        :meth:`~src.engine.solver.vector.kernel.VectorCFR.exploitability` —
        chips per dealt hand — so the two kernels' numbers can be compared.
        """
        weights = np.asarray(initial_range, dtype=self.dtype)
        pairs = float(weights @ self.game.compatible[Street.PREFLOP] @ weights)
        gains = [
            float(weights @ self.best_response_value(player, initial_range)) / pairs
            for player in (0, 1)
        ]
        return (gains[0] + gains[1]) / 2.0


__all__: Sequence[str] = ("DEFAULT_DTYPE", "BucketVectorCFR")
