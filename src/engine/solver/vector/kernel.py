"""Vector-form CFR over a compiled betting tree.

One iteration is three passes over the tree, each level-by-level rather than
recursive:

    forward     push both players' range vectors down, node level by node level,
                until every node and every terminal holds the range that reaches
                it
    terminals   value every terminal at once, from the ranges the forward pass
                deposited — folds by blocker-corrected mass, showdowns by one
                matrix product against the board's win/lose sign matrix
    backward    pull counterfactual values back up, and at each node collapse
                the per-hand regrets onto their bucket rows

Nothing here is per-node Python. Nodes are grouped by ``(level, action count,
street, acting seat)`` — all four fixed by the public tree, none of them
data-dependent — and each group is a handful of array operations on a dense
``(nodes, hands, actions)`` block. There are at most a few hundred such groups
against 57,604 nodes, which is the whole point: the interpreter runs a few
hundred times per iteration instead of tens of millions.

Player indexing is button-relative throughout, matching the tree's
button-symmetric node identity: player 0 is the button, player 1 is not.
``CompiledTree.terminal_value`` is likewise the button's payoff, so the
non-button's terminal value is its negation.

Values carried:
    reach[p][n]     probability that p's hand h reaches node n under the current
                    strategy, one entry per live hand
    value[p][n]     counterfactual value: p's expected payoff from n downward
                    with the OPPONENT's reach folded in and p's own reach to n
                    left out. This is the quantity CFR's regret is a difference
                    of, which is why the non-actor's value at a node is a plain
                    sum over actions — the actor's probabilities already entered
                    through the reach vector the forward pass pushed down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.vector.compiled_tree import EDGE_TO_TERMINAL, CompiledTree, TerminalKind
from src.engine.solver.vector.hand_context import HandContext, showdown_matrix
from src.engine.solver.vector.showdown import RankWalk

if TYPE_CHECKING:
    from collections.abc import Sequence

# A standard deck, fixing the width of the per-card range-mass table.
NUM_CARDS = 52

# Blocks of (nodes x hands x actions) floats are the kernel's working set. This
# caps one block's element count so a wide level does not allocate a multi-GB
# temporary; groups larger than this are processed in several chunks.
MAX_BLOCK_ELEMENTS = 8_000_000

DTYPE = np.float32


@dataclass(frozen=True, slots=True)
class NodeGroup:
    """Nodes that share a shape, so one array op covers all of them.

    Per node: ``node_ids``, ``slot_base`` (``tree.slot_base``) and ``edge_base``
    (``edge_offset``). ``slot_stride`` is one scalar — a street shares it.
    """

    level: int
    num_actions: int
    street: Street
    actor: int  # 0 when the button acts at these nodes, else 1
    node_ids: np.ndarray
    slot_base: np.ndarray
    slot_stride: int
    edge_base: np.ndarray


def slot_index(compiled: CompiledTree, group: NodeGroup, chunk: slice) -> np.ndarray:
    """``(n, B * A)`` storage indices for a chunk's ragged rows."""
    num_buckets = compiled.tree.num_buckets(group.street)
    span = (
        np.arange(num_buckets, dtype=np.int64)[:, None] * group.slot_stride
        + np.arange(group.num_actions, dtype=np.int64)[None, :]
    ).ravel()
    return group.slot_base[chunk][:, None] + span[None, :]


def child_targets(
    compiled: CompiledTree, group: NodeGroup, chunk: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Per-action child ids and a terminal mask, shaped ``(n, A)``."""
    base = group.edge_base[chunk]
    edges = base[:, None] + np.arange(group.num_actions, dtype=np.int64)[None, :]
    return compiled.edge_target[edges], compiled.edge_kind[edges] == EDGE_TO_TERMINAL


def strategy_block(
    source: np.ndarray, compiled: CompiledTree, group: NodeGroup, chunk: slice, uniform: np.floating
) -> np.ndarray:
    """Regret matching over ``source``, shaped ``(n, B, A)``; both kernels share it."""
    num_actions = group.num_actions
    num_buckets = compiled.tree.num_buckets(group.street)
    block = source[slot_index(compiled, group, chunk)].reshape(-1, num_buckets, num_actions)

    positive = np.maximum(block, 0.0)
    total = positive.sum(axis=-1, keepdims=True)
    return np.where(total > 0, positive / np.where(total > 0, total, 1.0), uniform)


@dataclass(frozen=True, slots=True)
class BucketSegments:
    """Hand ordering that turns the hand→bucket collapse into a segment sum.

    Sorting hands by bucket once per street makes every bucket a contiguous run,
    so accumulating many hands' regrets onto one bucket row is
    ``np.add.reduceat`` over that axis — one call for a whole block — rather than
    a scattered ``np.add.at``, which is roughly an order of magnitude slower.
    """

    hand_order: np.ndarray
    segment_start: np.ndarray
    segment_bucket: np.ndarray


def build_groups(compiled: CompiledTree) -> list[NodeGroup]:
    """Partition nodes into shape-identical groups, ordered by level."""
    tree = compiled.tree
    keys: dict[tuple[int, int, Street, int], list[int]] = {}
    for node in tree.nodes:
        key = (
            int(compiled.depth[node.node_id]),
            node.num_actions,
            node.street,
            0 if node.actor_is_button else 1,
        )
        keys.setdefault(key, []).append(node.node_id)

    groups = []
    for (level, num_actions, street, actor), ids in sorted(keys.items(), key=lambda kv: kv[0][0]):
        node_ids = np.array(ids, dtype=np.int64)
        groups.append(
            NodeGroup(
                level=level,
                num_actions=num_actions,
                street=street,
                actor=actor,
                node_ids=node_ids,
                slot_base=tree.slot_base[node_ids],
                slot_stride=int(tree.slot_stride[node_ids[0]]),
                edge_base=compiled.edge_offset[node_ids],
            )
        )
    return groups


def build_segments(bucket_of_hand: np.ndarray) -> BucketSegments:
    """Sort hands by bucket and record where each bucket's run begins."""
    hand_order = np.argsort(bucket_of_hand, kind="stable").astype(np.int64)
    sorted_buckets = bucket_of_hand[hand_order]
    boundary = np.flatnonzero(np.diff(sorted_buckets)) + 1
    segment_start = np.concatenate([[0], boundary]).astype(np.int64)
    return BucketSegments(
        hand_order=hand_order,
        segment_start=segment_start,
        segment_bucket=sorted_buckets[segment_start].astype(np.int64),
    )


class VectorCFR:
    """Full-tree vector CFR+ on one sampled public board.

    The board is an input rather than a branch: sampling it once per iteration
    (public chance sampling) and vectorising over private hands is what keeps a
    HUNL-sized tree tractable, since enumerating runouts is the combinatorial
    term and enumerating hands is not.

    Storage is the ragged ``(node, bucket, action)`` layout ``BettingTree``
    already defines, so regrets written here are readable by anything that
    already understands that layout.
    """

    def __init__(
        self,
        compiled: CompiledTree,
        context: HandContext,
        *,
        cfr_plus: bool = True,
        showdown: str = "matmul",
        groups: list[NodeGroup] | None = None,
    ):
        self.compiled = compiled
        self.cfr_plus = cfr_plus
        self.showdown = showdown
        self.iteration = 0

        tree = compiled.tree
        self.num_hands = context.num_hands
        self.regrets = np.zeros(tree.num_slots, dtype=DTYPE)
        self.strategy_sum = np.zeros(tree.num_slots, dtype=DTYPE)

        self.groups = build_groups(compiled) if groups is None else groups

        shape = (2, compiled.num_nodes, self.num_hands)
        self.reach = np.zeros(shape, dtype=DTYPE)
        self.value = np.zeros(shape, dtype=DTYPE)
        self.terminal_reach = np.zeros((2, compiled.num_terminals, self.num_hands), dtype=DTYPE)
        self.terminal_value = np.zeros((2, compiled.num_terminals, self.num_hands), dtype=DTYPE)

        self.strategy_cache: list[np.ndarray] = []
        # When set, regret increments land here instead of in ``regrets`` and the
        # CFR+ floor is the caller's business. See ``_scatter_to_buckets``.
        self.regret_target: np.ndarray | None = None
        # Per-pass controls a sampling trainer sets before each board; the
        # defaults are plain simultaneous CFR+ and leave every write unchanged.
        self.strategy_weight = 1.0  # multiplies the strategy_sum increment (t^gamma under DCFR)
        self.delta_scale = 1.0  # multiplies the regret increment (t under linear weighting)
        # (positive, negative) multipliers on stored regrets of bucket rows this
        # board OCCUPIES, applied before the increment -- DCFR's discount placed
        # per visit, as the scalar kernel places it.
        self.regret_discount: tuple[float, float] | None = None
        self.update_players: tuple[int, ...] = (0, 1)  # whose rows this pass writes
        self.bind(context)

    def bind(self, context: HandContext) -> None:
        """Point the kernel at another board, keeping its scratch arrays.

        Every five-card board leaves the same number of live hands, so a
        sampling trainer allocates the ~3 GB of per-hand scratch once and only
        the board-specific tables change here: bucket segments, the showdown
        order, and which cards each hand holds.
        """
        if context.num_hands != self.num_hands:
            raise ValueError(
                f"Board has {context.num_hands} live hands; this kernel is shaped for "
                f"{self.num_hands}. Scratch is sized once, so every board must agree."
            )
        self.context = context
        self.segments = {
            street: build_segments(context.buckets_for(street))
            for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
        }
        if self.showdown == "walk":
            self.rank_walk = RankWalk(context.showdown_rank, context.hand_cards)
        else:
            self.showdown_sign = showdown_matrix(context.showdown_rank, context.blocks)
        # (H, 52) incidence, so per-card range mass is one matrix product. Folds
        # need only blocker-corrected mass, and going through it costs O(H x 52)
        # where the showdown path costs O(H x H) -- a ~20x difference across
        # 50,952 fold terminals, which is why they do not share a code path.
        self.card_incidence = np.zeros((self.num_hands, NUM_CARDS), dtype=DTYPE)
        self.card_incidence[np.arange(self.num_hands)[:, None], context.hand_cards] = 1.0
        self.hand_cards = context.hand_cards

    # ---- per-group shape helpers ----------------------------------------

    def chunks(self, group: NodeGroup) -> list[slice]:
        per_node = self.context.num_hands * group.num_actions
        step = max(1, MAX_BLOCK_ELEMENTS // per_node)
        total = group.node_ids.shape[0]
        return [slice(start, min(start + step, total)) for start in range(0, total, step)]

    def _slot_index(self, group: NodeGroup, chunk: slice) -> np.ndarray:
        return slot_index(self.compiled, group, chunk)

    def _strategy_block(
        self, group: NodeGroup, chunk: slice, *, use_average: bool = False
    ) -> np.ndarray:
        """The current iterate, or with ``use_average`` the normalised strategy
        sum — the only form carrying CFR's convergence guarantee, so anything
        scoring the solver must ask for it."""
        source = self.strategy_sum if use_average else self.regrets
        return strategy_block(
            source, self.compiled, group, chunk, np.float32(1.0 / group.num_actions)
        )

    def child_targets(self, group: NodeGroup, chunk: slice) -> tuple[np.ndarray, np.ndarray]:
        return child_targets(self.compiled, group, chunk)

    # ---- the three passes ------------------------------------------------

    def forward(self, initial_range: np.ndarray, *, use_average: bool = False) -> None:
        """Push both players' ranges down to every node and terminal.

        Bucket-space strategies are cached here for the backward pass to reuse.
        Regret matching plus the gather it reads is ~15% of an iteration, and
        the strategy cannot change between the two passes — recomputing it was
        paying that twice. The cache is one float per storage slot.
        """
        self.reach[:] = 0.0
        self.terminal_reach[:] = 0.0
        self.reach[0, 0] = initial_range
        self.reach[1, 0] = initial_range
        self.strategy_cache = []

        for group in self.groups:
            actor, other = group.actor, 1 - group.actor
            buckets = self.context.buckets_for(group.street)
            for chunk in self.chunks(group):
                nodes = group.node_ids[chunk]
                bucket_strategy = self._strategy_block(group, chunk, use_average=use_average)
                self.strategy_cache.append(bucket_strategy)
                strategy = bucket_strategy[:, buckets, :]
                targets, is_terminal = self.child_targets(group, chunk)

                actor_reach = self.reach[actor, nodes][:, :, None] * strategy
                other_reach = self.reach[other, nodes]

                if not use_average:
                    self._accumulate_strategy_sum(group, chunk, bucket_strategy)

                for action in range(group.num_actions):
                    target = targets[:, action]
                    terminal = is_terminal[:, action]
                    self._deposit(
                        self.terminal_reach,
                        target[terminal],
                        actor,
                        other,
                        actor_reach[terminal, :, action],
                        other_reach[terminal],
                    )
                    inner = ~terminal
                    self._deposit(
                        self.reach,
                        target[inner],
                        actor,
                        other,
                        actor_reach[inner, :, action],
                        other_reach[inner],
                    )

    @staticmethod
    def _deposit(
        store: np.ndarray,
        target: np.ndarray,
        actor: int,
        other: int,
        actor_reach: np.ndarray,
        other_reach: np.ndarray,
    ) -> None:
        """Write child ranges. Assignment, not accumulation — the tree is a tree.

        ``CompiledTree.parent_count`` is verified all-ones at production
        settings, so no node is reachable by two edges and no child's range is
        ever a sum of contributions. A DAG would need ``np.add.at`` here.
        """
        if target.shape[0] == 0:
            return
        store[actor, target] = actor_reach
        store[other, target] = other_reach

    def evaluate_terminals(self) -> None:
        """Value every terminal at once, from the ranges the forward pass left.

        Two batched operations cover all 101,904 terminals. Folds take a
        blocker-corrected range mass, which needs no card information beyond
        which holdings collide — the winner is public. Showdowns take one matrix
        product against the board's win/lose sign matrix, with both players'
        halves stacked into a single call so BLAS sees one large multiply
        instead of two.
        """
        kind = self.compiled.terminal_kind
        magnitude = self.compiled.terminal_value.astype(DTYPE)

        folds = np.flatnonzero(kind == TerminalKind.FOLD)
        if folds.shape[0]:
            for player in (0, 1):
                sign = 1.0 if player == 0 else -1.0
                reach = self.terminal_reach[1 - player, folds]
                self.terminal_value[player, folds] = (
                    sign * magnitude[folds][:, None] * self._compatible_mass(reach)
                )

        showdowns = np.flatnonzero(kind == TerminalKind.SHOWDOWN)
        if showdowns.shape[0]:
            scale = magnitude[showdowns][:, None]
            if self.showdown == "walk":
                for player in (0, 1):
                    beaten = self.rank_walk.values(self.terminal_reach[1 - player, showdowns])
                    self.terminal_value[player, showdowns] = scale * beaten
                return
            count = showdowns.shape[0]
            # showdown_sign is antisymmetric, so each player reads its own
            # win/lose direction off the same matrix; only the range it is
            # multiplied against changes.
            stacked = np.concatenate(
                [self.terminal_reach[1, showdowns], self.terminal_reach[0, showdowns]]
            )
            beaten = stacked @ self.showdown_sign.T
            self.terminal_value[0, showdowns] = scale * beaten[:count]
            self.terminal_value[1, showdowns] = scale * beaten[count:]

    def _compatible_mass(self, reach: np.ndarray) -> np.ndarray:
        """``(T, H)`` opponent mass that does not collide with each hand.

        Inclusion-exclusion over the two cards rather than an ``(H, H)`` product:
        total range, minus everything holding either of my cards, plus my own
        holding back because subtracting both cards removed it twice.
        """
        total = reach.sum(axis=1, keepdims=True)
        per_card = reach @ self.card_incidence
        first = per_card[:, self.hand_cards[:, 0]]
        second = per_card[:, self.hand_cards[:, 1]]
        return total - first - second + reach

    def backward(self) -> None:
        """Pull counterfactual values up and write regrets onto bucket rows."""
        self.value[:] = 0.0

        cached = iter(reversed(self.strategy_cache))
        for group in reversed(self.groups):
            actor, other = group.actor, 1 - group.actor
            buckets = self.context.buckets_for(group.street)

            for chunk in reversed(self.chunks(group)):
                nodes = group.node_ids[chunk]
                strategy = next(cached)[:, buckets, :]
                targets, is_terminal = self.child_targets(group, chunk)

                actor_children = self.gather_children(targets, is_terminal, actor)
                other_children = self.gather_children(targets, is_terminal, other)

                actor_value = (strategy * actor_children).sum(axis=-1)
                self.value[actor, nodes] = actor_value
                self.value[other, nodes] = other_children.sum(axis=-1)

                self._scatter_to_buckets(group, chunk, actor_children, actor_value)

    def gather_children(
        self, targets: np.ndarray, is_terminal: np.ndarray, player: int
    ) -> np.ndarray:
        """``(n, H, A)`` child values, reading node or terminal storage per edge."""
        node_values = self.value[player]
        terminal_values = self.terminal_value[player]
        out = np.empty((targets.shape[0], self.context.num_hands, targets.shape[1]), dtype=DTYPE)
        for action in range(targets.shape[1]):
            target = targets[:, action]
            terminal = is_terminal[:, action]
            out[terminal, :, action] = terminal_values[target[terminal]]
            inner = ~terminal
            out[inner, :, action] = node_values[target[inner]]
        return out

    def _scatter_to_buckets(
        self,
        group: NodeGroup,
        chunk: slice,
        actor_children: np.ndarray,
        actor_value: np.ndarray,
    ) -> None:
        """Collapse per-hand regrets onto bucket rows and apply them.

        This is the imperfect-recall accumulation the abstraction forces: many
        hands at one public node share a bucket, so their counterfactual
        advantages sum into a single stored row. Sorting by bucket makes that a
        ``reduceat`` over a contiguous axis.

        The two terms of the advantage are collapsed separately and subtracted
        in bucket space, which is exact — a sum of differences is a difference
        of sums — and avoids materialising a second ``(nodes, hands, actions)``
        array just to hold the subtraction.
        """
        if group.actor not in self.update_players:
            return
        segments = self.segments[group.street]
        num_buckets = self.compiled.tree.num_buckets(group.street)
        num_actions = group.num_actions

        children = np.add.reduceat(
            actor_children[:, segments.hand_order, :], segments.segment_start, axis=1
        )
        node = np.add.reduceat(actor_value[:, segments.hand_order], segments.segment_start, axis=1)

        block = np.zeros((children.shape[0], num_buckets, num_actions), dtype=DTYPE)
        block[:, segments.segment_bucket, :] = children - node[:, :, None]
        if self.delta_scale != 1.0:
            block *= DTYPE(self.delta_scale)

        index = self._slot_index(group, chunk)
        if self.regret_target is None:
            self.apply_regret_block(index, block, segments.segment_bucket)
        else:
            # Deferred mode: this pass is one of several whose updates belong to
            # the same iteration, so the increment goes to a buffer and the
            # CFR+ floor waits until they have all been summed. Flooring each
            # contribution separately would clip a negative that a later one was
            # about to cancel, which is a different algorithm.
            self.regret_target[index] += block.reshape(children.shape[0], -1)

    def apply_regret_block(
        self, index: np.ndarray, block: np.ndarray, occupied: np.ndarray
    ) -> None:
        """``regrets[index] = discount(regrets[index]) + block``, then the CFR+ floor.

        ``block`` is ``(n, B, A)`` and ``occupied`` the buckets this board
        populates. Only those rows are discounted: a row no hand on this board
        falls in received no information, and decaying it anyway is the eager
        schedule the scalar kernel measured 12-18% worse (``numba_ops``).
        """
        rows = block.shape[0]
        stored = self.regrets[index]
        if self.regret_discount is not None:
            positive, negative = self.regret_discount
            current = stored.reshape(rows, block.shape[1], block.shape[2])
            touched = current[:, occupied, :]
            current[:, occupied, :] = np.where(
                touched > 0, touched * DTYPE(positive), touched * DTYPE(negative)
            )
        updated = stored + block.reshape(rows, -1)
        if self.cfr_plus:
            np.maximum(updated, 0.0, out=updated)
        self.regrets[index] = updated

    def _accumulate_strategy_sum(
        self, group: NodeGroup, chunk: slice, bucket_strategy: np.ndarray
    ) -> None:
        """Weight the current iterate by the actor's own reach and accumulate.

        The reach is collapsed to buckets *before* multiplying by the strategy,
        not after. That is exact rather than an approximation: the strategy is
        constant within a bucket, so ``sum_h in b (reach[h] * sigma[b,a])``
        factors as ``(sum_h in b reach[h]) * sigma[b,a]``. It moves the
        reduction from a ``(nodes, hands, actions)`` array to a ``(nodes,
        hands)`` one and the product into the smaller bucket space.
        """
        if group.actor not in self.update_players:
            return
        segments = self.segments[group.street]
        nodes = group.node_ids[chunk]
        reach = self.reach[group.actor, nodes][:, segments.hand_order]
        collapsed = np.add.reduceat(reach, segments.segment_start, axis=1)

        num_buckets = self.compiled.tree.num_buckets(group.street)
        mass = np.zeros((collapsed.shape[0], num_buckets), dtype=DTYPE)
        mass[:, segments.segment_bucket] = collapsed
        if self.strategy_weight != 1.0:
            mass *= DTYPE(self.strategy_weight)

        index = self._slot_index(group, chunk)
        self.strategy_sum[index] += (mass[:, :, None] * bucket_strategy).reshape(
            collapsed.shape[0], -1
        )

    def iterate(self, initial_range: np.ndarray) -> None:
        """One full-tree CFR iteration: every infoset at every node is updated."""
        self.iteration += 1
        self.forward(initial_range)
        self.evaluate_terminals()
        self.backward()

    # ---- scoring ---------------------------------------------------------

    def best_response_value(
        self, br_player: int, initial_range: np.ndarray, *, unconstrained: bool = False
    ) -> np.ndarray:
        """Root counterfactual value of ``br_player``'s best response.

        The opponent plays the *average* strategy — the one CFR's guarantee is
        about.

        ``unconstrained`` decides WHICH exploitability this is, and the two
        answer different questions:

        constrained (default)
            The responder maximises per ``(node, bucket)`` — it sees exactly what
            the abstraction shows. This is exploitability *inside the abstract
            game*, the quantity that must fall to zero if the kernel is
            minimising regret.
        unconstrained
            The responder maximises per ``(node, hand)`` — it sees its actual
            holding on the actual board, as a real opponent does. This cannot
            fall below the abstraction's own error, so it does NOT go to zero,
            and the gap between the two IS that error.

        Reporting only the first is the trap this option exists to close: a
        kernel can drive its own game's exploitability arbitrarily low while
        remaining wide open to anyone who can tell two hands in one bucket
        apart.

        A responder's own reach never enters its counterfactual value, so the
        forward pass need not know a best response is being computed — only the
        opponent's reach is load-bearing, and that is the average strategy's.

        Only the responder's value chain is computed. The opponent's is not just
        unnecessary here, it would be *wrong*: it is a sum over the responder's
        actions, which presumes the responder mixes, and a best response does
        not. Nothing downstream reads it, so it is left untouched rather than
        filled with a value that does not mean what it appears to.
        """
        self.forward(initial_range, use_average=True)
        self.evaluate_terminals()
        self.value[:] = 0.0

        for group in reversed(self.groups):
            for chunk in reversed(self.chunks(group)):
                nodes = group.node_ids[chunk]
                targets, is_terminal = self.child_targets(group, chunk)
                children = self.gather_children(targets, is_terminal, br_player)

                if group.actor == br_player:
                    self.value[br_player, nodes] = (
                        children.max(axis=-1) if unconstrained else self._maximise(group, children)
                    )
                else:
                    # The actor's own probabilities already rode down in the
                    # reach vector, so the responder's value sums over actions.
                    self.value[br_player, nodes] = children.sum(axis=-1)

        return self.value[br_player, 0]

    def _maximise(self, group: NodeGroup, actor_children: np.ndarray) -> np.ndarray:
        """Per-hand value of the best action, chosen once per bucket.

        The choice is made on bucket-summed values because the responder cannot
        see more than the abstraction does: every hand in a bucket shares one
        stored row and therefore one action.
        """
        segments = self.segments[group.street]
        collapsed = np.add.reduceat(
            actor_children[:, segments.hand_order, :], segments.segment_start, axis=1
        )
        best_per_segment = collapsed.argmax(axis=-1)

        num_buckets = self.compiled.tree.num_buckets(group.street)
        best = np.zeros((collapsed.shape[0], num_buckets), dtype=np.int64)
        best[:, segments.segment_bucket] = best_per_segment
        chosen = best[:, self.context.buckets_for(group.street)]
        return np.take_along_axis(actor_children, chosen[:, :, None], axis=2)[:, :, 0]

    def exploitability(
        self, initial_range: np.ndarray, compatible_pairs: float, *, unconstrained: bool = False
    ) -> float:
        """Mean of both players' best-response gains, in chips per hand.

        ``compatible_pairs`` normalises the unnormalised root values: with an
        all-ones initial range, a root value summed over hands is a sum over
        every ordered pair of non-colliding holdings, so dividing by that count
        yields chips per dealt hand. Zero-sum makes the mean of the two gains
        the standard exploitability.
        """
        gains = [
            float(
                self.best_response_value(player, initial_range, unconstrained=unconstrained).sum()
            )
            / compatible_pairs
            for player in (0, 1)
        ]
        return (gains[0] + gains[1]) / 2.0


__all__: Sequence[str] = (
    "DTYPE",
    "MAX_BLOCK_ELEMENTS",
    "BucketSegments",
    "NodeGroup",
    "VectorCFR",
    "build_groups",
    "build_segments",
)
