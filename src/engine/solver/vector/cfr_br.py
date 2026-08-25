"""CFR-BR: the CFR player trains against a best response in the REAL game.

Johanson, Bard, Burch & Bowling, AAAI-12. Plain CFR converges to an equilibrium
of the ABSTRACT game, which is provably not the least-exploitable strategy the
abstraction can represent; CFR-BR converges to (approximately) that one instead,
because the opponent is not a regret-matched abstract mixer but a best response
computed with its exact holding. Two details the paper is explicit about and
that are easy to get backwards:

    the BR responds to the CFR player's CURRENT strategy on that same
    iteration, not to its average -- Theorem 3's proof is exactly the statement
    that a pointwise best response has non-positive regret;

    the OUTPUT is the CFR player's average strategy. Under the Hybrid-agent
    (below) the paper's Theorem 6 guarantees the average and not the current
    strategy, so ``strategy_sum`` remains the deliverable.

**Where the best response is allowed to act.** A per-hand argmax is a legal
strategy only where the player's information set already contains everything the
argmax read. Our sampler draws ONE full board per iteration, so the last sampled
chance event is the river deal: an argmax at a flop or turn node would be
choosing with the runout in hand -- clairvoyance, "not an opponent that exists"
(``mixture._visible_partition``). Hence the paper's Hybrid-agent split, with the
boundary placed where our sampling stops rather than where theirs does:

    trunk       preflop, flop, turn -- the opponent plays regret matching over
                its own regret table (``trunk_regrets``), which is a regret
                minimiser, so Theorem 5/6 still apply
    subgame     river -- exact per-hand best response, discarded each iteration

``br_streets`` moves that boundary. Widening it past what the board sample
supports prices in clairvoyance and is an ABLATION, not the algorithm. Feeding a
whole enumerated board set as one iteration's contexts removes the restriction
entirely: the joint maximisation over boards sharing a street's visible prefix
is then a real best response at every street, which is how the toy game checks
the mechanism.

Three tree passes per iteration against plain PCS's one: one that computes both
seats' hybrid strategies from the current iterates, then one CFR update per
seat, each needing its own forward pass because the opponent's reach differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street
from src.engine.solver.numba_ops import compute_dcfr_strategy_weight
from src.engine.solver.vector.kernel import (
    DTYPE,
    NodeGroup,
    VectorCFR,
    build_groups,
    regret_match,
)
from src.engine.solver.vector.mixture import GLOBAL_HANDS, VISIBLE_CARDS
from src.engine.solver.vector.pcs import STREETS, dcfr_discount

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.engine.solver.betting_tree import BettingTree
    from src.engine.solver.vector.compiled_tree import CompiledTree
    from src.engine.solver.vector.hand_context import HandContext

# Where the hybrid opponent best-responds instead of regret-matching. `river` is
# the only one a one-board-per-iteration sample supports; the rest are ablations
# that let the opponent act on cards it has not been dealt.
BR_REGIONS: dict[str, tuple[Street, ...]] = {
    "river": (Street.RIVER,),
    "turn_river": (Street.TURN, Street.RIVER),
    "postflop": (Street.FLOP, Street.TURN, Street.RIVER),
    "all": STREETS,
}


class TrunkLayout:
    """Compact ``(node, bucket, action)`` slots for the trunk, both seats at once.

    The hybrid opponent's trunk regrets are a different object from either
    seat's blueprint -- they minimise regret against the CFR player's sequence,
    not against a hybrid -- so they need their own store. Sized to the trunk
    only, which is where it earns its place: at production settings the river
    holds 93% of the table's rows, so leaving them out makes this array a
    rounding error against the two it sits beside instead of a third copy of
    them. Both seats fit one array because a node has one actor.
    """

    def __init__(self, tree: BettingTree, br_streets: Sequence[Street]):
        excluded = frozenset(br_streets)
        base = np.full(len(tree), -1, dtype=np.int64)
        cursor = 0
        for node in tree.nodes:
            if node.street in excluded:
                continue
            base[node.node_id] = cursor
            cursor += tree.num_buckets(node.street) * node.num_actions
        self.base = base
        self.num_slots = cursor


class HybridAgent:
    """One seat's CFR-BR opponent on one board: trunk regret matching, river BR.

    ``picks`` is filled by :meth:`CFRBestResponse.hybrid_pass` during the
    backward walk and read by the forward pass that follows, which is why it is
    keyed by the ``(group, chunk)`` position the kernel counts in forward order.
    """

    def __init__(
        self,
        player: int,
        *,
        br_streets: frozenset[Street],
        trunk_regrets: np.ndarray,
        layout: TrunkLayout,
        compiled: CompiledTree,
        positions: int,
    ):
        self.player = player
        self.br_streets = br_streets
        self.trunk_regrets = trunk_regrets
        self.layout = layout
        self.compiled = compiled
        self.picks: list[np.ndarray | None] = [None] * positions

    def trunk_index(self, group: NodeGroup, chunk: slice) -> np.ndarray:
        num_buckets = self.compiled.tree.num_buckets(group.street)
        span = (
            np.arange(num_buckets, dtype=np.int64)[:, None] * group.num_actions
            + np.arange(group.num_actions, dtype=np.int64)[None, :]
        ).ravel()
        return self.layout.base[group.node_ids[chunk]][:, None] + span[None, :]

    def trunk_strategy(self, group: NodeGroup, chunk: slice) -> np.ndarray:
        """``(n, B, A)`` regret matching over the trunk table."""
        num_buckets = self.compiled.tree.num_buckets(group.street)
        block = self.trunk_regrets[self.trunk_index(group, chunk)].reshape(
            -1, num_buckets, group.num_actions
        )
        return regret_match(block, np.float32(1.0 / group.num_actions))

    def block(self, group: NodeGroup, chunk: slice, position: int) -> tuple[np.ndarray, bool]:
        if group.street not in self.br_streets:
            return self.trunk_strategy(group, chunk), False
        actions = self.picks[position]
        if actions is None:
            raise RuntimeError(
                f"No best response recorded at position {position} "
                f"({group.street.name}, actor {group.actor}). The hybrid pass must run "
                "before the CFR update pass that reads it."
            )
        return actions, True


def apply_regret_block(
    store: np.ndarray,
    index: np.ndarray,
    block: np.ndarray,
    occupied: np.ndarray,
    *,
    discount: tuple[float, float] | None,
    cfr_plus: bool,
) -> None:
    """``VectorCFR.apply_regret_block`` against an arbitrary flat store."""
    rows = block.shape[0]
    stored = store[index]
    if discount is not None:
        positive, negative = discount
        current = stored.reshape(rows, block.shape[1], block.shape[2])
        touched = current[:, occupied, :]
        current[:, occupied, :] = np.where(
            touched > 0, touched * DTYPE(positive), touched * DTYPE(negative)
        )
    updated = stored + block.reshape(rows, -1)
    if cfr_plus:
        np.maximum(updated, 0.0, out=updated)
    store[index] = updated


class CFRBestResponse:
    """CFR-BR over sampled boards, writing the same ``(node, bucket, action)`` table.

    ``regrets``/``strategy_sum`` are the caller's shared blueprint arrays and
    carry only the CFR player's rows for each seat; ``trunk_regrets`` is the
    hybrid opponent's own store and is deliberately NOT checkpointed -- it is
    scaffolding for the opponent, not part of the answer, and a resumed task
    simply relearns it.
    """

    def __init__(
        self,
        compiled: CompiledTree,
        regrets: np.ndarray,
        strategy_sum: np.ndarray,
        trunk_regrets: np.ndarray,
        *,
        br_streets: Sequence[Street] = (Street.RIVER,),
        weighting: str = "linear",
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.0,
        dcfr_gamma: float = 2.0,
        cfr_plus: bool = True,
        showdown: str = "walk",
        num_boards: int = 1,
    ):
        if weighting not in ("none", "linear", "dcfr"):
            raise ValueError(f"Unknown iteration weighting {weighting!r}.")
        self.compiled = compiled
        self.regrets = regrets
        self.strategy_sum = strategy_sum
        self.trunk_regrets = trunk_regrets
        self.br_streets = frozenset(br_streets)
        self.weighting = weighting
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma
        self.cfr_plus = cfr_plus
        self.showdown = showdown
        self.num_boards = num_boards
        self.groups: list[NodeGroup] = build_groups(compiled)
        self.layout = TrunkLayout(compiled.tree, tuple(self.br_streets))
        if trunk_regrets.shape[0] < self.layout.num_slots:
            raise ValueError(
                f"trunk_regrets holds {trunk_regrets.shape[0]} slots; this trunk needs "
                f"{self.layout.num_slots}. Both sides must derive it from the same tree "
                "and the same br_streets."
            )
        self.kernels: list[VectorCFR] = []
        self.agents: list[dict[int, HybridAgent]] = []
        self.plan: list[tuple[NodeGroup, slice]] = []
        self._delta: np.ndarray | None = None
        self._global_hand_id: list[np.ndarray] = []
        self._partition: dict[Street, list[list[int]]] = {}
        self.boards = 0  # board passes completed, for throughput accounting
        self.best_responses = 0  # (node group, board) blocks a best response chose at
        # Per-pass controls, set by _schedule.
        self.strategy_weight = 1.0
        self.delta_scale = 1.0
        self.regret_discount: tuple[float, float] | None = None

    # ---- binding ---------------------------------------------------------

    def _bind(self, contexts: Sequence[HandContext]) -> None:
        if len(contexts) != self.num_boards:
            raise ValueError(
                f"This driver is built for {self.num_boards} boards per iteration; "
                f"got {len(contexts)}. Scratch is sized once."
            )
        if not self.kernels:
            for context in contexts:
                kernel = VectorCFR(
                    self.compiled,
                    context,
                    cfr_plus=self.cfr_plus,
                    showdown=self.showdown,
                    groups=self.groups,
                )
                kernel.regrets = self.regrets
                kernel.strategy_sum = self.strategy_sum
                self.kernels.append(kernel)
            first = self.kernels[0]
            self.plan = [(group, chunk) for group in self.groups for chunk in first.chunks(group)]
            self.agents = [
                {
                    player: HybridAgent(
                        player,
                        br_streets=self.br_streets,
                        trunk_regrets=self.trunk_regrets,
                        layout=self.layout,
                        compiled=self.compiled,
                        positions=len(self.plan),
                    )
                    for player in (0, 1)
                }
                for _ in contexts
            ]
        else:
            for kernel, context in zip(self.kernels, contexts, strict=True):
                kernel.bind(context)
        self._global_hand_id = [
            context.hand_cards[:, 0].astype(np.int64) * 52 + context.hand_cards[:, 1]
            for context in contexts
        ]

    def _partition_boards(self, boards: Sequence[Sequence[int] | np.ndarray] | None) -> None:
        """Board indices grouped by what is face up on each street.

        Two runouts a player cannot yet tell apart are ONE observation, so its
        best response must pick one action for the whole group. Without the
        board cards every board is alone in its group, which is right for a
        single sampled board and is why ``br_streets`` must then stop at the
        river.
        """
        if boards is None:
            self._partition = {street: [[i] for i in range(self.num_boards)] for street in STREETS}
            return
        self._partition = {}
        for street in STREETS:
            visible = VISIBLE_CARDS[street]
            groups: dict[tuple[int, ...], list[int]] = {}
            for index, board in enumerate(boards):
                groups.setdefault(tuple(sorted(int(c) for c in board[:visible])), []).append(index)
            self._partition[street] = list(groups.values())

    # ---- schedule --------------------------------------------------------

    def _schedule(self, iteration: int) -> None:
        if self.weighting == "dcfr":
            weight = float(compute_dcfr_strategy_weight(iteration, self.dcfr_gamma))
            scale = 1.0
            self.regret_discount = dcfr_discount(iteration, self.dcfr_alpha, self.dcfr_beta)
        elif self.weighting == "linear":
            weight = scale = float(max(iteration, 1))
            self.regret_discount = None
        else:
            weight = scale = 1.0
            self.regret_discount = None
        self.strategy_weight = weight / self.num_boards
        self.delta_scale = scale / self.num_boards

    # ---- the three passes ------------------------------------------------

    def prepare(
        self,
        contexts: Sequence[HandContext],
        iteration: int,
        boards: Sequence[Sequence[int] | np.ndarray] | None = None,
    ) -> None:
        """Bind the boards and set this iteration's weights; :meth:`iterate` calls it."""
        self._bind(contexts)
        self._partition_boards(boards)
        self._schedule(iteration)

    def iterate(
        self,
        contexts: Sequence[HandContext],
        iteration: int,
        boards: Sequence[Sequence[int] | np.ndarray] | None = None,
    ) -> None:
        """One CFR-BR iteration on the sampled board(s) at absolute ``iteration``.

        ``boards`` are the five-card boards behind ``contexts``. They decide
        which runouts the best response may tell apart; omit them only when
        every street in ``br_streets`` is fully dealt by then.
        """
        self.prepare(contexts, iteration, boards=boards)
        initial = np.ones(self.kernels[0].num_hands, dtype=DTYPE)

        for kernel in self.kernels:
            kernel.opponent = None
            kernel.update_players = ()
            kernel.strategy_weight = self.strategy_weight
            kernel.delta_scale = self.delta_scale
            kernel.regret_discount = self.regret_discount
            kernel.forward(initial)
            kernel.evaluate_terminals()
            kernel.value[:] = 0.0
        self.hybrid_pass()

        for player in (0, 1):
            self._cfr_pass(player, initial)
        self.boards += self.num_boards

    def hybrid_pass(self) -> None:
        """Both seats' hybrid strategies for this iteration, and their trunk regrets.

        One backward walk carries both value chains: at a node the actor's own
        chain maximises (river) or mixes over its trunk strategy, while the
        non-actor's chain is a plain sum over the actor's actions -- correct
        because the actor's probabilities already rode down in the forward
        reach, which is exactly the current iterate the best response is
        supposed to answer.
        """
        for position in range(len(self.plan) - 1, -1, -1):
            group, chunk = self.plan[position]
            actor, other = group.actor, 1 - group.actor
            nodes = group.node_ids[chunk]
            actor_children, other_children = [], []
            for kernel in self.kernels:
                targets, is_terminal = kernel.child_targets(group, chunk)
                actor_children.append(kernel.gather_children(targets, is_terminal, actor))
                other_children.append(kernel.gather_children(targets, is_terminal, other))

            if group.street in self.br_streets:
                self._maximise(group, position, actor, nodes, actor_children)
            else:
                self._mix_trunk(group, chunk, actor, nodes, actor_children)

            for kernel, children in zip(self.kernels, other_children, strict=True):
                kernel.value[other, nodes] = children.sum(axis=-1)

    def _maximise(
        self,
        group: NodeGroup,
        position: int,
        actor: int,
        nodes: np.ndarray,
        children: list[np.ndarray],
    ) -> None:
        """Per-hand argmax, joint over boards this street cannot tell apart."""
        for members in self._partition[group.street]:
            if len(members) == 1:
                only = members[0]
                per_board = {only: children[only].argmax(axis=-1).astype(np.int8)}
            else:
                totals = np.zeros(
                    (children[0].shape[0], GLOBAL_HANDS, group.num_actions), dtype=DTYPE
                )
                for index in members:
                    totals[:, self._global_hand_id[index], :] += children[index]
                best = totals.argmax(axis=-1)
                per_board = {
                    index: best[:, self._global_hand_id[index]].astype(np.int8) for index in members
                }
            for index, chosen in per_board.items():
                self.agents[index][actor].picks[position] = chosen
                self.best_responses += 1
                self.kernels[index].value[actor, nodes] = np.take_along_axis(
                    children[index], chosen.astype(np.int64)[:, :, None], axis=2
                )[:, :, 0]

    def _mix_trunk(
        self,
        group: NodeGroup,
        chunk: slice,
        actor: int,
        nodes: np.ndarray,
        children: list[np.ndarray],
    ) -> None:
        """Trunk value under regret matching, and the trunk regret this leaves."""
        agent = self.agents[0][actor]
        strategy = agent.trunk_strategy(group, chunk)
        num_buckets = self.compiled.tree.num_buckets(group.street)
        block = np.zeros((nodes.shape[0], num_buckets, group.num_actions), dtype=DTYPE)
        occupied: set[int] = set()

        for kernel, child in zip(self.kernels, children, strict=True):
            segments = kernel.segments[group.street]
            value = (strategy[:, kernel.context.buckets_for(group.street), :] * child).sum(axis=-1)
            kernel.value[actor, nodes] = value
            collapsed = np.add.reduceat(
                child[:, segments.hand_order, :], segments.segment_start, axis=1
            )
            node_value = np.add.reduceat(
                value[:, segments.hand_order], segments.segment_start, axis=1
            )
            block[:, segments.segment_bucket, :] += collapsed - node_value[:, :, None]
            occupied.update(segments.segment_bucket.tolist())

        if self.delta_scale != 1.0:
            block *= DTYPE(self.delta_scale)
        apply_regret_block(
            self.trunk_regrets,
            agent.trunk_index(group, chunk),
            block,
            np.array(sorted(occupied), dtype=np.int64),
            discount=self.regret_discount,
            cfr_plus=self.cfr_plus,
        )

    def _cfr_pass(self, player: int, initial: np.ndarray) -> None:
        """One seat's CFR update against the hybrid opponent just computed."""
        for index, kernel in enumerate(self.kernels):
            kernel.opponent = self.agents[index][1 - player]
            kernel.update_players = (player,)

        if self.num_boards == 1:
            kernel = self.kernels[0]
            kernel.forward(initial)
            kernel.evaluate_terminals()
            kernel.backward()
            kernel.opponent = None
            return

        if self._delta is None:
            self._delta = np.zeros(self.compiled.tree.num_slots, dtype=DTYPE)
        self._delta[:] = 0.0
        occupied: dict[Street, set[int]] = {street: set() for street in STREETS}
        for kernel in self.kernels:
            kernel.regret_target = self._delta
            try:
                kernel.forward(initial)
                kernel.evaluate_terminals()
                kernel.backward()
            finally:
                kernel.regret_target = None
            for street in STREETS:
                occupied[street].update(kernel.segments[street].segment_bucket.tolist())

        union = {
            street: np.array(sorted(rows), dtype=np.int64) for street, rows in occupied.items()
        }
        first = self.kernels[0]
        for group in self.groups:
            if group.actor != player:
                continue
            buckets = self.compiled.tree.num_buckets(group.street)
            for chunk in first.chunks(group):
                index = first._slot_index(group, chunk)  # noqa: SLF001 -- the kernel's own layout
                block = self._delta[index].reshape(-1, buckets, group.num_actions)
                first.apply_regret_block(index, block, union[group.street])
        for kernel in self.kernels:
            kernel.opponent = None


__all__ = ("BR_REGIONS", "CFRBestResponse", "HybridAgent", "TrunkLayout")
