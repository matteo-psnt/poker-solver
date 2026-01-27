"""Range-vs-range CFR over the resolver's local lookahead tree.

This replaces the old root-only "fast CFR" (regret matching against a fixed
value vector — which just converges to argmax, i.e. a best response to a frozen
forecast). Here both players carry per-combo strategies at every decision node
in the local tree, so the opponent counter-adapts inside the solve and the root
strategy is balanced rather than greedily exploitable.

Valuation model
---------------
Terminal nodes are valued exactly, range-vs-range, with card-removal
correction (see :class:`RunoutEvaluator`). Depth-limit and chance-node leaves
are valued as *call-then-check-down*: any pending bet is called and the hand is
checked to showdown on sampled runouts. This drops the blueprint's future-street
betting from leaf values — the standard poor-man's depth limit; multiple biased
continuation strategies (Pluribus-style) are the known upgrade.

Everything is a function of public state + ranges: the opponent's dealt cards
never enter (the honesty contract shared with the LBR evaluator).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.core.game.evaluator import get_evaluator
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import ALL_COMBOS, COMBO_CARDS, NUM_COMBOS, blocked_combos
from src.engine.search.tree_builder import LocalTree, LocalTreeNode

_EPS = 1e-12
_MIN_ITERATIONS = 8
_DECK: list[Card] = Card.get_full_deck()
_CARD_A = COMBO_CARDS[:, 0]
_CARD_B = COMBO_CARDS[:, 1]


class RunoutEvaluator:
    """Exact per-combo showdown masses vs a reach vector on one complete board.

    For every alive combo ``h`` and opponent reach vector ``w`` computes
    ``W[h]/T[h]/L[h]`` — the total reach mass of opponent combos that lose to /
    tie / beat ``h`` — with exact card-removal (combos sharing a card with ``h``
    are excluded via inclusion-exclusion over per-card rank-suffix sums), in
    O(n) per query after O(n log n) setup.
    """

    def __init__(self, board: tuple[Card, ...]):
        if len(board) != 5:
            raise ValueError(f"RunoutEvaluator needs a complete board, got {len(board)} cards")
        evaluator = get_evaluator()
        alive = np.nonzero(~blocked_combos(board))[0]
        ranks = np.array([evaluator.evaluate(ALL_COMBOS[i], board) for i in alive], dtype=np.int64)
        # Sort best -> worst (smaller rank wins).
        order = np.argsort(ranks, kind="stable")
        self.sorted_combo = alive[order]  # global combo index per sorted position
        sorted_ranks = ranks[order]
        self.n_alive = len(alive)
        self.alive = alive

        # Group boundaries: runs of equal rank.
        self.group_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_ranks)) + 1)).astype(
            np.int64
        )
        self.num_groups = len(self.group_starts)
        group_end = np.append(self.group_starts[1:], self.n_alive)
        group_of_pos = np.searchsorted(self.group_starts, np.arange(self.n_alive), side="right") - 1
        self._group = group_of_pos  # group id per sorted position

        # Per-combo card deck-indices in sorted order (for bincount scatter).
        self._a_idx = COMBO_CARDS[self.sorted_combo, 0]
        self._b_idx = COMBO_CARDS[self.sorted_combo, 1]

        # Per-card structures: sorted positions of combos containing each card,
        # flattened card-by-card, plus per-position pointers into the flat global
        # suffix-sum array: own-group start, strictly-worse start, and segment end.
        card_positions: list[list[int]] = [[] for _ in range(52)]
        for pos in range(self.n_alive):
            card_positions[int(self._a_idx[pos])].append(pos)
            card_positions[int(self._b_idx[pos])].append(pos)

        self.card_pos_flat = np.array(
            [p for positions in card_positions for p in positions], dtype=np.int64
        )
        card_offsets = np.zeros(53, dtype=np.int64)
        np.cumsum([len(p) for p in card_positions], out=card_offsets[1:])

        ptr_group = np.zeros((self.n_alive, 2), dtype=np.int64)
        ptr_worse = np.zeros((self.n_alive, 2), dtype=np.int64)
        seg_end = np.zeros((self.n_alive, 2), dtype=np.int64)
        for k in range(52):
            positions = np.array(card_positions[k], dtype=np.int64)
            if len(positions) == 0:
                continue
            member_groups = group_of_pos[positions]
            base = card_offsets[k]
            pg = base + np.searchsorted(positions, self.group_starts[member_groups], side="left")
            pw = base + np.searchsorted(positions, group_end[member_groups], side="left")
            side = (self._a_idx[positions] != k).astype(np.int64)  # 0 => card a, 1 => card b
            ptr_group[positions, side] = pg
            ptr_worse[positions, side] = pw
            seg_end[positions, side] = card_offsets[k + 1]
        self._ptr_group = ptr_group
        self._ptr_worse = ptr_worse
        self._seg_end = seg_end

    def masses(self, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (win, tie, alive) mass vectors over all combos for reach ``reach``.

        ``win[h]`` = reach mass of alive combos strictly worse than ``h`` that do
        not share a card with ``h``; ``tie[h]`` likewise for equal rank;
        ``alive[h]`` = total non-blocking reach mass. Lose mass is
        ``alive - win - tie``. Entries for combos not alive on this board are 0.
        """
        w_sorted = reach[self.sorted_combo]
        group_sums = np.add.reduceat(w_sorted, self.group_starts)
        # Mass in groups strictly after g: group-level suffix sums with sentinel 0.
        suffix = np.zeros(self.num_groups + 1, dtype=np.float64)
        suffix[:-1] = np.cumsum(group_sums[::-1])[::-1]
        total = float(suffix[0])

        # Global suffix sums over the flat per-card position list; a per-card
        # segment's local suffix is the global suffix minus its value at the
        # segment end (segments are contiguous), so no per-card loop is needed.
        card_vals = w_sorted[self.card_pos_flat]
        global_suffix = np.zeros(len(card_vals) + 1, dtype=np.float64)
        global_suffix[:-1] = np.cumsum(card_vals[::-1])[::-1]

        end_a = global_suffix[self._seg_end[:, 0]]
        end_b = global_suffix[self._seg_end[:, 1]]
        pw_a = global_suffix[self._ptr_worse[:, 0]] - end_a
        pw_b = global_suffix[self._ptr_worse[:, 1]] - end_b
        pg_a = global_suffix[self._ptr_group[:, 0]] - end_a
        pg_b = global_suffix[self._ptr_group[:, 1]] - end_b

        g = self._group
        worse_total = suffix[g + 1]
        group_total = group_sums[g]
        self_mass = w_sorted

        win = np.zeros(NUM_COMBOS, dtype=np.float64)
        tie = np.zeros(NUM_COMBOS, dtype=np.float64)
        alive = np.zeros(NUM_COMBOS, dtype=np.float64)

        combos = self.sorted_combo
        # Card-a/b mass within the worse set (no both-cards combo can be there).
        win[combos] = worse_total - pw_a - pw_b
        # Blocked in-group mass = a + b - w[h] (`h` is the only both-cards combo
        # and is double-counted); excluding the blocked set already excludes h.
        tie[combos] = group_total - (pg_a - pw_a) - (pg_b - pw_b) + self_mass
        per_card = np.bincount(self._a_idx, weights=self_mass, minlength=52) + np.bincount(
            self._b_idx, weights=self_mass, minlength=52
        )
        alive[combos] = total - per_card[self._a_idx] - per_card[self._b_idx] + self_mass

        np.maximum(win, 0.0, out=win)
        np.maximum(tie, 0.0, out=tie)
        np.maximum(alive, 0.0, out=alive)
        return win, tie, alive


@dataclass
class SubgameSolution:
    """Root output of a local range-vs-range CFR solve."""

    # Average root strategy per combo: shape (NUM_COMBOS, num_root_actions).
    root_strategy: np.ndarray
    # Root counterfactual action values for the traversing player, per combo,
    # from the final iteration.
    root_values: np.ndarray
    iterations: int


def solve_subgame(
    tree: LocalTree,
    *,
    hero: int,
    hero_range: np.ndarray,
    opponent_range: np.ndarray,
    rules,
    budget_ms: int,
    num_runouts: int = 4,
    max_iterations: int | None = None,
) -> SubgameSolution:
    """Run RM+ CFR over the local tree; both players adapt per combo.

    Leaves are valued call-then-check-down on ``num_runouts`` sampled boards
    (exact single board when the root is already on the river). Iterates until
    the wall-clock budget runs out (at least ``_MIN_ITERATIONS``); pass
    ``max_iterations`` to pin the iteration count instead — results become
    machine/load-independent (the wall clock is ignored entirely).
    """
    root = tree.root
    if not root.actions:
        raise ValueError("Subgame tree has no root actions.")

    evaluators = _sample_runout_evaluators(root.state, num_runouts)
    # Per-combo count of runouts where the combo is alive (for averaging).
    alive_count = np.zeros(NUM_COMBOS, dtype=np.float64)
    for evaluator in evaluators:
        alive_count[evaluator.alive] += 1.0

    node_data: dict[int, _NodeData] = {}
    leaf_specs: dict[int, _LeafSpec] = {}
    _prepare_nodes(root, hero, rules, node_data, leaf_specs)

    reach_hero = hero_range.astype(np.float64)
    reach_opp = opponent_range.astype(np.float64)
    ctx = _PassContext(
        hero=hero,
        evaluators=evaluators,
        alive_count=alive_count,
        node_data=node_data,
        leaf_specs=leaf_specs,
    )

    deadline = time.perf_counter() + budget_ms / 1000.0
    iterations = 0
    root_values = np.zeros((NUM_COMBOS, len(root.actions)))
    while True:
        _, _, pass_values = _cfr_pass(root, reach_hero, reach_opp, ctx)
        assert pass_values is not None  # the root is a decision node, never a leaf
        root_values = pass_values
        iterations += 1
        if max_iterations is not None:
            if iterations >= max_iterations:
                break
        elif iterations >= _MIN_ITERATIONS and time.perf_counter() >= deadline:
            break

    avg = node_data[id(root)].strategy_sum
    totals = avg.sum(axis=1, keepdims=True)
    uniform = np.full(avg.shape[1], 1.0 / avg.shape[1])
    return SubgameSolution(
        root_strategy=np.where(totals > _EPS, avg / np.maximum(totals, _EPS), uniform),
        root_values=root_values,
        iterations=iterations,
    )


@dataclass
class _NodeData:
    regrets: np.ndarray  # (NUM_COMBOS, A), RM+ (clipped at 0)
    strategy_sum: np.ndarray  # reach-weighted average strategy accumulator


@dataclass(frozen=True)
class _LeafSpec:
    """Iteration-invariant leaf facts, precomputed once per solve."""

    is_fold: bool
    hero_payoff: float  # fold leaves only
    opp_payoff: float
    pot: float  # showdown / depth-limit leaves (after completing a pending call)
    invested: tuple[float, float]


@dataclass(frozen=True)
class _PassContext:
    """Iteration-invariant inputs threaded through the CFR recursion."""

    hero: int
    evaluators: list[RunoutEvaluator]
    alive_count: np.ndarray
    node_data: dict[int, _NodeData]
    leaf_specs: dict[int, _LeafSpec]


def _prepare_nodes(
    node: LocalTreeNode,
    hero: int,
    rules,
    node_data: dict[int, _NodeData],
    leaf_specs: dict[int, _LeafSpec],
) -> None:
    """Allocate per-node CFR state and precompute leaf facts."""
    if node.is_leaf or not node.children:
        state = node.state
        if state.is_terminal and state.ended_by_fold:
            leaf_specs[id(node)] = _LeafSpec(
                is_fold=True,
                hero_payoff=float(state.get_payoff(hero, rules)),
                opp_payoff=float(state.get_payoff(1 - hero, rules)),
                pot=0.0,
                invested=(0.0, 0.0),
            )
        else:
            call_state = _complete_pending_call(state)
            leaf_specs[id(node)] = _LeafSpec(
                is_fold=False,
                hero_payoff=0.0,
                opp_payoff=0.0,
                pot=float(call_state.pot),
                invested=rules.invested_chips(call_state),
            )
        return

    n_actions = len(node.actions)
    node_data[id(node)] = _NodeData(
        regrets=np.zeros((NUM_COMBOS, n_actions)),
        strategy_sum=np.zeros((NUM_COMBOS, n_actions)),
    )
    for child in node.children:
        _prepare_nodes(child, hero, rules, node_data, leaf_specs)


def _sample_runout_evaluators(state: GameState, num_runouts: int) -> list[RunoutEvaluator]:
    """Evaluators on completed boards; exact when the board is already complete."""
    board = state.board
    if len(board) == 5:
        return [RunoutEvaluator(board)]

    board_mask = 0
    for card in board:
        board_mask |= card.mask
    unseen = [card for card in _DECK if not (card.mask & board_mask)]
    missing = 5 - len(board)

    evaluators = []
    for _ in range(max(1, num_runouts)):
        picks = np.random.choice(len(unseen), size=missing, replace=False)
        runout = tuple(unseen[int(i)] for i in picks)
        evaluators.append(RunoutEvaluator(board + runout))
    return evaluators


def _cfr_pass(
    node: LocalTreeNode,
    reach_hero: np.ndarray,
    reach_opp: np.ndarray,
    ctx: _PassContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One CFR traversal; returns (v_hero, v_opp, actor action values or None)."""
    if node.is_leaf or not node.children:
        v_hero, v_opp = _leaf_values(ctx.leaf_specs[id(node)], ctx, reach_hero, reach_opp)
        return v_hero, v_opp, None

    nd = ctx.node_data[id(node)]
    actor_is_hero = node.state.current_player == ctx.hero
    reach_actor = reach_hero if actor_is_hero else reach_opp

    strategy = _regret_matching(nd.regrets)
    nd.strategy_sum += reach_actor[:, None] * strategy

    v_hero = np.zeros(NUM_COMBOS)
    v_opp = np.zeros(NUM_COMBOS)
    action_values = np.zeros((NUM_COMBOS, len(node.actions)))
    for a_idx, child in enumerate(node.children):
        sigma_a = strategy[:, a_idx]
        if actor_is_hero:
            child_vh, child_vo, _ = _cfr_pass(child, reach_hero * sigma_a, reach_opp, ctx)
            action_values[:, a_idx] = child_vh
            v_hero += sigma_a * child_vh
            v_opp += child_vo
        else:
            child_vh, child_vo, _ = _cfr_pass(child, reach_hero, reach_opp * sigma_a, ctx)
            action_values[:, a_idx] = child_vo
            v_opp += sigma_a * child_vo
            v_hero += child_vh

    v_actor = v_hero if actor_is_hero else v_opp
    nd.regrets += action_values - v_actor[:, None]
    np.maximum(nd.regrets, 0.0, out=nd.regrets)  # RM+
    return v_hero, v_opp, action_values


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    # RM+ invariant: `regrets` is clipped at 0 in place after every update, so
    # it is already nonnegative here.
    totals = regrets.sum(axis=1, keepdims=True)
    uniform = np.full(regrets.shape[1], 1.0 / regrets.shape[1])
    return np.where(totals > _EPS, regrets / np.maximum(totals, _EPS), uniform)


def _leaf_values(
    spec: _LeafSpec,
    ctx: _PassContext,
    reach_hero: np.ndarray,
    reach_opp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual value vectors at a leaf (terminal or depth-limit)."""
    # Fold: pot goes to the non-folder, cards never matter — so the alive mass
    # is taken against the ROOT board (embedded in the reach vectors), not
    # against any sampled runout.
    if spec.is_fold:
        return (
            spec.hero_payoff * _nonblocking_mass(reach_opp),
            spec.opp_payoff * _nonblocking_mass(reach_hero),
        )

    hero, opp = ctx.hero, 1 - ctx.hero
    pot = spec.pot
    v_hero = np.zeros(NUM_COMBOS)
    v_opp = np.zeros(NUM_COMBOS)
    for evaluator in ctx.evaluators:
        win_h, tie_h, alive_h = evaluator.masses(reach_opp)
        v_hero += win_h * pot + tie_h * (pot / 2.0) - spec.invested[hero] * alive_h

        win_o, tie_o, alive_o = evaluator.masses(reach_hero)
        v_opp += win_o * pot + tie_o * (pot / 2.0) - spec.invested[opp] * alive_o

    count = ctx.alive_count
    np.divide(v_hero, count, out=v_hero, where=count > 0)
    np.divide(v_opp, count, out=v_opp, where=count > 0)
    return v_hero, v_opp


def _nonblocking_mass(reach: np.ndarray) -> np.ndarray:
    """Per-combo total reach mass of combos not sharing a card (inclusion-exclusion)."""
    per_card = np.bincount(_CARD_A, weights=reach, minlength=52) + np.bincount(
        _CARD_B, weights=reach, minlength=52
    )
    return float(reach.sum()) - per_card[_CARD_A] - per_card[_CARD_B] + reach


def _complete_pending_call(state: GameState) -> GameState:
    """Fold a pending bet into the pot (call-then-check-down leaf valuation)."""
    if state.is_terminal or state.to_call <= 0:
        return state
    caller = state.current_player
    call_amount = min(state.to_call, state.stacks[caller])
    stacks = list(state.stacks)
    stacks[caller] -= call_amount
    return state.replace(
        pot=state.pot + call_amount,
        stacks=(stacks[0], stacks[1]),
        to_call=0,
        validate=False,
    )
