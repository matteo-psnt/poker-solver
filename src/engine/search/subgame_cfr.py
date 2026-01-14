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
never enter (the honesty contract shared with leaf-value estimation).
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from src.core.game.actions import ActionType
from src.core.game.evaluator import get_evaluator
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import ALL_COMBOS, COMBO_MASKS, NUM_COMBOS
from src.engine.search.tree_builder import LocalTree, LocalTreeNode

_EPS = 1e-12
_DECK: list[Card] = Card.get_full_deck()
_CARD_INDEX: dict[int, int] = {card.mask: i for i, card in enumerate(_DECK)}

# (card_a_index, card_b_index) per combo, aligned with ALL_COMBOS.
_COMBO_CARDS = np.array(
    [(_CARD_INDEX[c1.mask], _CARD_INDEX[c2.mask]) for c1, c2 in ALL_COMBOS], dtype=np.int64
)


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
        board_mask = 0
        for card in board:
            board_mask |= card.mask

        alive = np.nonzero((COMBO_MASKS & board_mask) == 0)[0]
        ranks = np.array([evaluator.evaluate(ALL_COMBOS[i], board) for i in alive], dtype=np.int64)
        # Sort best -> worst (smaller rank wins).
        order = np.argsort(ranks, kind="stable")
        self.sorted_combo = alive[order]  # global combo index per sorted position
        sorted_ranks = ranks[order]
        self.n_alive = len(alive)
        self.alive = alive

        # Group boundaries: runs of equal rank.
        boundaries = [0]
        for pos in range(1, self.n_alive):
            if sorted_ranks[pos] != sorted_ranks[pos - 1]:
                boundaries.append(pos)
        self.group_starts = np.array(boundaries, dtype=np.int64)
        self.num_groups = len(boundaries)
        group_of_pos = np.searchsorted(self.group_starts, np.arange(self.n_alive), side="right")
        group_of_pos -= 1

        # Per-combo group id (global combo index -> group), -1 when not alive.
        self.group_of_combo = np.full(NUM_COMBOS, -1, dtype=np.int64)
        self.group_of_combo[self.sorted_combo] = group_of_pos

        # Per-card structures: sorted positions of combos containing each card,
        # flattened, plus per-combo pointers to its own group's start/end within
        # each of its two cards' position lists.
        card_positions: list[list[int]] = [[] for _ in range(52)]
        for pos in range(self.n_alive):
            a, b = _COMBO_CARDS[self.sorted_combo[pos]]
            card_positions[a].append(pos)
            card_positions[b].append(pos)

        self.card_pos_flat = np.array(
            [p for positions in card_positions for p in positions], dtype=np.int64
        )
        card_offsets = np.zeros(53, dtype=np.int64)
        for k in range(52):
            card_offsets[k + 1] = card_offsets[k] + len(card_positions[k])
        self.card_offsets = card_offsets

        # Pointers (flat indices into per-card suffix-sum arrays) per combo/card:
        # start of own group and start of strictly-worse groups.
        group_end = np.append(self.group_starts[1:], self.n_alive)
        ptr_group = np.zeros((NUM_COMBOS, 2), dtype=np.int64)
        ptr_worse = np.zeros((NUM_COMBOS, 2), dtype=np.int64)
        for pos in range(self.n_alive):
            combo = int(self.sorted_combo[pos])
            g = group_of_pos[pos]
            for side in (0, 1):
                k = int(_COMBO_CARDS[combo][side])
                positions = card_positions[k]
                base = int(card_offsets[k])
                ptr_group[combo, side] = base + bisect_left(positions, int(self.group_starts[g]))
                ptr_worse[combo, side] = base + bisect_right(positions, int(group_end[g]) - 1)
        self.ptr_group = ptr_group
        self.ptr_worse = ptr_worse

    def masses(self, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (win, tie, alive) mass vectors over all combos for reach ``reach``.

        ``win[h]`` = reach mass of alive combos strictly worse than ``h`` that do
        not share a card with ``h``; ``tie[h]`` likewise for equal rank;
        ``alive[h]`` = total non-blocking reach mass. Lose mass is
        ``alive - win - tie``. Entries for combos not alive on this board are 0.
        """
        w_sorted = reach[self.sorted_combo]
        group_sums = np.add.reduceat(w_sorted, self.group_starts)
        # Mass in groups strictly after g: suffix sums with sentinel 0.
        suffix = np.zeros(self.num_groups + 1, dtype=np.float64)
        suffix[:-1] = np.cumsum(group_sums[::-1])[::-1]
        total = float(suffix[0])

        # Per-card suffix sums over sorted positions (flat, one sentinel per card).
        card_vals = w_sorted[self.card_pos_flat]
        card_suffix = np.zeros(len(card_vals) + 52, dtype=np.float64)
        for k in range(52):
            lo, hi = int(self.card_offsets[k]), int(self.card_offsets[k + 1])
            seg = card_vals[lo:hi]
            # flat layout with per-card sentinel: [suffix(seg), 0]
            card_suffix[lo + k : hi + k + 1] = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))

        combos = self.sorted_combo
        g = self.group_of_combo[combos]
        a_idx = _COMBO_CARDS[combos, 0]
        b_idx = _COMBO_CARDS[combos, 1]
        # Adjust flat pointers for the per-card sentinels inserted before them.
        card_of_ptr_a = a_idx
        card_of_ptr_b = b_idx
        pw_a = card_suffix[self.ptr_worse[combos, 0] + card_of_ptr_a]
        pw_b = card_suffix[self.ptr_worse[combos, 1] + card_of_ptr_b]
        pg_a = card_suffix[self.ptr_group[combos, 0] + card_of_ptr_a]
        pg_b = card_suffix[self.ptr_group[combos, 1] + card_of_ptr_b]

        worse_total = suffix[g + 1]
        group_total = group_sums[g]
        self_mass = reach[combos]

        win = np.zeros(NUM_COMBOS, dtype=np.float64)
        tie = np.zeros(NUM_COMBOS, dtype=np.float64)
        alive = np.zeros(NUM_COMBOS, dtype=np.float64)

        # Card-a/b mass within the worse set (no both-cards combo can be there).
        win[combos] = worse_total - pw_a - pw_b
        # Card mass within own group: suffix(group start) - suffix(worse start).
        in_group_a = pg_a - pw_a
        in_group_b = pg_b - pw_b
        # Blocked in-group mass = a + b - w[h] (`h` is the only both-cards combo
        # and is double-counted); excluding the blocked set already excludes h.
        tie[combos] = group_total - in_group_a - in_group_b + self_mass
        per_card_all = np.zeros(52, dtype=np.float64)
        np.add.at(per_card_all, a_idx, self_mass)
        np.add.at(per_card_all, b_idx, self_mass)
        alive[combos] = total - per_card_all[a_idx] - per_card_all[b_idx] + self_mass

        np.maximum(win, 0.0, out=win)
        np.maximum(tie, 0.0, out=tie)
        np.maximum(alive, 0.0, out=alive)
        return win, tie, alive


@dataclass
class SubgameSolution:
    """Root output of a local range-vs-range CFR solve."""

    root_actions: list
    # Average root strategy per combo: shape (NUM_COMBOS, num_root_actions).
    root_strategy: np.ndarray
    # Root counterfactual action values for the traversing player, per combo.
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
    min_iterations: int = 8,
    max_iterations: int | None = None,
) -> SubgameSolution:
    """Run RM+ CFR over the local tree; both players adapt per combo.

    Leaves are valued call-then-check-down on ``num_runouts`` sampled boards
    (exact single board when the root is already on the river). Iterates until
    the wall-clock budget runs out (at least ``min_iterations``); pass
    ``max_iterations`` for deterministic tests.
    """
    root = tree.root
    if not root.actions:
        raise ValueError("Subgame tree has no root actions.")

    evaluators = _sample_runout_evaluators(root.state, num_runouts)
    node_data: dict[int, _NodeData] = {}
    _init_nodes(root, node_data)

    deadline = time.perf_counter() + budget_ms / 1000.0
    iterations = 0
    while True:
        _cfr_pass(
            root,
            hero=hero,
            reach_hero=hero_range.astype(np.float64, copy=True),
            reach_opp=opponent_range.astype(np.float64, copy=True),
            rules=rules,
            evaluators=evaluators,
            node_data=node_data,
        )
        iterations += 1
        if max_iterations is not None:
            # Fixed-iteration mode (deterministic tests): the wall clock must not
            # cut the loop short, or results vary with machine load.
            if iterations >= max_iterations:
                break
        elif iterations >= min_iterations and time.perf_counter() >= deadline:
            break

    root_nd = node_data[id(root)]
    avg = root_nd.strategy_sum.copy()
    totals = avg.sum(axis=1, keepdims=True)
    uniform = np.full(avg.shape[1], 1.0 / avg.shape[1])
    avg = np.where(totals > _EPS, avg / np.maximum(totals, _EPS), uniform)

    return SubgameSolution(
        root_actions=list(root.actions),
        root_strategy=avg,
        root_values=root_nd.last_action_values,
        iterations=iterations,
    )


@dataclass
class _NodeData:
    regrets: np.ndarray  # (NUM_COMBOS, A), RM+ (clipped at 0)
    strategy_sum: np.ndarray  # reach-weighted average strategy accumulator
    last_action_values: np.ndarray  # (NUM_COMBOS, A) actor counterfactual values


def _init_nodes(node: LocalTreeNode, node_data: dict[int, _NodeData]) -> None:
    if node.is_leaf or not node.children:
        return
    n_actions = len(node.actions)
    node_data[id(node)] = _NodeData(
        regrets=np.zeros((NUM_COMBOS, n_actions)),
        strategy_sum=np.zeros((NUM_COMBOS, n_actions)),
        last_action_values=np.zeros((NUM_COMBOS, n_actions)),
    )
    for child in node.children:
        _init_nodes(child, node_data)


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
    *,
    hero: int,
    reach_hero: np.ndarray,
    reach_opp: np.ndarray,
    rules,
    evaluators: list[RunoutEvaluator],
    node_data: dict[int, _NodeData],
) -> tuple[np.ndarray, np.ndarray]:
    """One CFR traversal; returns (v_hero, v_opp) counterfactual value vectors."""
    if node.is_leaf or not node.children:
        return _leaf_values(node.state, hero, reach_hero, reach_opp, rules, evaluators)

    nd = node_data[id(node)]
    actor_is_hero = node.state.current_player == hero
    reach_actor = reach_hero if actor_is_hero else reach_opp

    strategy = _regret_matching(nd.regrets)
    nd.strategy_sum += reach_actor[:, None] * strategy

    n_actions = len(node.actions)
    v_hero = np.zeros(NUM_COMBOS)
    v_opp = np.zeros(NUM_COMBOS)
    action_values = np.zeros((NUM_COMBOS, n_actions))
    for a_idx, child in enumerate(node.children):
        sigma_a = strategy[:, a_idx]
        if actor_is_hero:
            child_vh, child_vo = _cfr_pass(
                child,
                hero=hero,
                reach_hero=reach_hero * sigma_a,
                reach_opp=reach_opp,
                rules=rules,
                evaluators=evaluators,
                node_data=node_data,
            )
            action_values[:, a_idx] = child_vh
            v_hero += sigma_a * child_vh
            v_opp += child_vo
        else:
            child_vh, child_vo = _cfr_pass(
                child,
                hero=hero,
                reach_hero=reach_hero,
                reach_opp=reach_opp * sigma_a,
                rules=rules,
                evaluators=evaluators,
                node_data=node_data,
            )
            action_values[:, a_idx] = child_vo
            v_opp += sigma_a * child_vo
            v_hero += child_vh

    v_actor = v_hero if actor_is_hero else v_opp
    nd.regrets += action_values - v_actor[:, None]
    np.maximum(nd.regrets, 0.0, out=nd.regrets)  # RM+
    nd.last_action_values = action_values
    return v_hero, v_opp


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    totals = positive.sum(axis=1, keepdims=True)
    n_actions = regrets.shape[1]
    uniform = np.full(n_actions, 1.0 / n_actions)
    return np.where(totals > _EPS, positive / np.maximum(totals, _EPS), uniform)


def _leaf_values(
    state: GameState,
    hero: int,
    reach_hero: np.ndarray,
    reach_opp: np.ndarray,
    rules,
    evaluators: list[RunoutEvaluator],
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual value vectors at a leaf (terminal or depth-limit)."""
    opp = 1 - hero

    # Fold terminal: pot goes to the non-folder, cards never matter — so the
    # alive mass is taken against the ROOT board (embedded in the reach vectors),
    # not against any sampled runout.
    if (
        state.is_terminal
        and state.betting_history
        and state.betting_history[-1].type == ActionType.FOLD
    ):
        folder = state.current_player
        pot, invested = _pot_and_invested(state)
        hero_payoff = -invested[hero] if folder == hero else pot - invested[hero]
        opp_payoff = -invested[opp] if folder == opp else pot - invested[opp]
        return hero_payoff * _nonblocking_mass(reach_opp), opp_payoff * _nonblocking_mass(
            reach_hero
        )

    # Showdown / depth-limit: complete any pending call, then check down.
    call_state = _complete_pending_call(state, rules)
    pot, invested = _pot_and_invested(call_state)

    v_hero = np.zeros(NUM_COMBOS)
    v_opp = np.zeros(NUM_COMBOS)
    count_hero = np.zeros(NUM_COMBOS)
    count_opp = np.zeros(NUM_COMBOS)
    for ev in evaluators:
        win_h, tie_h, alive_h = ev.masses(reach_opp)
        v_hero += win_h * pot + tie_h * (pot / 2.0) - invested[hero] * alive_h
        count_hero[ev.alive] += 1.0

        win_o, tie_o, alive_o = ev.masses(reach_hero)
        v_opp += win_o * pot + tie_o * (pot / 2.0) - invested[opp] * alive_o
        count_opp[ev.alive] += 1.0

    np.divide(v_hero, count_hero, out=v_hero, where=count_hero > 0)
    np.divide(v_opp, count_opp, out=v_opp, where=count_opp > 0)
    return v_hero, v_opp


def _nonblocking_mass(reach: np.ndarray) -> np.ndarray:
    """Per-combo total reach mass of combos not sharing a card (inclusion-exclusion)."""
    per_card = np.zeros(52, dtype=np.float64)
    np.add.at(per_card, _COMBO_CARDS[:, 0], reach)
    np.add.at(per_card, _COMBO_CARDS[:, 1], reach)
    total = float(reach.sum())
    return total - per_card[_COMBO_CARDS[:, 0]] - per_card[_COMBO_CARDS[:, 1]] + reach


def _pot_and_invested(state: GameState) -> tuple[float, tuple[float, float]]:
    starting = (state.pot + state.stacks[0] + state.stacks[1]) / 2.0
    return float(state.pot), (starting - state.stacks[0], starting - state.stacks[1])


def _complete_pending_call(state: GameState, rules) -> GameState:
    """Fold a pending bet into the pot (call-then-check-down leaf valuation)."""
    if state.is_terminal or state.to_call <= 0:
        return state
    caller = state.current_player
    call_amount = min(state.to_call, state.stacks[caller])
    stacks = list(state.stacks)
    stacks[caller] -= call_amount
    return GameState(
        street=state.street,
        pot=state.pot + call_amount,
        stacks=(stacks[0], stacks[1]),
        board=state.board,
        hole_cards=state.hole_cards,
        betting_history=state.betting_history,
        button_position=state.button_position,
        current_player=state.current_player,
        is_terminal=state.is_terminal,
        to_call=0,
        last_aggressor=state.last_aggressor,
        blind_to_call=state.blind_to_call,
        _skip_validation=True,
    )
