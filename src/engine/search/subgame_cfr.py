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
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from src.core.game.evaluator import get_evaluator
from src.core.game.state import FULL_DECK, Card, GameState
from src.engine.search.range_inference import ALL_COMBOS, COMBO_CARDS, NUM_COMBOS, blocked_combos
from src.shared.numeric import NORMALIZE_EPS

if TYPE_CHECKING:
    from src.engine.search.tree_builder import LocalTree, LocalTreeNode

_MIN_ITERATIONS = 8
_CARD_A = COMBO_CARDS[:, 0]
_CARD_B = COMBO_CARDS[:, 1]


@dataclass(frozen=True)
class Continuation:
    """One assumed line for the rest of the hand, past a depth-limit leaf.

    The module's valuation model (see the header) drops the blueprint's
    future-street betting and checks down. That is EXACT on the river -- there
    is no future street, so checking down is the rest of the hand -- and is an
    approximation only on flop and turn, which is where the ~25% of leaves that
    stay truncated at any ``max_depth`` live. This type is what makes a
    different assumption expressible.

    ``pot_fraction`` is what EACH player is assumed to put in before showdown,
    as a fraction of the pot at the leaf. Zero is the check-down the module has
    always used, so :data:`CHECK_DOWN` reproduces today's numbers exactly.

    NOT YET WIRED TO A CHOICE. The literature's depth-limited solving gives the
    OPPONENT a choice among several continuations, which is what stops the hero
    exploiting one naive assumption. Doing that properly means adding the choice
    as a decision node in the local tree so CFR learns a strategy over it --
    taking a max inside the leaf instead would break the zero-sum structure the
    solve rests on. Until then this is a parameter with one production value.
    """

    name: str
    pot_fraction: float


CHECK_DOWN = Continuation(name="check-down", pot_fraction=0.0)
"""Today's assumption, and the default everywhere. Exact on the river."""


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
        win = np.zeros(NUM_COMBOS, dtype=np.float64)
        tie = np.zeros(NUM_COMBOS, dtype=np.float64)
        alive = np.zeros(NUM_COMBOS, dtype=np.float64)
        _masses(
            np.ascontiguousarray(reach, dtype=np.float64),
            self.sorted_combo,
            self.group_starts,
            self.card_pos_flat,
            self._seg_end,
            self._ptr_worse,
            self._ptr_group,
            self._group,
            self._a_idx,
            self._b_idx,
            win,
            tie,
            alive,
        )
        return win, tie, alive


@jit(nopython=True, cache=True)
def _pairwise_leaf(a, start, n):
    """numpy's ``pairwise_sum`` below its block size: a plain loop under 8
    elements, eight accumulators up to 128."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[start + i]
        return res
    r0 = a[start]
    r1 = a[start + 1]
    r2 = a[start + 2]
    r3 = a[start + 3]
    r4 = a[start + 4]
    r5 = a[start + 5]
    r6 = a[start + 6]
    r7 = a[start + 7]
    i = 8
    limit = n - (n % 8)
    while i < limit:
        r0 += a[start + i]
        r1 += a[start + i + 1]
        r2 += a[start + i + 2]
        r3 += a[start + i + 3]
        r4 += a[start + i + 4]
        r5 += a[start + i + 5]
        r6 += a[start + i + 6]
        r7 += a[start + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[start + i]
        i += 1
    return res


@jit(nopython=True, cache=True)
def _pairwise_sum(a, start, n):
    """numpy's ``pairwise_sum`` over ``a[start:start + n]``, bit for bit.

    ``np.sum`` and the per-segment reduce behind ``np.add.reduceat`` both use
    it, and reproducing their association is what keeps these kernels'
    output byte-identical to the vectorised numpy they replaced (checked
    empirically). Above 128 elements numpy recurses on a split at the
    largest multiple of 8 below the half; that tree is walked here with an
    explicit stack because a CACHED kernel that calls itself does not reload
    safely -- the second process to import it segfaulted.
    """
    if n <= 128:
        return _pairwise_leaf(a, start, n)
    starts = np.empty(64, dtype=np.int64)
    sizes = np.empty(64, dtype=np.int64)
    lefts = np.empty(64, dtype=np.float64)
    has_left = np.zeros(64, dtype=np.int64)
    starts[0] = start
    sizes[0] = n
    depth = 1
    while True:
        top = depth - 1
        if sizes[top] > 128:
            half = sizes[top] // 2
            half -= half % 8
            starts[depth] = starts[top]
            sizes[depth] = half
            has_left[depth] = 0
            depth += 1
            continue
        value = _pairwise_leaf(a, starts[top], sizes[top])
        depth -= 1
        while True:
            if depth == 0:
                return value
            parent = depth - 1
            if has_left[parent] == 0:
                lefts[parent] = value
                has_left[parent] = 1
                half = sizes[parent] // 2
                half -= half % 8
                starts[depth] = starts[parent] + half
                sizes[depth] = sizes[parent] - half
                has_left[depth] = 0
                depth += 1
                break
            value = lefts[parent] + value
            depth -= 1


@jit(nopython=True, cache=True)
def _masses(
    reach,
    sorted_combo,
    group_starts,
    card_pos_flat,
    seg_end,
    ptr_worse,
    ptr_group,
    group,
    a_idx,
    b_idx,
    win,
    tie,
    alive,
):
    """``RunoutEvaluator.masses`` as loops: ~20 numpy calls on 1,326-vectors
    were almost all dispatch. Every sum keeps numpy's order (``_pairwise_sum``,
    sequential cumsum and bincount), so the values are the same bytes."""
    n = sorted_combo.shape[0]
    num_groups = group_starts.shape[0]

    w_sorted = np.empty(n, dtype=np.float64)
    for pos in range(n):
        w_sorted[pos] = reach[sorted_combo[pos]]

    # np.add.reduceat: the segment's first element plus the pairwise sum of the rest.
    group_sums = np.empty(num_groups, dtype=np.float64)
    for g in range(num_groups):
        lo = group_starts[g]
        hi = group_starts[g + 1] if g + 1 < num_groups else n
        group_sums[g] = w_sorted[lo] + _pairwise_sum(w_sorted, lo + 1, hi - lo - 1)

    # Mass in groups strictly after g: a sequential suffix sum, sentinel 0.
    suffix = np.zeros(num_groups + 1, dtype=np.float64)
    acc = 0.0
    for g in range(num_groups - 1, -1, -1):
        acc = acc + group_sums[g] if g < num_groups - 1 else group_sums[g]
        suffix[g] = acc
    total = suffix[0]

    m = card_pos_flat.shape[0]
    global_suffix = np.zeros(m + 1, dtype=np.float64)
    acc = 0.0
    for k in range(m - 1, -1, -1):
        value = w_sorted[card_pos_flat[k]]
        acc = acc + value if k < m - 1 else value
        global_suffix[k] = acc

    # np.bincount(a) + np.bincount(b): two sequential scatters, then one add.
    per_card_a = np.zeros(52, dtype=np.float64)
    per_card_b = np.zeros(52, dtype=np.float64)
    for pos in range(n):
        per_card_a[a_idx[pos]] += w_sorted[pos]
    for pos in range(n):
        per_card_b[b_idx[pos]] += w_sorted[pos]
    per_card = np.empty(52, dtype=np.float64)
    for k in range(52):
        per_card[k] = per_card_a[k] + per_card_b[k]

    for pos in range(n):
        combo = sorted_combo[pos]
        end_a = global_suffix[seg_end[pos, 0]]
        end_b = global_suffix[seg_end[pos, 1]]
        pw_a = global_suffix[ptr_worse[pos, 0]] - end_a
        pw_b = global_suffix[ptr_worse[pos, 1]] - end_b
        pg_a = global_suffix[ptr_group[pos, 0]] - end_a
        pg_b = global_suffix[ptr_group[pos, 1]] - end_b
        g = group[pos]
        self_mass = w_sorted[pos]
        # Card-a/b mass within the worse set (no both-cards combo can be there).
        w = suffix[g + 1] - pw_a - pw_b
        # Blocked in-group mass = a + b - w[h] (`h` is the only both-cards combo
        # and is double-counted); excluding the blocked set already excludes h.
        t = group_sums[g] - (pg_a - pw_a) - (pg_b - pw_b) + self_mass
        a = total - per_card[a_idx[pos]] - per_card[b_idx[pos]] + self_mass
        win[combo] = w if w >= 0.0 else 0.0
        tie[combo] = t if t >= 0.0 else 0.0
        alive[combo] = a if a >= 0.0 else 0.0


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
    rng: np.random.Generator | None = None,
    continuation: Continuation = CHECK_DOWN,
    root_prior: np.ndarray | None = None,
    root_prior_weight: float = 0.0,
) -> SubgameSolution:
    """Run RM+ CFR over the local tree; both players adapt per combo.

    Leaves are valued call-then-check-down on ``num_runouts`` sampled boards
    (exact single board when the root is already on the river). Iterates until
    the wall-clock budget runs out (at least ``_MIN_ITERATIONS``); pass
    ``max_iterations`` to pin the iteration count instead — results become
    machine/load-independent (the wall clock is ignored entirely).

    ``root_prior`` is a ``(NUM_COMBOS, n_root_actions)`` blueprint strategy
    seeded into the root's ``strategy_sum`` as a pseudo-count worth
    ``root_prior_weight`` iterations. Regrets start at zero, so an unseeded
    truncated solve returns something close to UNIFORM rather than close to the
    blueprint — which at the deployed budget is what made the resolver play
    uniform-random over {check, three bet sizes, all-in}. The prior makes a
    starved solve degrade toward the blueprint instead.
    """
    root = tree.root
    if not root.actions:
        raise ValueError("Subgame tree has no root actions.")

    evaluators = _sample_runout_evaluators(root.state, num_runouts, rng)
    # Per-combo count of runouts where the combo is alive (for averaging).
    alive_count = np.zeros(NUM_COMBOS, dtype=np.float64)
    for evaluator in evaluators:
        alive_count[evaluator.alive] += 1.0

    node_data: dict[int, _NodeData] = {}
    leaf_specs: dict[int, _LeafSpec] = {}
    _prepare_nodes(root, hero, rules, node_data, leaf_specs)

    reach_hero = hero_range.astype(np.float64)
    if root_prior is not None and root_prior_weight > 0.0:
        expected = (NUM_COMBOS, len(root.actions))
        if root_prior.shape != expected:
            raise ValueError(
                f"root_prior has shape {root_prior.shape}, expected {expected} "
                "(one row per combo, one column per root action)."
            )
        # Same units the loop accumulates in (`reach_actor * strategy` per
        # iteration), so the weight really is "worth this many iterations".
        node_data[id(root)].strategy_sum += (
            float(root_prior_weight) * reach_hero[:, None] * root_prior
        )
    reach_opp = opponent_range.astype(np.float64)
    ctx = _PassContext(
        hero=hero,
        evaluators=evaluators,
        alive_count=alive_count,
        node_data=node_data,
        leaf_specs=leaf_specs,
        continuation=continuation,
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

    return SubgameSolution(
        root_strategy=_normalize_or_uniform(node_data[id(root)].strategy_sum),
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
    continuation: Continuation = CHECK_DOWN


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


def _sample_runout_evaluators(
    state: GameState, num_runouts: int, rng: np.random.Generator | None = None
) -> list[RunoutEvaluator]:
    """Evaluators on completed boards; exact when the board is already complete."""
    board = state.board
    if len(board) == 5:
        return [RunoutEvaluator(board)]

    if rng is None:
        rng = np.random.default_rng()
    board_mask = 0
    for card in board:
        board_mask |= card.mask
    unseen = [card for card in FULL_DECK if not (card.mask & board_mask)]
    missing = 5 - len(board)

    evaluators = []
    for _ in range(max(1, num_runouts)):
        picks = rng.choice(len(unseen), size=missing, replace=False)
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
        v_hero, v_opp = _leaf_values(
            ctx.leaf_specs[id(node)], ctx, reach_hero, reach_opp, ctx.continuation
        )
        return v_hero, v_opp, None

    nd = ctx.node_data[id(node)]
    actor_is_hero = node.state.current_player == ctx.hero
    reach_actor = reach_hero if actor_is_hero else reach_opp

    # RM+ invariant: `nd.regrets` is clipped at 0 in place after every update,
    # so normalizing the raw rows IS regret matching here.
    strategy = _normalize_or_uniform(nd.regrets)
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


def _normalize_or_uniform(rows: np.ndarray) -> np.ndarray:
    """Normalize each (combos x actions) row to a distribution, uniform where empty.

    The resolver's one normalization: RM+ regret matching (rows are already
    clipped at 0) and average-strategy extraction are the same operation on
    nonnegative rows. Intentionally distinct from the 1-D numba training
    kernels in ``numba_ops`` (exact ``sum > 0`` semantics, hot path) — see
    the note on ``regret_matching`` there.
    """
    totals = rows.sum(axis=1, keepdims=True)
    uniform = np.full(rows.shape[1], 1.0 / rows.shape[1])
    return np.where(totals > NORMALIZE_EPS, rows / np.maximum(totals, NORMALIZE_EPS), uniform)


def _leaf_values(
    spec: _LeafSpec,
    ctx: _PassContext,
    reach_hero: np.ndarray,
    reach_opp: np.ndarray,
    continuation: Continuation = CHECK_DOWN,
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual value vectors at a leaf (terminal or depth-limit).

    ``continuation`` says what is ASSUMED to happen between this leaf and
    showdown. The default reproduces the check-down valuation exactly, so this
    parameter changes nothing until a caller passes something else.
    """
    # Fold: pot goes to the non-folder, cards never matter — so the alive mass
    # is taken against the ROOT board (embedded in the reach vectors), not
    # against any sampled runout.
    #
    # A continuation does NOT apply here, and that is not an oversight: the hand
    # is over, there is no rest-of-hand to assume anything about, and this branch
    # is EXACT. Scaling it would replace an exact answer with a modelled one.
    if spec.is_fold:
        return (
            spec.hero_payoff * nonblocking_mass(reach_opp),
            spec.opp_payoff * nonblocking_mass(reach_hero),
        )

    hero, opp = ctx.hero, 1 - ctx.hero
    # Both players are assumed to put in the same extra amount and see the
    # showdown, so the pot grows by twice what each commits. Symmetric on
    # purpose: an asymmetric continuation would encode a read, and everything
    # here must stay a function of public state + ranges.
    extra = continuation.pot_fraction * spec.pot
    pot = spec.pot + 2.0 * extra
    invested = (spec.invested[0] + extra, spec.invested[1] + extra)
    v_hero = np.zeros(NUM_COMBOS)
    v_opp = np.zeros(NUM_COMBOS)
    for evaluator in ctx.evaluators:
        win_h, tie_h, alive_h = evaluator.masses(reach_opp)
        v_hero += win_h * pot + tie_h * (pot / 2.0) - invested[hero] * alive_h

        win_o, tie_o, alive_o = evaluator.masses(reach_hero)
        v_opp += win_o * pot + tie_o * (pot / 2.0) - invested[opp] * alive_o

    count = ctx.alive_count
    np.divide(v_hero, count, out=v_hero, where=count > 0)
    np.divide(v_opp, count, out=v_opp, where=count > 0)
    return v_hero, v_opp


def nonblocking_mass(reach: np.ndarray) -> np.ndarray:
    """Per-combo total reach mass of combos not sharing a card (inclusion-exclusion)."""
    return _nonblocking_mass(np.ascontiguousarray(reach, dtype=np.float64), _CARD_A, _CARD_B)


@jit(nopython=True, cache=True)
def _nonblocking_mass(reach, card_a, card_b):
    n = reach.shape[0]
    per_card_a = np.zeros(52, dtype=np.float64)
    per_card_b = np.zeros(52, dtype=np.float64)
    for i in range(n):
        per_card_a[card_a[i]] += reach[i]
    for i in range(n):
        per_card_b[card_b[i]] += reach[i]
    total = _pairwise_sum(reach, 0, n)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (
            total
            - (per_card_a[card_a[i]] + per_card_b[card_a[i]])
            - (per_card_a[card_b[i]] + per_card_b[card_b[i]])
            + reach[i]
        )
    return out


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
