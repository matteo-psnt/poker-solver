"""Local Best Response (LBR) for the trained HUNL blueprint.

Exact best response is intractable for full HUNL, so we measure a trustworthy
*lower bound* on exploitability with Local Best Response (Lisý & Bowling 2017).
The LBR player plays a concrete strategy against the frozen blueprint: at each of
its decisions it picks, over a menu of actions, the one maximizing a cheap
*myopic* value that assumes it then checks/calls to showdown. Because that is a
realizable strategy, its **realized** value against the blueprint is a lower
bound on the blueprint's true exploitability — the same property validated
exactly on Kuhn/Leduc in :mod:`local_best_response`.

The LBR player re-decides at *every* node against the full, Bayes-updated
opponent range (not just its first action, as the one-ply rollout in
:mod:`exploitability` does), which already makes it a far more informative
lower bound.

Off-tree bets (opt-in, ``include_off_tree``)
--------------------------------------------
LBR's menu can also include **off-tree bet sizes** the blueprint never trained
on; a real opponent betting an unseen size is what exposes the action
abstraction. This is **off by default** and here is why. Infoset keys encode the
*full-hand* betting sequence (see
:func:`~src.core.game.state.GameState.normalized_betting_sequence` — history is
cumulative across streets), and off-tree amounts normalize to unseen pot
fractions (``b0.63``). The *immediate* response to an off-tree bet is handled
correctly by translation, but the off-tree amount stays baked into the history,
so on *later* streets the opponent's infoset misses and falls back to
uniform-random play. That over-states the opponent's foolishness on multi-street
off-tree lines, biasing the number on the very path that must stay exact. With
``include_off_tree=False`` the LBR player uses only on-tree actions, the opponent
never leaves the blueprint's tree, and the result is a fully rigorous lower
bound — hence the default. Making off-tree rigorous requires carrying a parallel
translated (abstract-game) state for opponent lookups; that is a separate
milestone.

Opponent model / what this number means
----------------------------------------
Exploitability is defined against a *complete* strategy, but training only
defines the blueprint on-tree. This evaluator completes it with the standard
**translation policy**: when the LBR player makes an off-tree bet, the blueprint
responds as if the bet had been the nearest on-tree size (see
:mod:`src.engine.search.action_translation`). The reported figure is therefore
the *LBR exploitability of the blueprint strategy under translation*. The
blueprint is normally deployed through the runtime resolver (``resolver.enabled``
defaults to ``True``), whose off-tree response is a re-solve rather than a
translation; the translation completion is a documented, cheap proxy for that
deployed system and the two can differ in either direction.

Correctness note: the myopic pot-arithmetic value is a *scoring function* used
only to choose the LBR action. The returned mbb/g figure is the **realized**
payoff of playing those chosen actions out against the blueprint to true
terminals (including the blueprint's own future bets). Approximations in the
scorer only loosen the bound; they never invalidate it.

References:
- Lisý & Bowling, "Equilibrium Approximation Quality of Current No-Limit Poker
  Bots" (2017).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.evaluator import get_evaluator
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState
from src.engine.search.action_translation import translate_action_distribution
from src.engine.search.range_inference import ALL_COMBOS, NUM_COMBOS
from src.engine.solver.infoset_encoder import encode_infoset_key

_EPS = 1e-12

# Canonical 52-card deck; runouts and deals draw from it by index/mask.
_DECK: list[Card] = Card.get_full_deck()
_DECK_MASKS: np.ndarray = np.array([card.mask for card in _DECK], dtype=np.int64)

# Off-tree bet sizes (as a fraction of the pot) the LBR player may bet when it is
# first to put money in on a street. These are deliberately not the blueprint's
# trained sizes; overbets in particular probe the action abstraction.
DEFAULT_OFF_TREE_POT_FRACTIONS: tuple[float, ...] = (0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)


@dataclass
class LBRConfig:
    """Settings for the HUNL LBR evaluator."""

    num_hands: int = 2000
    equity_runouts: int = 24
    off_tree_pot_fractions: tuple[float, ...] = DEFAULT_OFF_TREE_POT_FRACTIONS
    # Off-tree bet sizes are OFF by default: see the "Off-tree bets" note in the
    # module docstring — with the full-hand infoset encoding they leak into a
    # uniform-random opponent on later streets, so the leak-free on-tree menu is
    # the trustworthy default. Enable only for exploratory action-abstraction probing.
    include_off_tree: bool = False
    seed: int | None = None


@dataclass
class LBRResult:
    """LBR exploitability estimate with statistical uncertainty."""

    exploitability_mbb: float
    exploitability_bb: float
    lbr_utility_p0: float
    lbr_utility_p1: float
    std_error_mbb: float
    confidence_95_mbb: tuple[float, float]
    num_hands: int


class _HUNLLocalBestResponse:
    """Plays LBR against a frozen blueprint and measures realized value.

    One instance owns a shared hand evaluator and per-call caches; it is not
    thread-safe. Callers should use :func:`compute_lbr_exploitability`.
    """

    def __init__(self, blueprint, config: LBRConfig, rng: np.random.Generator):
        self.blueprint = blueprint
        self.config = config
        self.rng = rng
        self.rules: GameRules = blueprint.rules
        self.action_model: ActionModel = blueprint.action_model
        self.card_abstraction = blueprint.card_abstraction
        self.storage = blueprint.storage
        self.evaluator = get_evaluator()
        self._masks = np.array([c1.mask | c2.mask for c1, c2 in ALL_COMBOS], dtype=np.int64)

    # -- Realized play -----------------------------------------------------

    def play_hand(self, lbr_player: int, initial_state: GameState) -> float:
        """Play one hand with ``lbr_player`` using LBR; return its realized payoff.

        The opponent is carried as a **range** (``belief``), never a sampled
        hand: its actions are drawn from the range-aggregate distribution and
        Bayes-update the range, and terminals are valued analytically against the
        surviving range. Only public chance (the board) and the LBR player's own
        hand are sampled, which removes the dominant opponent-hand variance.
        """
        opp = 1 - lbr_player
        state = initial_state
        belief = self._initial_belief(initial_state, opp)

        prev_state: GameState | None = None
        prev_lbr_action: Action | None = None

        while not state.is_terminal:
            if self.blueprint.is_chance_node(state):
                state = self.blueprint.sample_chance_outcome(state)
                prev_state, prev_lbr_action = None, None
                continue

            if state.current_player == lbr_player:
                action = self._choose_lbr_action(state, lbr_player, belief)
                prev_state, prev_lbr_action = state, action
            else:
                legal, vecs = self._opp_action_matrix(state, opp, prev_state, prev_lbr_action)
                action, belief = self._sample_opponent(legal, vecs, belief)

            state = state.apply_action(action, self.rules)

        return self._terminal_value(state, lbr_player, opp, belief)

    def _initial_belief(self, state: GameState, opp: int) -> np.ndarray:
        """Uniform opponent range vector masked by the LBR player's known cards."""
        known = self._known_mask(state, opp)
        weights = np.where((self._masks & known) == 0, 1.0, 0.0)
        total = weights.sum()
        if total <= _EPS:
            return np.full(NUM_COMBOS, 1.0 / NUM_COMBOS)
        return weights / total

    def _known_mask(self, state: GameState, opp: int) -> int:
        """Bitmask of cards the LBR player can see (its holes + board)."""
        lbr = 1 - opp
        mask = 0
        for card in state.hole_cards[lbr]:
            mask |= card.mask
        for card in state.board:
            mask |= card.mask
        return mask

    def _terminal_value(
        self, state: GameState, lbr_player: int, opp: int, belief: np.ndarray
    ) -> float:
        """Realized LBR payoff at a terminal, valued against the surviving range.

        A fold ends the hand independently of the cards, so the pot-based payoff
        is exact. A showdown is valued as the belief-weighted payoff over every
        opponent combo still in range, with the board rolled out if the hand went
        all-in early — integrating out the opponent hand rather than sampling it.
        """
        if not self._is_showdown(state):
            return float(state.get_payoff(lbr_player, self.rules))

        if len(state.board) < 5:
            state = self.blueprint.deal_remaining_cards(state)

        known = self._known_mask(state, opp)
        weights = np.where((self._masks & known) == 0, belief, 0.0)
        total = weights.sum()
        if total <= _EPS:
            return float(state.get_payoff(lbr_player, self.rules))

        ev = 0.0
        for idx in np.nonzero(weights)[0]:
            opp_state = self._with_opponent_hole(state, opp, ALL_COMBOS[idx])
            ev += float(weights[idx]) * float(opp_state.get_payoff(lbr_player, self.rules))
        return ev / float(total)

    # -- LBR action choice -------------------------------------------------

    def _choose_lbr_action(self, state: GameState, lbr_player: int, belief: np.ndarray) -> Action:
        opp = 1 - lbr_player
        opp_weights = belief
        lbr_hand = state.hole_cards[lbr_player]
        candidates = self._action_menu(state)
        best_action = candidates[0]
        best_value = float("-inf")
        for action in candidates:
            value = self._score_action(state, lbr_player, opp, lbr_hand, opp_weights, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _action_menu(self, state: GameState) -> list[Action]:
        """On-tree legal actions plus off-tree bet sizes when leading."""
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        if not self.config.include_off_tree or state.to_call > 0:
            return legal

        seen = {(a.type, a.amount) for a in legal}
        pot = state.pot
        effective_stack = min(state.stacks)
        for frac in self.config.off_tree_pot_fractions:
            amount = round(frac * pot)
            if amount <= 0 or amount > effective_stack:
                continue
            action = Action(ActionType.BET, amount)
            if (action.type, action.amount) in seen:
                continue
            if self.rules.is_action_valid(state, action):
                legal.append(action)
                seen.add((action.type, action.amount))
        return legal

    def _score_action(
        self,
        state: GameState,
        lbr_player: int,
        opp: int,
        lbr_hand: tuple[Card, Card],
        opp_weights: np.ndarray,
        action: Action,
    ) -> float:
        """Myopic (check/call-to-showdown) value of ``action``; selection only."""
        pot = float(state.pot)
        if action.type == ActionType.FOLD:
            return 0.0

        if action.type in (ActionType.CHECK, ActionType.CALL):
            wp = self._equity(lbr_hand, state.board, opp_weights)
            to_call = float(state.to_call)
            # Check: no chips added. Call: risk `to_call` to contest the pot.
            return wp * pot - (1.0 - wp) * to_call

        # Aggressive action (bet/raise/all-in): fold equity + called equity.
        size = self._chips_committed(state, action)
        fold_probs = self._opp_fold_probs(state, opp, action)
        fold_equity = float(np.dot(opp_weights, fold_probs))
        weight_sum = float(opp_weights.sum())
        fp = fold_equity / weight_sum if weight_sum > _EPS else 0.0

        continue_weights = opp_weights * (1.0 - fold_probs)
        wp = self._equity(lbr_hand, state.board, continue_weights)
        called_value = wp * (pot + size) - (1.0 - wp) * size
        return fp * pot + (1.0 - fp) * called_value

    def _chips_committed(self, state: GameState, action: Action) -> float:
        """Chips the LBR player puts in beyond any amount already matched."""
        if action.type == ActionType.BET:
            return float(action.amount)
        if action.type == ActionType.RAISE:
            return float(state.to_call + action.amount)
        if action.type == ActionType.ALL_IN:
            return float(action.amount)
        return 0.0

    # -- Opponent model (translation completion) ---------------------------

    def _opponent_distribution(
        self,
        state: GameState,
        opp: int,
        prev_state: GameState | None,
        prev_lbr_action: Action | None,
    ) -> dict[Action, float]:
        """Blueprint response distribution over legal actions at ``state``.

        On-tree nodes read the blueprint infoset directly. Off-tree nodes (the
        LBR player made an off-tree bet) map the bet to the nearest on-tree size
        and read the blueprint there, projecting the response back onto the real
        legal actions — the translation completion described in the module docs.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        if not legal:
            return {}

        direct = self._infoset_distribution(state, opp, legal)
        if direct is not None:
            return direct

        if prev_state is not None and prev_lbr_action is not None:
            translated = self._translated_distribution(
                state, opp, legal, prev_state, prev_lbr_action
            )
            if translated:
                return translated

        uniform = 1.0 / len(legal)
        return {action: uniform for action in legal}

    def _infoset_distribution(
        self, state: GameState, player: int, legal: list[Action]
    ) -> dict[Action, float] | None:
        """Blueprint strategy over ``legal`` at ``state``, or ``None`` if off-tree."""
        infoset_key = encode_infoset_key(state, player, self.card_abstraction)
        infoset = self.storage.get_infoset(infoset_key)
        if infoset is None:
            return None

        legal_set = set(legal)
        valid_indices: list[int] = []
        valid_actions: list[Action] = []
        for idx, action in enumerate(infoset.legal_actions):
            if action in legal_set and self.rules.is_action_valid(state, action):
                valid_indices.append(idx)
                valid_actions.append(action)
        if not valid_indices:
            return None

        strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=True)
        dist: dict[Action, float] = {}
        for action, prob in zip(valid_actions, strategy):
            dist[action] = dist.get(action, 0.0) + float(prob)
        return dist

    def _translated_distribution(
        self,
        state: GameState,
        opp: int,
        legal: list[Action],
        prev_state: GameState,
        prev_lbr_action: Action,
    ) -> dict[Action, float]:
        """Project the blueprint's on-tree response onto real legal actions."""
        on_tree = translate_action_distribution(
            prev_state, prev_lbr_action, self.action_model, self.rules
        )
        dist: dict[Action, float] = {}
        for proxy_action, weight in on_tree:
            proxy_state = prev_state.apply_action(proxy_action, self.rules)
            proxy_legal = list(
                self.rules.get_legal_actions(proxy_state, action_model=self.action_model)
            )
            proxy_dist = self._infoset_distribution(proxy_state, opp, proxy_legal)
            if proxy_dist is None:
                continue
            for action, prob in proxy_dist.items():
                for real_action, share in self._map_to_real(state, action, legal):
                    dist[real_action] = dist.get(real_action, 0.0) + weight * prob * share

        total = sum(dist.values())
        if total <= _EPS:
            return {}
        return {action: prob / total for action, prob in dist.items()}

    def _map_to_real(
        self, state: GameState, action: Action, legal: list[Action]
    ) -> list[tuple[Action, float]]:
        """Map a proxy-node response to the real legal actions at ``state``."""
        if action in legal:
            return [(action, 1.0)]
        # Fold/check/call carry across states unchanged; only bet/raise sizes
        # need re-discretizing to the real legal menu.
        return translate_action_distribution(state, action, self.action_model, self.rules)

    def _opp_action_matrix(
        self,
        state: GameState,
        opp: int,
        prev_state: GameState | None,
        prev_lbr_action: Action | None,
    ) -> tuple[list[Action], dict[Action, np.ndarray]]:
        """Per-combo blueprint action probabilities at an opponent node.

        Returns the legal actions and, for each, a length-``NUM_COMBOS`` vector
        giving the probability the opponent takes it holding each combo. The
        opponent's bucket (hence its whole distribution) is action-independent
        per combo, so it is computed once and cached by infoset key across all
        combos that share a bucket.
        """
        legal = list(self.rules.get_legal_actions(state, action_model=self.action_model))
        vecs: dict[Action, np.ndarray] = {action: np.zeros(NUM_COMBOS) for action in legal}
        if not legal:
            return legal, vecs

        known = self._known_mask(state, opp)
        cache: dict[object, dict[Action, float]] = {}
        for idx in range(NUM_COMBOS):
            if self._masks[idx] & known:
                continue
            opp_state = self._with_opponent_hole(state, opp, ALL_COMBOS[idx])
            key = encode_infoset_key(opp_state, opp, self.card_abstraction)
            dist = cache.get(key)
            if dist is None:
                dist = self._opponent_distribution(opp_state, opp, prev_state, prev_lbr_action)
                cache[key] = dist
            for action, prob in dist.items():
                vec = vecs.get(action)
                if vec is not None:
                    vec[idx] += prob
        return legal, vecs

    def _sample_opponent(
        self, legal: list[Action], vecs: dict[Action, np.ndarray], belief: np.ndarray
    ) -> tuple[Action, np.ndarray]:
        """Sample an opponent action from the range aggregate and Bayes-update."""
        aggregate = np.array([float(np.dot(belief, vecs[action])) for action in legal])
        total = aggregate.sum()
        if total <= _EPS:
            return legal[int(self.rng.integers(0, len(legal)))], belief
        aggregate /= total
        choice = int(self.rng.choice(len(legal), p=aggregate))
        action = legal[choice]
        posterior = belief * vecs[action]
        mass = posterior.sum()
        return action, (posterior / mass if mass > _EPS else belief)

    def _opp_fold_probs(self, state: GameState, opp: int, action: Action) -> np.ndarray:
        """Per-combo probability the opponent folds to ``action`` (scorer input)."""
        resp_state = state.apply_action(action, self.rules)
        legal, vecs = self._opp_action_matrix(resp_state, opp, state, action)
        fold_probs = np.zeros(NUM_COMBOS)
        for candidate in legal:
            if candidate.type == ActionType.FOLD:
                fold_probs += vecs[candidate]
        return fold_probs

    def _with_opponent_hole(
        self, state: GameState, opp: int, combo: tuple[Card, Card]
    ) -> GameState:
        holes = list(state.hole_cards)
        holes[opp] = combo
        return GameState(
            street=state.street,
            pot=state.pot,
            stacks=state.stacks,
            board=state.board,
            hole_cards=(holes[0], holes[1]),
            betting_history=state.betting_history,
            button_position=state.button_position,
            current_player=state.current_player,
            is_terminal=state.is_terminal,
            to_call=state.to_call,
            last_aggressor=state.last_aggressor,
        )

    # -- Equity ------------------------------------------------------------

    def _equity(
        self, lbr_hand: tuple[Card, Card], board: tuple[Card, ...], opp_weights: np.ndarray
    ) -> float:
        """Weighted win probability of ``lbr_hand`` vs the opponent range.

        Board cards are rolled out to the river when incomplete. Scorer-only, so
        Monte-Carlo runout noise is acceptable — it only loosens the bound.
        """
        known = 0
        for card in lbr_hand:
            known |= card.mask
        for card in board:
            known |= card.mask
        active = np.where((opp_weights > _EPS) & ((self._masks & known) == 0))[0]
        if active.size == 0:
            return 0.5

        if len(board) >= 5:
            return self._showdown_equity(lbr_hand, board, opp_weights, active)

        runouts = max(1, self.config.equity_runouts)
        total = 0.0
        for _ in range(runouts):
            full_board = self._complete_board(board, known)
            total += self._showdown_equity(lbr_hand, full_board, opp_weights, active)
        return total / runouts

    def _showdown_equity(
        self,
        lbr_hand: tuple[Card, Card],
        board: tuple[Card, ...],
        opp_weights: np.ndarray,
        active: np.ndarray,
    ) -> float:
        board_mask = 0
        for card in board:
            board_mask |= card.mask
        lbr_rank = self.evaluator.evaluate(lbr_hand, board)
        acc = 0.0
        weight = 0.0
        for idx in active:
            if self._masks[idx] & board_mask:
                continue
            w = opp_weights[idx]
            opp_rank = self.evaluator.evaluate(ALL_COMBOS[idx], board)
            if lbr_rank < opp_rank:
                acc += w
            elif lbr_rank == opp_rank:
                acc += 0.5 * w
            weight += w
        return acc / weight if weight > _EPS else 0.5

    def _complete_board(self, board: tuple[Card, ...], known: int) -> tuple[Card, ...]:
        needed = 5 - len(board)
        drawn: list[Card] = []
        used = known
        while len(drawn) < needed:
            idx = int(self.rng.integers(0, 52))
            mask = int(_DECK_MASKS[idx])
            if used & mask:
                continue
            used |= mask
            drawn.append(_DECK[idx])
        return tuple(board) + tuple(drawn)

    # -- Helpers -----------------------------------------------------------

    def _sample(self, dist: dict[Action, float]) -> Action:
        actions = list(dist.keys())
        probs = np.array([dist[a] for a in actions], dtype=np.float64)
        total = probs.sum()
        if total <= _EPS:
            return actions[int(self.rng.integers(0, len(actions)))]
        probs /= total
        return actions[int(self.rng.choice(len(actions), p=probs))]

    @staticmethod
    def _is_showdown(state: GameState) -> bool:
        if not state.betting_history:
            return False
        return state.betting_history[-1].type != ActionType.FOLD


def compute_lbr_exploitability(blueprint, config: LBRConfig | None = None) -> LBRResult:
    """Estimate the blueprint's exploitability via Local Best Response.

    Plays ``config.num_hands`` deals; on each, the LBR player exploits from both
    positions (paired sampling on the same deal to cut variance) and the sample
    exploitability is ``(u_p0 + u_p1) / 2``. This mirrors the convention in
    :func:`~src.pipeline.evaluation.exploitability.compute_exploitability`,
    which assumes button-symmetrized on-policy value is ~0, so the two figures
    are directly comparable — LBR should never report *less*.
    """
    if config is None:
        config = LBRConfig()
    if config.seed is not None:
        # The blueprint deals the board via the ``random`` module; seed it too so
        # the whole evaluation is reproducible, not just our own sampling.
        random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    engine = _HUNLLocalBestResponse(blueprint, config, rng)

    starting_stack = blueprint.config.game.starting_stack
    big_blind = blueprint.config.game.big_blind

    utilities_p0: list[float] = []
    utilities_p1: list[float] = []
    samples: list[float] = []
    for hand in range(config.num_hands):
        button = hand % 2
        state = _deal_initial_state(engine, starting_stack, button, rng)
        u0 = engine.play_hand(0, state)
        u1 = engine.play_hand(1, state)
        utilities_p0.append(u0)
        utilities_p1.append(u1)
        samples.append((u0 + u1) / 2.0)

    exploitability = float(np.mean(samples)) if samples else float("nan")
    if len(samples) >= 2:
        se = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    else:
        se = 0.0

    exploitability_mbb = (exploitability / big_blind) * 1000.0
    se_mbb = (se / big_blind) * 1000.0
    return LBRResult(
        exploitability_mbb=exploitability_mbb,
        exploitability_bb=exploitability / big_blind,
        lbr_utility_p0=float(np.mean(utilities_p0)) if utilities_p0 else float("nan"),
        lbr_utility_p1=float(np.mean(utilities_p1)) if utilities_p1 else float("nan"),
        std_error_mbb=se_mbb,
        confidence_95_mbb=(exploitability_mbb - 1.96 * se_mbb, exploitability_mbb + 1.96 * se_mbb),
        num_hands=len(samples),
    )


def _deal_initial_state(
    engine: _HUNLLocalBestResponse,
    starting_stack: int,
    button: int,
    rng: np.random.Generator,
) -> GameState:
    """Deal a fresh hand (both hole cards) using ``rng`` for reproducibility."""
    order = rng.permutation(52)
    cards = [_DECK[int(i)] for i in order[:4]]
    hole_cards = ((cards[0], cards[1]), (cards[2], cards[3]))
    return engine.rules.create_initial_state(
        starting_stack=starting_stack,
        hole_cards=hole_cards,
        button=button,
    )
