"""Generic extensive-form game protocol for exact evaluation.

This module defines a minimal, engine-agnostic interface for finite
extensive-form games with imperfect information. It intentionally has no
dependency on the HUNL engine: states and actions are opaque, hashable values.

The purpose is validation infrastructure. Exact best-response and exploitability
(see :mod:`best_response`) can be computed against *any* game implementing this
protocol, which lets us anchor the evaluation machinery on small games with
known equilibria (e.g. Kuhn poker) before trusting it on HUNL.

Conventions:
- Players are numbered ``0 .. num_players - 1``.
- Chance nodes report ``current_player == CHANCE``.
- ``initial_state`` may itself be a chance node (e.g. the deal).
- States must be hashable and immutable so results can be memoized safely.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Protocol, TypeVar

# Sentinel player id for chance nodes.
CHANCE = -1

# An opaque game state. Must be hashable and immutable.
State = Hashable
# An opaque action. Must be hashable.
Action = Hashable
# An information-set key from a given player's point of view.
InfoKey = Hashable

# A behavioural policy: given an information-set key and the legal actions at a
# node, return a probability distribution aligned with ``legal_actions`` order.
Policy = Callable[[InfoKey, Sequence[Action]], Sequence[float]]

# Games specialize these to their own concrete state/action types, so that
# implementations (and their consumers) get precise checking rather than the
# erased ``Hashable`` upper bound.
StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class ExtensiveGame(Protocol[StateT, ActionT]):
    """Finite extensive-form game with imperfect information.

    Generic in the concrete state and action types so that a specific game
    (e.g. ``ExtensiveGame[KuhnState, str]``) conforms structurally without
    widening its method parameters to ``Hashable``.
    """

    num_players: int

    def initial_state(self) -> StateT:
        """Return the root state (may be a chance node)."""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Return True if ``state`` is terminal."""
        ...

    def returns(self, state: StateT) -> Sequence[float]:
        """Return per-player utilities at a terminal ``state``."""
        ...

    def current_player(self, state: StateT) -> int:
        """Return the acting player id, or :data:`CHANCE` for chance nodes."""
        ...

    def chance_outcomes(self, state: StateT) -> Sequence[tuple[ActionT, float]]:
        """Return ``(action, probability)`` pairs at a chance ``state``.

        Probabilities must sum to 1.
        """
        ...

    def legal_actions(self, state: StateT) -> Sequence[ActionT]:
        """Return legal actions at a decision ``state``.

        All histories sharing an information set must return the same actions in
        the same order, so a policy defined on the information set is well-posed.
        """
        ...

    def next_state(self, state: StateT, action: ActionT) -> StateT:
        """Return the successor state after applying ``action``."""
        ...

    def information_state_key(self, state: StateT, player: int) -> InfoKey:
        """Return ``player``'s information-set key at ``state``.

        Two states that ``player`` cannot distinguish must map to equal keys.
        """
        ...


def collect_infoset_states[StateT: Hashable, ActionT: Hashable](
    game: ExtensiveGame[StateT, ActionT], player: int, policy: Policy
) -> dict[InfoKey, list[tuple[StateT, float]]]:
    """Group ``player``'s decision states by information set with counterfactual reach.

    The counterfactual reach of a state is the product of chance and opponent
    action probabilities along the path; ``player``'s own action probabilities
    are excluded, so every subtree below their decisions is explored at full
    weight. Zero-reach subtrees are pruned. This grouping is the shared first
    phase of both exact best response and LBR: a responder must aggregate value
    across all histories in an information set before choosing an action.
    """
    infoset_states: dict[InfoKey, list[tuple[StateT, float]]] = defaultdict(list)

    def collect(state: StateT, cf_reach: float) -> None:
        if cf_reach == 0.0 or game.is_terminal(state):
            return
        actor = game.current_player(state)
        if actor == CHANCE:
            for action, prob in game.chance_outcomes(state):
                collect(game.next_state(state, action), cf_reach * prob)
            return
        if actor == player:
            key = game.information_state_key(state, player)
            infoset_states[key].append((state, cf_reach))
            for action in game.legal_actions(state):
                # Counterfactual: do not weight by the player's own action prob.
                collect(game.next_state(state, action), cf_reach)
            return
        legal = game.legal_actions(state)
        probs = policy(game.information_state_key(state, actor), legal)
        for action, prob in zip(legal, probs):
            collect(game.next_state(state, action), cf_reach * prob)

    collect(game.initial_state(), 1.0)
    return infoset_states


class TabularStrategy:
    """A :data:`Policy` backed by an explicit ``info_key -> action -> prob`` map.

    Missing information sets or actions fall back to a uniform distribution over
    the legal actions, which keeps best-response evaluation well-defined for
    partially specified strategies.
    """

    def __init__(self, table: dict[InfoKey, dict[Action, float]] | None = None):
        self.table: dict[InfoKey, dict[Action, float]] = table or {}

    def __call__(self, info_key: InfoKey, legal_actions: Sequence[Action]) -> list[float]:
        entry = self.table.get(info_key)
        n = len(legal_actions)
        if not entry:
            return [1.0 / n] * n
        probs = [max(entry.get(action, 0.0), 0.0) for action in legal_actions]
        total = sum(probs)
        if total <= 0.0:
            return [1.0 / n] * n
        return [p / total for p in probs]

    def set(self, info_key: InfoKey, distribution: dict[Action, float]) -> None:
        """Set the action distribution for one information set."""
        self.table[info_key] = dict(distribution)
