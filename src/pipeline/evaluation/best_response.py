"""Exact best response and exploitability for extensive-form games.

Unlike the rollout-based estimator in :mod:`exploitability`, this computes an
*exact*, information-set-consistent best response by full tree traversal. It is
only tractable for small games, so its role is to validate the evaluation
methodology against known equilibria (see the Kuhn poker tests) and to serve as
ground truth for abstracted subgames.

Correctness hinges on information-set consistency: a best responder must commit
to a single action per information set, aggregating counterfactual value across
all histories it cannot distinguish. A naive per-history maximum would let the
responder peek at hidden information and over-state exploitability (a converged
strategy would spuriously appear exploitable). The two-phase algorithm here
groups histories by information set and maximizes the aggregated counterfactual
value, following the standard best-response computation (e.g. OpenSpiel).

References:
- Johanson et al. "Accelerated Best Response Calculation in Large Extensive
  Games" (2011)
"""

from __future__ import annotations

from src.pipeline.evaluation.game_tree import (
    CHANCE,
    ActionT,
    ExtensiveGame,
    InfoKey,
    Policy,
    StateT,
    collect_infoset_states,
)


def best_response_value(
    game: ExtensiveGame[StateT, ActionT], br_player: int, policy: Policy
) -> float:
    """Return the expected value to ``br_player`` of a best response to ``policy``.

    ``policy`` supplies the strategy for every *other* player; ``br_player``
    plays an exact best response. Chance follows the game's own distribution.

    Args:
        game: The extensive-form game.
        br_player: The player computing a best response.
        policy: Behavioural strategy used for all non-``br_player`` decisions.

    Returns:
        The best-response expected utility to ``br_player`` at the root.
    """
    # Phase 1: group every br_player decision state by its information set, with
    # counterfactual reach (see :func:`collect_infoset_states`).
    infoset_states = collect_infoset_states(game, br_player, policy)

    # Phase 2: value of a state under best play by br_player and fixed policy for
    # opponents. Memoized on state; the best action per information set is cached
    # and reused across all indistinguishable histories.
    value_cache: dict[StateT, float] = {}
    br_action_cache: dict[InfoKey, ActionT] = {}

    def br_action(info_key: InfoKey, state: StateT) -> ActionT:
        cached = br_action_cache.get(info_key)
        if cached is not None:
            return cached
        legal = game.legal_actions(state)
        # States reachable with positive counterfactual probability. Unreachable
        # information sets (empty list) never affect the root value; any action
        # is fine, so the aggregated value stays 0 and the first action is taken.
        states = infoset_states.get(info_key, ())
        best_action = legal[0]
        best_value = float("-inf")
        for action in legal:
            total = sum(cf * value(game.next_state(s, action)) for s, cf in states)
            if total > best_value:
                best_value = total
                best_action = action
        br_action_cache[info_key] = best_action
        return best_action

    def value(state: StateT) -> float:
        cached = value_cache.get(state)
        if cached is not None:
            return cached
        if game.is_terminal(state):
            result = float(game.returns(state)[br_player])
        else:
            player = game.current_player(state)
            if player == CHANCE:
                # Chance probabilities are strictly positive by construction.
                result = sum(
                    prob * value(game.next_state(state, action))
                    for action, prob in game.chance_outcomes(state)
                )
            elif player == br_player:
                key = game.information_state_key(state, br_player)
                result = value(game.next_state(state, br_action(key, state)))
            else:
                legal = game.legal_actions(state)
                probs = policy(game.information_state_key(state, player), legal)
                # Skip zero-probability branches: they contribute nothing to the
                # expectation, and evaluating them would descend into subtrees the
                # opponent never reaches (whose best-response actions are moot).
                result = sum(
                    prob * value(game.next_state(state, action))
                    for action, prob in zip(legal, probs)
                    if prob > 0.0
                )
        value_cache[state] = result
        return result

    return value(game.initial_state())


def on_policy_value(game: ExtensiveGame[StateT, ActionT], player: int, policy: Policy) -> float:
    """Return the expected value to ``player`` when every player follows ``policy``."""
    value_cache: dict[StateT, float] = {}

    def value(state: StateT) -> float:
        cached = value_cache.get(state)
        if cached is not None:
            return cached
        if game.is_terminal(state):
            result = float(game.returns(state)[player])
        else:
            actor = game.current_player(state)
            if actor == CHANCE:
                result = sum(
                    prob * value(game.next_state(state, action))
                    for action, prob in game.chance_outcomes(state)
                )
            else:
                legal = game.legal_actions(state)
                probs = policy(game.information_state_key(state, actor), legal)
                result = sum(
                    prob * value(game.next_state(state, action))
                    for action, prob in zip(legal, probs)
                )
        value_cache[state] = result
        return result

    return value(game.initial_state())


def exploitability(game: ExtensiveGame[StateT, ActionT], policy: Policy) -> float:
    """Return NashConv per player: the average gain of a best responder.

    NashConv = sum_i [ BR_i(policy_{-i}) - u_i(policy) ], and exploitability is
    reported as NashConv / num_players. It is 0 iff ``policy`` is a Nash
    equilibrium, and non-negative for any strategy. For two-player zero-sum games
    the on-policy terms cancel, so this reduces to (BR_0 + BR_1) / 2.
    """
    nash_conv = 0.0
    for player in range(game.num_players):
        nash_conv += best_response_value(game, player, policy)
        nash_conv -= on_policy_value(game, player, policy)
    return nash_conv / game.num_players
