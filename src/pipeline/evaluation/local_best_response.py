"""Local Best Response (LBR) for extensive-form games.

LBR (Lisý & Bowling 2017) is a *local* best response: at each of its decision
points it picks the action that looks best assuming a fixed, cheap continuation
afterward, rather than solving the whole subtree optimally. Because the
resulting strategy is a concrete strategy, its value against the opponent is a
**lower bound** on true exploitability — but a far more informative one than a
one-ply rollout, and famously enough to expose "near-optimal" bots as highly
exploitable.

This generic version validates the LBR machinery on small games where the exact
best response (see :mod:`best_response`) is computable, giving the bridge
property ``LBR <= exact_BR``. The HUNL variant reuses the same shape — belief
over the opponent's range, a limited menu of actions (including off-tree bet
sizes), and a fixed continuation — but runs on the search-layer machinery
because HUNL cannot implement this explicit game protocol.

The continuation policy used for lookahead defaults to the opponent's own
strategy ``policy``; callers can pass a different one (e.g. a passive
"always check/call then roll out" policy, as standard poker LBR does).

References:
- Lisý & Bowling, "Equilibrium Approximation Quality of Current No-Limit Poker
  Bots" (2017)
"""

from __future__ import annotations

from src.pipeline.evaluation.best_response import on_policy_value
from src.pipeline.evaluation.game_tree import (
    CHANCE,
    ActionT,
    ExtensiveGame,
    InfoKey,
    Policy,
    StateT,
    collect_infoset_states,
)


def local_best_response_value(
    game: ExtensiveGame[StateT, ActionT],
    lbr_player: int,
    policy: Policy,
    continuation: Policy | None = None,
) -> float:
    """Return the value to ``lbr_player`` of playing LBR against ``policy``.

    At each ``lbr_player`` information set, LBR commits to the action maximizing
    the expected value under the ``continuation`` policy; it then re-decides at
    the next information set. The returned value is the realized value of that
    strategy against ``policy``, which is a lower bound on
    :func:`~src.pipeline.evaluation.best_response.best_response_value`.
    """
    cont = continuation if continuation is not None else policy

    def _actor_strategy(state: StateT, actor: int, use_cont: bool) -> tuple[list, list[float]]:
        legal = list(game.legal_actions(state))
        key = game.information_state_key(state, actor)
        source = cont if (use_cont and actor == lbr_player) else policy
        return legal, list(source(key, legal))

    # Continuation value: value to lbr_player when lbr plays `cont` and the
    # opponent plays `policy`. Used only to score candidate actions.
    cont_cache: dict[StateT, float] = {}

    def cont_value(state: StateT) -> float:
        cached = cont_cache.get(state)
        if cached is not None:
            return cached
        if game.is_terminal(state):
            result = float(game.returns(state)[lbr_player])
        else:
            player = game.current_player(state)
            if player == CHANCE:
                result = sum(
                    prob * cont_value(game.next_state(state, action))
                    for action, prob in game.chance_outcomes(state)
                )
            else:
                legal, probs = _actor_strategy(state, player, use_cont=True)
                result = sum(
                    prob * cont_value(game.next_state(state, action))
                    for action, prob in zip(legal, probs)
                    if prob > 0.0
                )
        cont_cache[state] = result
        return result

    # Group lbr decision states by information set with counterfactual reach.
    infoset_states = collect_infoset_states(game, lbr_player, policy)

    # Myopic action per information set: argmax over cf-weighted continuation value.
    lbr_action: dict[InfoKey, ActionT] = {}
    for key, states in infoset_states.items():
        legal = game.legal_actions(states[0][0])
        best_action = legal[0]
        best_value = float("-inf")
        for action in legal:
            total = sum(cf * cont_value(game.next_state(s, action)) for s, cf in states)
            if total > best_value:
                best_value = total
                best_action = action
        lbr_action[key] = best_action

    # Realized value: lbr plays its myopic choice at every node, opponent plays policy.
    real_cache: dict[StateT, float] = {}

    def real_value(state: StateT) -> float:
        cached = real_cache.get(state)
        if cached is not None:
            return cached
        if game.is_terminal(state):
            result = float(game.returns(state)[lbr_player])
        else:
            player = game.current_player(state)
            if player == CHANCE:
                result = sum(
                    prob * real_value(game.next_state(state, action))
                    for action, prob in game.chance_outcomes(state)
                )
            elif player == lbr_player:
                key = game.information_state_key(state, lbr_player)
                action = lbr_action.get(key)
                if action is None:
                    # Unreachable under the opponent's play; continuation is fine.
                    legal, probs = _actor_strategy(state, player, use_cont=True)
                    result = sum(
                        prob * real_value(game.next_state(state, a))
                        for a, prob in zip(legal, probs)
                        if prob > 0.0
                    )
                else:
                    result = real_value(game.next_state(state, action))
            else:
                legal, probs = _actor_strategy(state, player, use_cont=False)
                result = sum(
                    prob * real_value(game.next_state(state, action))
                    for action, prob in zip(legal, probs)
                    if prob > 0.0
                )
        real_cache[state] = result
        return result

    return real_value(game.initial_state())


def local_exploitability(
    game: ExtensiveGame[StateT, ActionT],
    policy: Policy,
    continuation: Policy | None = None,
) -> float:
    """Return the LBR-based exploitability lower bound (NashConv per player).

    Non-negative, and never exceeds the exact
    :func:`~src.pipeline.evaluation.best_response.exploitability`.
    """
    nash_conv = 0.0
    for player in range(game.num_players):
        nash_conv += local_best_response_value(game, player, policy, continuation)
        nash_conv -= on_policy_value(game, player, policy)
    return nash_conv / game.num_players
