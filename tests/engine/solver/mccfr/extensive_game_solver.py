"""Run the production MCCFR kernel on an arbitrary small extensive-form game.

The regret math, the update placement, the averaging scheme and the sampling
scheme in ``src/engine/solver/mccfr`` are game-agnostic; only the state machine,
the infoset encoding and the board-dealing above them are HUNL's. This module
supplies the HUNL-shaped pieces from an
:class:`~src.pipeline.evaluation.reference.game_tree.ExtensiveGame` instead, so the *real*
``MCCFRSolver`` -- not a reimplementation of it -- can be trained on Kuhn or
Leduc and scored against a known equilibrium.

That distinction is the whole point. ``src/pipeline/evaluation/tabular_cfr.py``
validates the evaluation harness with a second, independent CFR implementation;
nothing until now put the shipped kernel in front of a game whose answer we know.

What this exercises: ``apply_regret_updates`` (CFR+/DCFR weighting), the
regret-matching strategy, the average-strategy accumulation point, external and
outcome sampling, and the shared-array storage the traversal writes through.

What it deliberately does NOT exercise: ``infoset.encoder`` (169 preflop classes,
equity buckets, SPR), the action abstraction, and HUNL chance -- all three are
replaced here, so a passing test says nothing about them.

This module is a test helper (not named ``test_*``) so pytest does not collect it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from src.core.game.state import GameState, Street
from src.engine.solver.infoset.model import InfoSetKey
from src.engine.solver.mccfr import MCCFRSolver
from src.pipeline.evaluation.reference.game_tree import CHANCE, ExtensiveGame, InfoKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.actions.action_model import ActionModel
    from src.core.game.actions import Action
    from src.engine.solver.storage.base import KeyedStorage
    from src.shared.config import Config

# InfoSetKey demands a preflop hand string on PREFLOP; the adapted games carry
# their whole information state in ``betting_sequence`` instead, so this is an
# inert filler that keeps every adapted key on one street.
_HAND_PLACEHOLDER = "--"


def adapted_infoset_key(info_key: InfoKey) -> InfoSetKey:
    """Pack a game's information-state key into the solver's ``InfoSetKey``.

    ``player_position`` is left at 0 rather than carrying the acting player.
    Turn order is public in these games, so the information-state key already
    determines who is acting and adding the player would be redundant -- and
    leaving it out is what lets :func:`average_policy` rebuild the exact key from
    the evaluation harness's info key, which does not carry a player either.

    Injectivity, which is all the kernel needs from a key, comes from ``repr``
    over the game's own key tuple.
    """
    return InfoSetKey(
        player_position=0,
        street=Street.PREFLOP,
        betting_sequence=repr(info_key),
        preflop_hand=_HAND_PLACEHOLDER,
        postflop_bucket=None,
        spr_bucket=0,
    )


@dataclass(frozen=True, slots=True)
class AdaptedState:
    """An ``ExtensiveGame`` state wearing the shape the traversal reads.

    The traversal only ever touches ``is_terminal``, ``current_player``,
    ``apply_action`` and ``get_payoff`` on a state, so those four are the entire
    contract being met here.
    """

    game: Any
    inner: Any
    actions: ActionCodec

    @property
    def is_terminal(self) -> bool:
        return self.game.is_terminal(self.inner)

    @property
    def current_player(self) -> int:
        return self.game.current_player(self.inner)

    def apply_action(self, action: Action, rules: object | None = None) -> AdaptedState:
        inner = self.game.next_state(self.inner, self.actions.to_game[action])
        return AdaptedState(self.game, inner, self.actions)

    def get_payoff(self, player: int, rules: object | None = None) -> float:
        return float(self.game.returns(self.inner)[player])

    def __str__(self) -> str:
        return f"AdaptedState({self.inner})"


class ActionCodec:
    """Bidirectional map between a game's actions and core ``Action`` objects.

    The kernel stores ``Action`` objects as an infoset's action list and indexes
    regrets by their position in it, so the game's own action labels have to be
    carried across as stable, distinct ``Action`` values. Their ``type`` and
    ``amount`` are labels only -- nothing in the traversal interprets them.
    """

    def __init__(self, mapping: dict[Any, Action]):
        self.to_core: dict[Any, Action] = dict(mapping)
        self.to_game: dict[Action, Any] = {v: k for k, v in mapping.items()}
        if len(self.to_game) != len(self.to_core):
            raise ValueError(f"Action mapping is not injective: {mapping}")
        # Legal-action lists are interned so repeat visits to a node hand the
        # traversal the *same* list object its infoset was created with. That is
        # what selects the `infoset.legal_actions is legal_actions` fast path,
        # matching how production reaches it via its own legal-action memo.
        self._interned: dict[tuple[Any, ...], list[Action]] = {}

    def core_actions(self, game_actions: Sequence[Any]) -> list[Action]:
        key = tuple(game_actions)
        interned = self._interned.get(key)
        if interned is None:
            interned = [self.to_core[action] for action in game_actions]
            self._interned[key] = interned
        return interned


class AdaptedRules:
    """The two ``GameRules`` methods the traversal and its action filter call."""

    def __init__(self, actions: ActionCodec):
        self.actions = actions

    def get_legal_actions(
        self, state: AdaptedState, action_model: ActionModel | None = None
    ) -> list[Action]:
        return self.actions.core_actions(state.game.legal_actions(state.inner))

    def is_action_valid(self, state: AdaptedState, action: Action) -> bool:
        return action in set(self.get_legal_actions(state))


class ExtensiveGameSolver(MCCFRSolver):
    """``MCCFRSolver`` with its HUNL state machine swapped for a generic game.

    Everything overridden here is a game-shaped seam; ``train_iteration``, the
    traversal, the regret kernel and storage are inherited untouched. Notably
    ``train_iteration``'s ``traversing_player = iteration % 2`` alternation is
    inherited rather than reimplemented, so the training schedule is the
    production one by construction rather than by careful copying.

    HUNL's other schedule -- ``button = (iteration // 2) % 2`` in
    ``chance.deal_initial_state`` -- has no counterpart here and is dropped with
    it: these games have fixed positional roles (P0 always acts first), so there
    is no button to rotate and role-swapping would deal a different game.
    """

    def __init__(
        self,
        game: ExtensiveGame,
        action_mapping: dict[Any, Action],
        storage: KeyedStorage,
        config: Config,
    ):
        super().__init__(
            action_model=cast("ActionModel", None),
            card_abstraction=cast("Any", None),
            storage=storage,
            config=config,
        )
        self.game = game
        self.actions = ActionCodec(action_mapping)
        self.rules = cast("Any", AdaptedRules(self.actions))

    def _wrap(self, inner: Any) -> GameState:
        # Cast, not inheritance: AdaptedState satisfies the traversal's state
        # contract structurally but shares no ancestry with GameState.
        return cast("GameState", AdaptedState(self.game, inner, self.actions))

    def deal_initial_state(self) -> GameState:
        return self._wrap(self.game.initial_state())

    def is_chance_node(self, state: GameState) -> bool:
        adapted = cast("AdaptedState", state)
        return adapted.game.current_player(adapted.inner) == CHANCE

    def sample_chance_outcome(self, state: GameState) -> GameState:
        adapted = cast("AdaptedState", state)
        outcomes = adapted.game.chance_outcomes(adapted.inner)
        threshold = random.random()
        cumulative = 0.0
        for action, probability in outcomes:
            cumulative += probability
            if threshold < cumulative:
                return self._wrap(adapted.game.next_state(adapted.inner, action))
        return self._wrap(adapted.game.next_state(adapted.inner, outcomes[-1][0]))

    def deal_remaining_cards(self, state: GameState) -> GameState:
        """No board to complete; every terminal here is already payable.

        The traversal runs every terminal through this method, so it must return
        the state rather than refuse.
        """
        return state

    def encode_infoset_key(self, state: GameState, player: int) -> InfoSetKey:
        adapted = cast("AdaptedState", state)
        return adapted_infoset_key(adapted.game.information_state_key(adapted.inner, player))


def _policy_from(solver: ExtensiveGameSolver, *, use_average: bool):
    def policy(info_key: InfoKey, legal_actions: Sequence[Any]) -> list[float]:
        infoset = solver.storage.get_infoset(adapted_infoset_key(info_key))
        if infoset is None:
            # Never visited during training; the harness still needs a
            # distribution, and uniform is what the solver would play.
            return [1.0 / len(legal_actions)] * len(legal_actions)

        distribution = infoset.get_average_strategy() if use_average else infoset.get_strategy()
        index = {action: i for i, action in enumerate(infoset.legal_actions)}
        # Realign onto the harness's action order rather than assuming the stored
        # order matches: a silent misalignment would look like a strategy bug.
        return [float(distribution[index[solver.actions.to_core[a]]]) for a in legal_actions]

    return policy


def average_policy(solver: ExtensiveGameSolver):
    """The solver's AVERAGE strategy as an evaluation-harness policy.

    This is the strategy CFR's convergence guarantee is about; the current
    iterate carries no such guarantee. Reading the wrong one is the failure mode
    that would let a broken averaging scheme pass a convergence assertion, so
    ``test_kernel_conformance`` pins the two apart explicitly.
    """
    return _policy_from(solver, use_average=True)


def current_policy(solver: ExtensiveGameSolver):
    """The solver's CURRENT iterate (regret matching), for contrast only."""
    return _policy_from(solver, use_average=False)
