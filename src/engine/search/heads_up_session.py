"""Interactive heads-up hand: a human seat versus the trained blueprint.

A resumable analogue of :func:`~src.pipeline.evaluation.resolver_match._play_game`.
That driver plays a *concrete* blueprint-vs-blueprint hand straight to a true
terminal; here one seat is a human, so the hand must **pause** whenever it is the
human's turn and resume when their action arrives. Everything else -- chance
dealing, the all-in runout, terminal payoffs -- mirrors that driver exactly so
the two agree on game semantics.

The bot plays the *raw blueprint* (no runtime resolver): the resolver is slower
(a subgame solve per decision) and measured to hurt at the frontier, so playing
the bare table is both faster and stronger for a "how good is it?" sit-down.
Wiring :class:`~src.engine.search.agent.BlueprintAgent` in for an optional
resolver seat is a later extension.

Untrained-node signal
---------------------
The blueprint is defined only on the infosets self-play reached; a lookup miss
falls back to a uniform-random legal action (see
:func:`~src.engine.solver.mccfr.policy.sample_action_from_strategy`). A human
explores off the self-play distribution far harder than self-play does, so these
misses are common -- and a bot that jams or folds at random because it has *no*
strategy there must not be mistaken for a bad blueprint. Every bot decision
records whether it was a trained lookup or a uniform fallback so the caller can
surface it; ``bot_untrained_decisions`` / ``bot_decisions`` summarize the hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, GameState, Street
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.mccfr.chance import draw_cards
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.engine.solver.protocols import Blueprint


@dataclass(frozen=True)
class HandEvent:
    """One realized decision in the hand, for the action log and hand history.

    ``street`` is the street the action was taken on (captured before it is
    applied, since a street-closing action advances the state). ``untrained`` is
    ``True`` only for a bot decision made at an infoset the blueprint never
    trained (a uniform-random fallback); it is always ``False`` for human moves.
    """

    seat: int
    actor: str  # "human" or "bot"
    action_type: str
    amount: int
    street: str
    untrained: bool


class HeadsUpHand:
    """Drives one heads-up hand between ``human_seat`` and the blueprint.

    Construct, then read :attr:`state` / :meth:`legal_actions` to render the
    human's decision; call :meth:`submit` with their chosen action to advance.
    The hand auto-plays chance nodes and every bot decision, pausing only when
    the human is to act, and settles payoffs at the terminal (completing an
    all-in board first, exactly as the evaluator does).

    RNG scope
    ---------
    ``rng`` seeds the initial hole-card deal and the bot's action sampling.
    Board cards -- each street and any all-in runout -- are dealt through the
    blueprint's canonical chance path, which draws from the process-global
    ``random`` module, so a fixed ``rng`` alone does not make the whole hand
    reproducible. Per-session reproducibility is a deliberate non-goal here (see
    the RNG note on :class:`~src.interfaces.api.play_service.PlayService`).
    """

    def __init__(
        self,
        blueprint: Blueprint,
        *,
        human_seat: int,
        button: int,
        rng: np.random.Generator | None = None,
    ):
        if human_seat not in (0, 1):
            raise ValueError(f"human_seat must be 0 or 1, got {human_seat}")
        if button not in (0, 1):
            raise ValueError(f"button must be 0 or 1, got {button}")
        self.blueprint = blueprint
        self.rules: GameRules = blueprint.rules
        self.action_model = blueprint.action_model
        self.human_seat = human_seat
        self.button = button
        self.rng = rng if rng is not None else np.random.default_rng()

        self.log: list[HandEvent] = []
        self.bot_decisions = 0
        self.bot_untrained_decisions = 0
        self.is_over = False
        self.payoffs: tuple[float, float] | None = None

        self.state = self._deal_initial_state()
        self._advance()

    # -- Public API --------------------------------------------------------

    def legal_actions(self) -> tuple[Action, ...]:
        """The human's legal actions, or empty when it is not their turn."""
        if self.is_over or self.state.current_player != self.human_seat:
            return ()
        return self.rules.get_legal_actions(self.state, action_model=self.action_model)

    def submit(self, action: Action) -> None:
        """Apply the human's ``action`` then auto-play to the next human turn."""
        if self.is_over:
            raise ValueError("Hand is already over")
        if self.state.current_player != self.human_seat:
            raise ValueError("It is not the human's turn to act")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action for the current state: {action}")
        self._record(self.human_seat, "human", action, untrained=False)
        self.state = self.state.apply_action(action, self.rules)
        self._advance()

    @property
    def showdown(self) -> bool:
        """True at a terminal reached by showdown rather than a fold."""
        return self.is_over and not self.state.ended_by_fold

    def human_payoff(self) -> float:
        """The human seat's net chips for the hand (requires a terminal)."""
        if self.payoffs is None:
            raise ValueError("Payoffs are only defined once the hand is over")
        return self.payoffs[self.human_seat]

    # -- Driver ------------------------------------------------------------

    def _advance(self) -> None:
        """Play chance nodes and bot decisions until the human acts or the hand ends."""
        while not self.state.is_terminal:
            if self.blueprint.is_chance_node(self.state):
                self.state = self.blueprint.sample_chance_outcome(self.state)
                continue
            if self.state.current_player == self.human_seat:
                return
            action, untrained = self._bot_action(self.state)
            self._record(self.state.current_player, "bot", action, untrained=untrained)
            self.bot_decisions += 1
            if untrained:
                self.bot_untrained_decisions += 1
            self.state = self.state.apply_action(action, self.rules)
        self._settle()

    def _bot_action(self, state: GameState) -> tuple[Action, bool]:
        """Sample the blueprint's action; the flag marks an untrained fallback.

        Reuses the canonical lookup primitives (encode key -> get_infoset ->
        distribution) so bot play matches
        :func:`~src.engine.solver.mccfr.policy.sample_action_from_strategy`, but
        keeps the ``distribution is None`` miss visible instead of swallowing it,
        and draws from this hand's ``rng`` rather than the process-global one.
        """
        legal = self.rules.get_legal_actions(state, action_model=self.action_model)
        if not legal:
            raise ValueError(f"No legal actions at state: {state}")
        key = encode_infoset_key(state, state.current_player, self.blueprint.card_abstraction)
        infoset = self.blueprint.storage.get_infoset(key)
        distribution = blueprint_action_distribution(
            infoset, state, self.rules, legal, use_average=True
        )
        if distribution is None:
            return legal[int(self.rng.integers(len(legal)))], True
        actions = list(distribution)
        probs = np.fromiter(distribution.values(), dtype=np.float64, count=len(actions))
        return actions[int(self.rng.choice(len(actions), p=probs))], False

    def _settle(self) -> None:
        """Finalize a terminal state: complete an all-in board, then score payoffs."""
        state = self.state
        if not state.ended_by_fold and len(state.board) < 5:
            state = self._complete_board(state)
            self.state = state
        self.is_over = True
        self.payoffs = (
            state.get_payoff(0, self.rules),
            state.get_payoff(1, self.rules),
        )

    def _complete_board(self, state: GameState) -> GameState:
        """Run out the remaining board for an early all-in (mirrors the evaluator)."""
        needed = 5 - len(state.board)
        extra = draw_cards(state, needed)
        return state.replace(
            street=Street.RIVER,
            board=(*state.board, *extra),
            is_terminal=True,
            to_call=0,
        )

    def _deal_initial_state(self) -> GameState:
        """Deal four distinct hole cards and post blinds."""
        order = self.rng.permutation(52)
        cards = [FULL_DECK[int(i)] for i in order[:4]]
        hole_cards = ((cards[0], cards[1]), (cards[2], cards[3]))
        return self.rules.create_initial_state(
            starting_stack=self.blueprint.config.game.starting_stack,
            hole_cards=hole_cards,
            button=self.button,
        )

    def _record(self, seat: int, actor: str, action: Action, *, untrained: bool) -> None:
        self.log.append(
            HandEvent(
                seat=seat,
                actor=actor,
                action_type=_action_type_name(action.type),
                amount=action.amount,
                street=str(self.state.street),
                untrained=untrained,
            )
        )


def _action_type_name(action_type: ActionType) -> str:
    """Wire-friendly action kind (e.g. ``"all-in"`` rather than ``"all_in"``)."""
    return action_type.name.lower().replace("_", "-")
