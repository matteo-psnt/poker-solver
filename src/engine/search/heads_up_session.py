"""Interactive heads-up hand: a human seat versus the trained blueprint.

A resumable analogue of :func:`~src.pipeline.evaluation.estimators.resolver_match._play_game`.
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

Every hand is reproducible from its seed
----------------------------------------
One ``seed`` determines everything a hand does: the hole cards, every board card,
and each of the bot's action samples. That needs two generators -- numpy for the
action draws, :class:`random.Random` for the board, because the dealing helpers
in :mod:`~src.engine.solver.mccfr.chance` use the stdlib one -- and both are
derived here from the single seed rather than asked for separately, so there is
no way to seed half a hand.

This was previously impossible and documented as a non-goal: board cards were
drawn from the ``random`` module's process-global instance, so a fixed seed
fixed the hole cards and nothing else, and two sessions in one process
interleaved their draws by arrival time. The helpers now take an explicit source.

Untrained-node signal
---------------------
The blueprint is defined only on the infosets self-play reached; a lookup miss
falls back to a uniform-random legal action. A human explores off the self-play
distribution far harder than self-play does, so these misses are common -- and a
bot that jams or folds at random because it has *no* strategy there must not be
mistaken for a bad blueprint. Every bot decision records whether it was a trained
lookup or a uniform fallback so the caller can surface it;
``bot_untrained_decisions`` / ``bot_decisions`` summarize the hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, GameState, Street
from src.engine.solver.mccfr.chance import draw_cards, is_chance_node, sample_chance_outcome
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.engine.solver.policy_source import ScorableBlueprint


@dataclass(frozen=True)
class HandEvent:
    """One realized decision in the hand, for the action log and hand history.

    ``street`` is the street the action was taken on (captured before it is
    applied, since a street-closing action advances the state). ``untrained`` is
    ``True`` only for a bot decision made at an infoset the blueprint never
    trained (a uniform-random fallback); it is always ``False`` for human moves.

    ``mix`` is the distribution the bot actually sampled from, kept so the caller
    can replay what the blueprint *would* have done after the hand ends. It is
    ``None`` for a human move and for an untrained fallback -- in the second case
    because there was no distribution, which is precisely the thing worth seeing.
    """

    seat: int
    actor: str  # "human" or "bot"
    action_type: str
    amount: int
    street: str
    untrained: bool
    mix: tuple[tuple[str, float], ...] | None = None


class HeadsUpHand:
    """Drives one heads-up hand between ``human_seat`` and the blueprint.

    Construct, then read :attr:`state` / :meth:`legal_actions` to render the
    human's decision; call :meth:`submit` with their chosen action to advance.
    The hand auto-plays chance nodes and every bot decision, pausing only when
    the human is to act, and settles payoffs at the terminal (completing an
    all-in board first, exactly as the evaluator does).
    """

    def __init__(
        self,
        blueprint: ScorableBlueprint,
        *,
        human_seat: int,
        button: int,
        seed: int | None = None,
    ):
        if human_seat not in (0, 1):
            raise ValueError(f"human_seat must be 0 or 1, got {human_seat}")
        if button not in (0, 1):
            raise ValueError(f"button must be 0 or 1, got {button}")
        self.blueprint = blueprint
        # Backend-agnostic policy access: key-addressed storage and the
        # tree-addressed one answer 'which infoset is this?' incompatibly.
        # Resolved once: the blueprint's storage backend does not change
        # mid-session, and this is on the per-decision path.
        self._policy_source = blueprint.policy_source
        self.rules: GameRules = blueprint.rules
        self.action_model = blueprint.action_model
        self.human_seat = human_seat
        self.button = button
        self.seed = seed

        # Both from the one seed. `SeedSequence` rather than reusing the integer
        # twice, so the two streams are independent instead of correlated.
        entropy = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(entropy)
        self._deal_rng = random.Random(int(entropy.generate_state(1, dtype=np.uint32)[0]))

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
        self._record(self.human_seat, "human", action, untrained=False, mix=None)
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
            if is_chance_node(self.state):
                # This session's own generator, not the process-global one: two
                # hands in flight in one server must not deal from one stream.
                self.state = sample_chance_outcome(self.state, self._deal_rng)
                continue
            if self.state.current_player == self.human_seat:
                return
            action, untrained, mix = self._bot_action(self.state)
            self._record(self.state.current_player, "bot", action, untrained=untrained, mix=mix)
            self.bot_decisions += 1
            if untrained:
                self.bot_untrained_decisions += 1
            self.state = self.state.apply_action(action, self.rules)
        self._settle()

    def _bot_action(
        self, state: GameState
    ) -> tuple[Action, bool, tuple[tuple[str, float], ...] | None]:
        """Sample the blueprint's action; the flag marks an untrained fallback.

        Reuses the canonical lookup primitives (bucket -> infoset ->
        distribution) so bot play matches the evaluator's, but keeps the
        ``distribution is None`` miss visible instead of swallowing it, and draws
        from this hand's ``rng`` rather than the process-global one.
        """
        legal = self.rules.get_legal_actions(state, action_model=self.action_model)
        if not legal:
            raise ValueError(f"No legal actions at state: {state}")
        source = self._policy_source
        bucket = source.bucket_for(state, state.current_player)
        infoset = source.infoset_at(state, bucket)
        distribution = blueprint_action_distribution(
            infoset, state, self.rules, legal, use_average=True
        )
        if distribution is None:
            return legal[int(self.rng.integers(len(legal)))], True, None
        actions = list(distribution)
        probs = np.fromiter(distribution.values(), dtype=np.float64, count=len(actions))
        chosen = actions[int(self.rng.choice(len(actions), p=probs))]
        mix = tuple(
            (_action_type_name(action.type), float(probability))
            for action, probability in distribution.items()
        )
        return chosen, False, mix

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
        extra = draw_cards(state, needed, self._deal_rng)
        return state.replace(
            validate=False,
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

    def _record(
        self,
        seat: int,
        actor: str,
        action: Action,
        *,
        untrained: bool,
        mix: tuple[tuple[str, float], ...] | None,
    ) -> None:
        self.log.append(
            HandEvent(
                seat=seat,
                actor=actor,
                action_type=_action_type_name(action.type),
                amount=action.amount,
                street=str(self.state.street),
                untrained=untrained,
                mix=mix,
            )
        )


def _action_type_name(action_type: ActionType) -> str:
    """Wire-friendly action kind (e.g. ``"all-in"`` rather than ``"all_in"``)."""
    return action_type.name.lower().replace("_", "-")
