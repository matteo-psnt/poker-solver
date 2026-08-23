"""An on-tree shadow of the real hand, so blueprint lookups keep working.

Infoset keys embed the full-hand normalized betting sequence, so ONE off-menu
size from the opponent makes every later lookup on that hand miss. The
blueprint then has no answer and `blueprint_action_distribution` returns
``None``, which callers turn into a uniform row.

MEASURED, not feared. At a flop node the blueprint played
``[0.142 0.075 0.401 0.237 0.145]``; after an opponent bet of 25 against a
menu of ``[66, 132, 250]`` the same lookup returned exactly
``[0.25 0.25 0.25 0.25]``. Off-tree LBR scored blueprint+resolver at
**+2568 mbb/hand for the exploiter** against **-254 for the bare blueprint** --
the bare blueprint survives because its evaluator hands it a translated state,
which the resolver never had.

So the resolver carries its own: a second ``GameState`` whose betting history
contains only abstract-menu actions, advanced by a translated proxy of every
realized action. The real state stays authoritative for legality, chips,
terminality and payoffs; the shadow is consulted ONLY to ask the blueprint what
it would do.

Deliberately narrower than the evaluator's ``ShadowTracker``, which also has to
offer an exploiter a menu, map its choice back to a real action, and gate
candidates. Nothing here chooses anything -- it only follows. The shared part
is :func:`translate_action_distribution`, which both call.

The shadow ALIASES the real state until the first proxy differs, so a hand that
never leaves the menu costs nothing and behaves bit-identically to no shadow at
all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.game.actions import ActionType, all_in
from src.engine.search.action_translation import translate_action_distribution

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.core.game.state import GameState


class TreeShadow:
    """Follows one hand, keeping a parallel state that stays on the abstract tree."""

    def __init__(self, rules: GameRules, action_model: ActionModel):
        self._rules = rules
        self._action_model = action_model
        self._shadow: GameState | None = None
        self._diverged = False
        self._broken = False

    def start(self, state: GameState) -> None:
        self._shadow = state
        self._diverged = False
        self._broken = False

    @property
    def started(self) -> bool:
        return self._shadow is not None

    @property
    def diverged(self) -> bool:
        """Whether a proxy has ever differed from the real action."""
        return self._diverged

    @property
    def broken(self) -> bool:
        """No structure-preserving proxy existed; the shadow is not usable."""
        return self._broken

    def state_for(self, real_state: GameState) -> GameState:
        """The state a blueprint lookup should key off for ``real_state``.

        The real state itself until something goes off-menu, so the on-tree path
        is unchanged; the shadow once it has diverged; the real state again if
        the shadow broke, which restores exactly the old behaviour rather than
        inventing a new one.
        """
        if self._shadow is None or self._broken or not self._diverged:
            return real_state
        return self._shadow

    def observe(self, real_state: GameState, action: Action) -> None:
        """Advance the shadow by an on-menu proxy of a realized ``action``."""
        if self._shadow is None:
            self.start(real_state)
        if self._broken:
            return
        assert self._shadow is not None

        # A new street was dealt since the last observe. Street advancement is
        # driven by action TYPE, which the proxy preserves, so ordinarily only
        # the public cards differ -- but only while the two streets agree. They
        # can come apart (an all-in on one side, a proxy that closed betting on
        # the other), and mirroring across that mismatch asks for a board of the
        # wrong length: measured as "Board should have 3 cards on flop, got 0".
        if self._diverged and self._shadow.board != real_state.board:
            if self._shadow.street is not real_state.street:
                self._broken = True
                return
            try:
                self._shadow = self._shadow.replace(board=real_state.board)
            except (ValueError, KeyError, AssertionError):
                self._broken = True
                return

        shadow = self._shadow
        if shadow.is_terminal:
            # The real hand outlived the shadow, so there is nothing left to
            # follow. `ShadowTracker` rules this out for the EXPLOITER by a
            # structural argument (its invariant 3); here the opponent's sizes
            # are arbitrary, so it is simply possible and must be survivable.
            self._broken = True
            return

        # ADVISORY, so it degrades instead of raising. Everything this touches
        # -- proxy selection, legality, applying an action -- is reachable with
        # an arbitrary opponent size behind it, and the caller has a correct
        # answer without any of it: the real state, which is what `state_for`
        # returns once broken. A shadow that raised would turn a lookup that
        # merely loses sharpness into a dead evaluation.
        try:
            proxy = self._proxy(shadow, action)
            if proxy is None:
                self._broken = True
                return
            if not self._diverged and proxy == action:
                self._shadow = real_state.apply_action(action, self._rules)
                return
            legal = self._rules.get_legal_actions(shadow, action_model=self._action_model)
            if proxy not in legal:
                self._broken = True
                return
            advanced = shadow.apply_action(proxy, self._rules)
        except (ValueError, KeyError, AssertionError):
            self._broken = True
            return
        self._diverged = True
        self._shadow = advanced

    def _proxy(self, shadow: GameState, action: Action) -> Action | None:
        """The on-menu action the shadow takes in place of ``action``."""
        if action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return action
        if action.type == ActionType.ALL_IN:
            # Normalises to the pot-independent token; the amount never leaks.
            # A zero stack means the shadow player is already all-in, so there
            # is no jam left to mirror -- measured, not hypothetical.
            stack = shadow.stacks[shadow.current_player]
            return all_in(stack) if stack > 0 else None
        # Weighted by construction; take the heaviest so one hand replays the
        # same way twice. A resolver that sampled here would make its own
        # strategy depend on an RNG draw the opponent could not observe.
        translated = translate_action_distribution(
            shadow, observed_action=action, action_model=self._action_model, rules=self._rules
        )
        if not translated:
            return None
        return max(translated, key=lambda pair: pair[1])[0]
