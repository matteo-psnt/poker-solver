"""Persistent on-tree shadow state for rigorous off-tree LBR.

The LBR exploiter may use bet/raise sizes outside the blueprint's action
abstraction. Infoset keys embed the full-hand normalized betting sequence, so
an off-tree amount would poison every later lookup on the hand (unseen token →
miss → uniform fallback), invalidating the bound. The fix is a second
``GameState`` — the *shadow* — carried alongside the real one, whose betting
history contains only abstract-menu actions: each off-tree exploiter size is
committed to an on-tree proxy on the shadow, and all blueprint lookups key off
the shadow. The real state stays authoritative for legality, chips,
terminality, and payoffs.

Structural invariants (the correctness core):

1. Shadow and real histories have identical action-TYPE sequences. Street
   closure (CALL, check-check) and template/raise-count selection are
   type-driven, so street, current player, to-call sign, and raise caps stay in
   lockstep.
2. One-sided all-in parity: a player all-in in the SHADOW is all-in in the
   real state. The converse can transiently fail: a map-back promotion
   (scaled opponent BET/RAISE reaching the real stack → ALL_IN) puts the real
   side all-in while the shadow plays a normal size. That asymmetry is benign
   and short-lived — the exploiter's every response to a real all-in (fold, or
   a call encoded ALL_IN) is terminal, so the real hand ends within one
   decision and the shadow never advances through the asymmetry (terminal
   transitions are never committed).
3. Caller lemma (requires equal starting stacks, which
   ``GameRules.create_initial_state`` enforces): with equal starting stacks,
   ``caller_stack - aggressor_stack == to_call``, so a player facing a bet
   always covers it, with equality iff the aggressor is all-in. With invariants
   1-2 this gives: shadow terminal ⟹ real terminal (a shadow CALL can only end
   the shadow if the shadow aggressor is all-in, in which case the real
   aggressor is too and the real hand is already over). The dangerous
   direction — shadow terminal while the real hand continues — is impossible
   by construction.
4. Every action applied to the shadow is in
   ``rules.get_legal_actions(shadow, action_model)`` at its application node
   (asserted in :meth:`ShadowTracker.commit`), so the shadow never leaves the
   abstract tree and blueprint lookups can only miss on genuinely untrained
   infosets.

SPR note: shadow-keyed lookups use the shadow's own ``min(stacks)/pot`` SPR
bucket. That is the consistent choice — it matches the on-tree history that
produced the shadow, i.e. the situation the blueprint actually trained on.

Proxy selection is *not* allowed to fall back to
``ActionModel.discretize_action``: its fallbacks can return ALL_IN or cross
action types, violating invariant 2. When no structure-preserving proxy exists
the candidate is gated out of the exploiter's menu instead — that only
restricts the exploiter, which loosens the lower bound but never invalidates
it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType, all_in, bet, raises
from src.core.game.rules import GameRules
from src.core.game.state import GameState, Street


@dataclass(frozen=True)
class MenuCandidate:
    """One exploiter menu entry: the real action plus its shadow realization.

    ``shadow_dist`` is a non-empty distribution over on-tree shadow actions: a
    singleton for mirrored on-tree actions, the pseudo-harmonic pair for
    off-tree sizes. Scoring uses the full mixture; the committed proxy is
    sampled once only if the candidate is chosen (keeps RNG draw order stable).
    """

    real_action: Action
    shadow_dist: tuple[tuple[Action, float], ...]


class ShadowTracker:
    """Carries the on-tree shadow ``GameState`` through one LBR hand.

    The shadow *aliases* the real state (``is``-identity) until the first
    committed proxy differs from the real action, so the
    ``include_off_tree=False`` path never diverges, draws no RNG, and is
    bitwise-identical to an eval without the tracker.

    ``broken`` is a deployed-opponent-only degradation: there the opponent's
    realized actions are REAL actions from the resolver, and after divergence
    one may have no structure-preserving shadow counterpart. Since in deployed
    mode the shadow only feeds the exploiter's *selection* scorer, breaking is
    harmless to the bound: the tracker re-aliases the real state and the scorer
    degrades to real-state lookups (today's behavior). Unreachable in blueprint
    mode, where opponent actions are chosen ON the shadow and exploiter actions
    are menu-gated.
    """

    def __init__(self, rules: GameRules, action_model: ActionModel):
        self._rules = rules
        self._action_model = action_model
        self._shadow: GameState | None = None
        self._diverged = False
        self._broken = False
        # Hands-level diagnostics (tests assert divergence actually happened).
        self.divergence_count = 0

    def start(self, initial_state: GameState) -> None:
        """Begin a hand: the shadow aliases the real initial state."""
        self._shadow = initial_state
        self._diverged = False
        self._broken = False

    @property
    def state(self) -> GameState:
        """The current shadow state (the real state itself until divergence)."""
        assert self._shadow is not None, "ShadowTracker.start() not called"
        return self._shadow

    @property
    def diverged(self) -> bool:
        return self._diverged

    @property
    def broken(self) -> bool:
        return self._broken

    def mark_broken(self, real_state: GameState) -> None:
        """Give up shadowing for the rest of the hand (deployed mode only)."""
        self._broken = True
        self._diverged = False
        self._shadow = real_state

    # -- Mirroring: real exploiter/opponent action -> shadow action ---------

    def counterpart(self, real_state: GameState, action: Action) -> Action | None:
        """Deterministic shadow mirror of an on-tree real action.

        Returns ``None`` when no structure-preserving mirror exists (the caller
        gates the action out of the menu, or marks the tracker broken in
        deployed mode).
        """
        if not self._diverged:
            return action
        if action.type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            # Legal on the shadow by invariants 1-3 (to_call-sign parity; the
            # caller lemma plus all-in parity guarantee CALL stays a live call).
            return action
        shadow = self.state
        if action.type == ActionType.ALL_IN:
            # Normalizes to the pot-independent token "a"; amount never leaks.
            return all_in(shadow.stacks[shadow.current_player])
        return self._nearest_same_type(real_state, action)

    def _nearest_same_type(self, real_state: GameState, action: Action) -> Action | None:
        """Nearest non-all-in same-type shadow-legal action, by relative size.

        Postflop sizes are strategically pot-relative, so nearness is measured
        in pot fractions (raises: fraction of the pot after the pending call).
        Preflop sizes are blind-anchored, so nearness is the total chip commit.
        Ties break toward the smaller amount.
        """
        shadow = self.state
        candidates = [
            legal
            for legal in self._rules.get_legal_actions(shadow, action_model=self._action_model)
            if legal.type == action.type
        ]
        if not candidates:
            return None

        if shadow.street == Street.PREFLOP:
            if action.type == ActionType.RAISE:
                real_total = real_state.to_call + action.amount
                return min(
                    candidates,
                    key=lambda a: (abs((shadow.to_call + a.amount) - real_total), a.amount),
                )
            return min(candidates, key=lambda a: (abs(a.amount - action.amount), a.amount))

        if action.type == ActionType.RAISE:
            real_frac = action.amount / (real_state.pot + real_state.to_call)
            base = shadow.pot + shadow.to_call
        else:
            real_frac = action.amount / real_state.pot
            base = shadow.pot
        return min(
            candidates,
            key=lambda a: (abs(a.amount / base - real_frac), a.amount),
        )

    def off_tree_dist(
        self, real_state: GameState, action: Action
    ) -> tuple[tuple[Action, float], ...] | None:
        """Gated proxy distribution for an off-tree BET/RAISE.

        The observed amount is rescaled to the shadow's pot (sizes are
        pot-relative; pre-divergence the ratio is 1), then interpolated over the
        shadow's **structure-preserving** candidates: same-type, non-all-in
        legal actions only. This mirrors
        :func:`~src.engine.search.action_translation.translate_action_distribution`'s
        pseudo-harmonic interpolation but never admits a jam proxy — a jam
        would break all-in parity (invariant 2), and with sparse raise menus
        (e.g. ``[min_raise, jam]``) post-validating the generic translation
        would gate the entire off-tree raise axis. Amounts beyond the largest
        structure-preserving size clamp to it (the completion treats the
        overbet as that size; approximating the completion is the translation
        policy's prerogative and never breaks the bound). Returns ``None`` —
        gate the size out of the menu — only when no same-type candidate
        exists at all.
        """
        shadow = self.state
        if action.type == ActionType.RAISE:
            scale = (shadow.pot + shadow.to_call) / (real_state.pot + real_state.to_call)
        else:
            scale = shadow.pot / real_state.pot
        target = max(1, round(action.amount * scale))

        candidates = sorted(
            {
                legal
                for legal in self._rules.get_legal_actions(shadow, action_model=self._action_model)
                if legal.type == action.type
            },
            key=lambda a: a.amount,
        )
        if not candidates:
            return None

        mode = self._action_model.config.action_model.off_tree_mapping
        if mode != "probabilistic" or len(candidates) == 1:
            nearest = min(candidates, key=lambda a: (abs(a.amount - target), a.amount))
            return ((nearest, 1.0),)
        if target <= candidates[0].amount:
            return ((candidates[0], 1.0),)
        if target >= candidates[-1].amount:
            return ((candidates[-1], 1.0),)
        upper_idx = next(i for i, a in enumerate(candidates) if a.amount >= target)
        lower, upper = candidates[upper_idx - 1], candidates[upper_idx]
        span = upper.amount - lower.amount
        if span <= 0:
            return ((lower, 1.0),)
        upper_weight = (target - lower.amount) / span
        return ((lower, 1.0 - upper_weight), (upper, upper_weight))

    # -- Map-back: opponent's shadow choice -> real action ------------------

    def map_back(self, real_state: GameState, shadow_action: Action) -> Action:
        """Realize the opponent's shadow decision in the real game.

        Sizes are rescaled to preserve the pot-relative (postflop) or total-BB
        (preflop) intent, promoting to ALL_IN when the real stack cannot cover
        — those promotions can only end the REAL hand first, the allowed
        divergence direction. Raises loudly on a rules-invalid result rather
        than silently substituting (an eval that quietly measured something
        else would be lying).
        """
        if not self._diverged:
            return shadow_action

        shadow = self.state
        real_stack = real_state.stacks[real_state.current_player]
        mapped: Action
        if shadow_action.type in (ActionType.FOLD, ActionType.CHECK):
            mapped = shadow_action
        elif shadow_action.type == ActionType.CALL:
            # Unreachable defensively: all-in parity + the caller lemma make a
            # shadow-legal CALL real-legal; kept for the covering-jam encoding.
            mapped = shadow_action if real_state.to_call < real_stack else all_in(real_stack)
        elif shadow_action.type == ActionType.ALL_IN:
            mapped = all_in(real_stack)
        elif shadow_action.type == ActionType.BET:
            amount = max(1, round(shadow_action.amount * real_state.pot / shadow.pot))
            mapped = all_in(real_stack) if amount >= real_stack else bet(amount)
        else:  # RAISE
            if shadow.street == Street.PREFLOP:
                shadow_total = shadow.to_call + shadow_action.amount
                amount = max(1, shadow_total - real_state.to_call)
            else:
                scale = (real_state.pot + real_state.to_call) / (shadow.pot + shadow.to_call)
                amount = max(1, round(shadow_action.amount * scale))
            if real_state.to_call + amount >= real_stack:
                mapped = all_in(real_stack)
            else:
                mapped = raises(amount)

        if not self._rules.is_action_valid(real_state, mapped):
            raise RuntimeError(
                f"Shadow action {shadow_action} mapped to invalid real action {mapped} "
                f"(real to_call={real_state.to_call}, stack={real_stack})"
            )
        return mapped

    # -- Advancing the shadow ------------------------------------------------

    def commit(self, real_action: Action, real_next: GameState, shadow_action: Action) -> None:
        """Advance the shadow one step to match the real transition.

        Pre-divergence identical steps keep aliasing the real state. The
        shadow-legality assert is the load-bearing zero-leak guarantee
        (invariant 4); the terminality assert is invariant 3's contrapositive
        (``real_next`` is non-terminal — terminal hands never commit).
        """
        if self._broken or (not self._diverged and shadow_action == real_action):
            self._shadow = real_next
            return
        if not self._diverged:
            self._diverged = True
            self.divergence_count += 1
        shadow = self.state
        assert shadow_action in self._rules.get_legal_actions(
            shadow, action_model=self._action_model
        ), f"shadow action {shadow_action} is off the abstract tree at {shadow}"
        self._shadow = shadow.apply_action(shadow_action, self._rules)
        assert not self._shadow.is_terminal, (
            "shadow terminal while the real hand continues — structural invariant broken"
        )

    def mirror_chance(self, real_after: GameState) -> None:
        """Mirror a board deal onto the shadow (same public cards).

        The street itself was already advanced identically on both sides by
        the closing action (street closure is type-driven), so only the board
        differs; current player / to-call were reset by that same transition.
        """
        if self._broken or not self._diverged:
            self._shadow = real_after
            return
        self._shadow = self.state.replace(board=real_after.board)

    def assert_sync(self, real_state: GameState) -> None:
        """Cheap structural-parity check, run at every decision node."""
        shadow = self.state
        assert shadow.street == real_state.street
        assert shadow.current_player == real_state.current_player
        assert (shadow.to_call > 0) == (real_state.to_call > 0)
        assert not shadow.is_terminal
        # One-sided by design (invariant 2): a map-back promotion can put the
        # REAL side all-in one decision before the hand ends, but a shadow-only
        # all-in would threaten shadow-terminal-first and must never happen.
        for seat in (0, 1):
            assert shadow.stacks[seat] > 0 or real_state.stacks[seat] == 0, (
                "shadow all-in without real all-in — structural invariant broken"
            )
