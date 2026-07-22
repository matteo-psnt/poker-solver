"""Play-against-the-blueprint service: sessions, redaction, and the action wire.

Holds one loaded blueprint (the cost centre -- load once, share read-only) and a
registry of in-progress hands (:class:`~src.engine.search.heads_up_session.HeadsUpHand`).
Turns each hand into a JSON view the UI renders, with two invariants the browser
can never see around:

* **Hidden information.** The bot's hole cards are omitted from every view until
  a genuine showdown. Redaction happens here, server-side; the cards are never
  serialized early, so devtools cannot reveal them.
* **No overloaded amounts on the wire.** ``Action.amount`` means different things
  per action type (raise-over-call vs total stack vs total bet). The client never
  sees or sends a raw amount: legal actions are sent as ``{id, kind, label,
  chips}`` and the client posts back only the ``id``, which maps to a concrete
  ``Action`` here. ``chips`` is always "chips the human puts in" -- a single
  unambiguous meaning for display.

Concurrency: a process-global lock serializes all session mutation. That is fully
correct for one local player. It is *not* a horizontal-scaling story -- the shared
``rules`` object memoizes into a mutable cache and card dealing draws from the
per-hand ``rng`` while the blueprint's own sampling path uses process-global RNG.
Multi-user productionization needs per-session ``rules``/RNG (or a per-session
worker), tracked separately; the lock keeps the single-player MVP honest.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.core.game.actions import Action, ActionType
from src.core.game.state import GameState
from src.engine.search.heads_up_session import HeadsUpHand
from src.engine.solver.protocols import Blueprint
from src.pipeline import services
from src.pipeline.training.components import build_evaluation_solver


class PlayService:
    """Loads a blueprint once and manages interactive hands against it."""

    def __init__(self, run_id: str, blueprint: Blueprint, *, max_sessions: int = 64):
        self.run_id = run_id
        self.blueprint = blueprint
        self.game = blueprint.config.game
        self._sessions: OrderedDict[str, HeadsUpHand] = OrderedDict()
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        # Seat/button assignment only; per-hand dealing/sampling has its own rng.
        self._table_rng = np.random.default_rng()

    @classmethod
    def from_run_dir(cls, run_dir: Path, run_id: str | None = None) -> PlayService:
        metadata = services.load_run_metadata(run_dir)
        solver, _ = build_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
        return cls(run_id=run_id or run_dir.name, blueprint=solver)

    def num_infosets(self) -> int:
        return self.blueprint.storage.num_infosets()

    # -- Session lifecycle -------------------------------------------------

    def new_hand(self, *, human_seat: int | None = None, button: int | None = None) -> dict:
        """Deal a fresh hand and return its (redacted) view.

        ``human_seat`` and ``button`` are randomized independently when unset, so
        the human plays both positions and both button assignments over a session.
        """
        with self._lock:
            seat = human_seat if human_seat is not None else int(self._table_rng.integers(2))
            btn = button if button is not None else int(self._table_rng.integers(2))
            hand = HeadsUpHand(
                self.blueprint,
                human_seat=seat,
                button=btn,
                rng=np.random.default_rng(),
            )
            session_id = uuid.uuid4().hex
            # A new key is appended last in an OrderedDict, so no move_to_end here
            # (submit_action refreshes LRU order on an existing key, which does).
            self._sessions[session_id] = hand
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return self._view(session_id, hand)

    def submit_action(self, session_id: str, action_id: int) -> dict:
        """Apply the human's chosen legal action (by id) and return the new view."""
        with self._lock:
            hand = self._require(session_id)
            legal = hand.legal_actions()
            if not legal:
                raise ValueError("It is not your turn to act")
            if not 0 <= action_id < len(legal):
                raise ValueError(f"Unknown action id: {action_id}")
            hand.submit(legal[action_id])
            self._sessions.move_to_end(session_id)
            return self._view(session_id, hand)

    def get_state(self, session_id: str) -> dict:
        with self._lock:
            hand = self._require(session_id)
            return self._view(session_id, hand)

    def _require(self, session_id: str) -> HeadsUpHand:
        hand = self._sessions.get(session_id)
        if hand is None:
            raise KeyError(session_id)
        return hand

    # -- Serialization -----------------------------------------------------

    def _view(self, session_id: str, hand: HeadsUpHand) -> dict:
        state = hand.state
        your_turn = not hand.is_over and state.current_player == hand.human_seat
        view = {
            "sessionId": session_id,
            "runId": self.run_id,
            "humanSeat": hand.human_seat,
            "button": hand.button,
            "bigBlind": self.game.big_blind,
            "smallBlind": self.game.small_blind,
            "startingStack": self.game.starting_stack,
            "street": str(state.street),
            "board": [repr(card) for card in state.board],
            "pot": state.pot,
            "stacks": list(state.stacks),
            "toCall": state.to_call,
            "currentPlayer": state.current_player,
            "yourHole": [repr(card) for card in state.hole_cards[hand.human_seat]],
            # Redaction: the bot's cards are serialized only at a real showdown.
            "botHole": (
                [repr(card) for card in state.hole_cards[1 - hand.human_seat]]
                if hand.showdown
                else None
            ),
            "yourTurn": your_turn,
            "isOver": hand.is_over,
            "legalActions": (
                [
                    _serialize_action(state, idx, action)
                    for idx, action in enumerate(hand.legal_actions())
                ]
                if your_turn
                else []
            ),
            "botDecisions": hand.bot_decisions,
            "botUntrainedDecisions": hand.bot_untrained_decisions,
            "log": [
                {
                    "seat": e.seat,
                    "actor": e.actor,
                    "actionType": e.action_type,
                    "amount": e.amount,
                    "street": e.street,
                    "untrained": e.untrained,
                }
                for e in hand.log
            ],
        }
        if hand.is_over:
            view["result"] = self._result(hand)
        return view

    def _result(self, hand: HeadsUpHand) -> dict:
        payoff = hand.human_payoff()
        state = hand.state
        if payoff > 0:
            outcome = "win"
        elif payoff < 0:
            outcome = "loss"
        else:
            outcome = "tie"
        return {
            "humanPayoff": payoff,
            "outcome": outcome,
            "terminal": "showdown" if hand.showdown else "fold",
            "endedByFold": state.ended_by_fold,
        }


# Human-readable label for a legal action; ``chips`` is always "chips the human
# adds to the pot", the one meaning that holds across every action type.
def _serialize_action(state: GameState, action_id: int, action: Action) -> dict:
    kind = action.type.name.lower().replace("_", "-")
    if action.type == ActionType.FOLD:
        label, chips = "Fold", 0
    elif action.type == ActionType.CHECK:
        label, chips = "Check", 0
    elif action.type == ActionType.CALL:
        chips = state.to_call
        label = f"Call {chips}"
    elif action.type == ActionType.BET:
        chips = action.amount
        label = f"Bet {chips}"
    elif action.type == ActionType.RAISE:
        # RAISE.amount is the raise over the call; chips put in is call + raise.
        chips = state.to_call + action.amount
        label = f"Raise ({chips})"
    elif action.type == ActionType.ALL_IN:
        # ALL_IN.amount is the whole stack committed. Facing a bet it is a call
        # (or re-jam); with nothing to call it is an opening shove. Label it as
        # what the player experiences, not by its internal type.
        chips = action.amount
        label = f"Call all-in {chips}" if state.to_call > 0 else f"All-in {chips}"
    else:  # pragma: no cover - exhaustive over ActionType
        raise ValueError(f"Unknown action type: {action.type}")
    return {"id": action_id, "kind": kind, "label": label, "chips": chips}
