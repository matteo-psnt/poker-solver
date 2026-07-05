"""Typed views over the Chipzen wire payloads we read.

Parsing is separated from the adapter so the protocol's shape is stated once and
the reconstruction never touches a raw dict. Two amount conventions are worth
restating because getting either wrong is silent:

- ``ActionEntry.amount`` on a raise or a call is the actor's TOTAL wager for that
  betting round, not the increment they added to it.
- ``TurnState.min_raise`` / ``max_raise`` are likewise totals.

Amounts are integers in the table's own chips; converting them into the units our
tree was trained in belongs to :class:`~src.interfaces.chipzen.adapter.TableScale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from src.core.game.state import Card

if TYPE_CHECKING:
    from collections.abc import Sequence

# Server-generated entries. They open every hand's history and are never offered
# as a move, so replay must skip them -- our own state posts blinds itself.
SYNTHETIC_ACTIONS = frozenset({"post_small_blind", "post_big_blind", "post_ante"})

POST_SMALL_BLIND = "post_small_blind"

PHASES = ("preflop", "flop", "turn", "river")


class ProtocolError(ValueError):
    """A payload does not match the published schema.

    Raised rather than defaulted because every caller is about to bet chips on
    the answer, and a field we misread is indistinguishable from one we invented.
    """


def parse_card(text: str) -> Card:
    """One card from their two-character notation (``"Ah"``, ``"Ts"``)."""
    try:
        return Card.new(text)
    except (ValueError, KeyError, IndexError) as exc:
        raise ProtocolError(f"'{text}' is not a card.") from exc


def parse_cards(texts: Sequence[str]) -> tuple[Card, ...]:
    """A board or a holding, in the order given."""
    return tuple(parse_card(text) for text in texts)


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ProtocolError(f"Payload is missing the required field '{key}'.")
    return payload[key]


@dataclass(frozen=True)
class GameConfig:
    """``match_start.game_config`` -- the rules for every hand of one match.

    The stack's depth in big blinds is ``starting_stack / big_blind``; both are
    in the table's chips, and neither is in blinds already.
    """

    variant: str
    starting_stack: int
    small_blind: int
    big_blind: int
    ante: int
    num_players: int

    @classmethod
    def parse(cls, payload: dict[str, Any], *, seats: int | None = None) -> Self:
        """Read a ``game_config``, with the table size from ``seats`` if known.

        ``num_players`` is NOT a field they send: their `match_start.game_config`
        is variant/starting_stack/blinds/ante/total_hands, and the table size
        lives in the sibling ``seats`` array. Requiring it here raised on every
        real match -- the SDK swallowed that under `safe_mode` and the seat then
        rebuilt itself from a guessed config on the first turn instead. It stays
        on the model because the protocol spec documents it and a caller may pass
        one; absent both, heads-up is the only table this blueprint plays anyway.
        """
        players = payload.get("num_players", seats)
        config = cls(
            variant=str(_require(payload, "variant")),
            starting_stack=int(_require(payload, "starting_stack")),
            small_blind=int(_require(payload, "small_blind")),
            big_blind=int(_require(payload, "big_blind")),
            ante=int(payload.get("ante", 0)),
            num_players=int(players) if players is not None else 2,
        )
        if config.num_players != 2:
            raise ProtocolError(
                f"This blueprint plays heads-up; the table seats {config.num_players}."
            )
        if config.big_blind < config.small_blind:
            raise ProtocolError(
                f"big_blind {config.big_blind} is below small_blind {config.small_blind}."
            )
        if config.ante:
            # `GameRules` takes a small and a big blind and nothing else, so an
            # ante is dead money our tree cannot represent -- every pot would be
            # the wrong size from the first decision, silently. Refusing costs a
            # match we would have misplayed anyway.
            raise ProtocolError(
                f"This table posts an ante of {config.ante}; the blueprint's game has none."
            )
        return config

    @property
    def depth_in_blinds(self) -> float:
        """Starting stack in big blinds -- the depth our tree is cut for."""
        return self.starting_stack / self.big_blind


@dataclass(frozen=True)
class ActionEntry:
    """One line of ``action_history``.

    ``amount`` is the actor's total wager for ``phase`` after this action (0 for a
    fold or check), which is what makes a history replayable without also
    tracking what each seat had already put in.
    """

    seat: int
    action: str
    amount: int
    phase: str
    is_timeout: bool

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        return cls(
            seat=int(_require(payload, "seat")),
            action=str(_require(payload, "action")),
            amount=int(payload.get("amount", 0)),
            phase=str(_require(payload, "phase")),
            is_timeout=bool(payload.get("is_timeout", False)),
        )

    @property
    def is_synthetic(self) -> bool:
        return self.action in SYNTHETIC_ACTIONS


@dataclass(frozen=True)
class TurnState:
    """``turn_request.state`` -- everything the server will accept an answer against.

    Every number here is authoritative. Our reconstruction of the hand exists to
    name an infoset, not to price a bet; where the two disagree, this is the one
    the server validates against.
    """

    hand_number: int
    phase: str
    board: tuple[Card, ...]
    hole_cards: tuple[Card, Card]
    pot: int
    your_stack: int
    opponent_stacks: tuple[int, ...]
    to_call: int
    min_raise: int
    max_raise: int
    action_history: tuple[ActionEntry, ...]
    # The server's own list of what it will accept THIS turn. Authoritative and
    # not derivable: facing an all-in, `max_raise` can still be positive while
    # `raise` is absent, and sending one earns an `action_rejected` and a safe
    # default chosen by them rather than by the blueprint. Empty means the frame
    # did not carry it -- a recorded fixture, typically -- and nothing is filtered.
    valid_actions: tuple[str, ...] = ()

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        hole = parse_cards(_require(payload, "your_hole_cards"))
        if len(hole) != 2:
            raise ProtocolError(f"Expected 2 hole cards, got {len(hole)}.")
        phase = str(_require(payload, "phase"))
        if phase not in PHASES:
            raise ProtocolError(f"'{phase}' is not a betting phase.")
        board = parse_cards(payload.get("board", ()))
        # One deck, so a card appears once. Probed 2026-08-24: a board of
        # `2c 2c 9h`, and a hand holding a card already on the board, both came
        # back with a confident action -- the bucket for a state that cannot
        # exist. A real server never sends either, which is exactly why nothing
        # downstream checks and why a garbage frame would be answered rather
        # than caught.
        seen = [*board, *hole]
        if len({card.mask for card in seen}) != len(seen):
            raise ProtocolError(
                f"A card repeats across board {[str(c) for c in board]} and "
                f"hand {[str(c) for c in hole]}."
            )
        return cls(
            hand_number=int(_require(payload, "hand_number")),
            phase=phase,
            board=board,
            hole_cards=(hole[0], hole[1]),
            pot=int(_require(payload, "pot")),
            your_stack=int(_require(payload, "your_stack")),
            opponent_stacks=tuple(int(stack) for stack in payload.get("opponent_stacks", ())),
            to_call=int(_require(payload, "to_call")),
            min_raise=int(payload.get("min_raise", 0)),
            max_raise=int(payload.get("max_raise", 0)),
            action_history=tuple(
                ActionEntry.parse(entry) for entry in payload.get("action_history", ())
            ),
            valid_actions=tuple(str(a) for a in payload.get("valid_actions", ())),
        )

    def allows(self, action: str) -> bool:
        """Whether the server will accept ``action`` this turn.

        True when the frame carried no list at all, so a recorded fixture that
        predates the field is answered rather than refused.
        """
        return not self.valid_actions or action in self.valid_actions

    def button_seat(self) -> int:
        """The seat on the button, read off the synthetic blind rather than remembered.

        Heads-up, the button posts the small blind. Deriving it here rather than
        carrying ``dealer_seat`` over from ``round_start`` is what lets a turn be
        answered from one frame -- which is the whole reconnect story, since a
        client that dropped mid-hand never saw that frame.
        """
        for entry in self.action_history:
            if entry.action == POST_SMALL_BLIND:
                return entry.seat
        raise ProtocolError("No post_small_blind in action_history; cannot locate the button.")

    def round_wager(self, seat: int) -> int:
        """What ``seat`` already has in for the current phase, in their chips.

        A player entry carries that seat's cumulative total for the round, so the
        largest is the current one -- not the sum, which would count a call after
        a raise twice. Synthetic postings are the exception and would need adding
        together; :meth:`GameConfig.parse` refuses the only case where more than
        one of them lands on a seat, so a max is right for every table we accept.
        """
        wager = 0
        for entry in self.action_history:
            if entry.phase == self.phase and entry.seat == seat:
                wager = max(wager, entry.amount)
        return wager
