"""Typed views over the GTO Wizard research API's payloads.

Mirrors `https://researcher.gtowizard.com/openapi.json`. Two conventions are
worth restating because getting either wrong is silent:

- A ``bX`` token and ``raise_range`` are the actor's CUMULATIVE wager for the
  current round, not the increment added to it.
- ``aivat_score_bb_per_100`` is the benchmark's metric. ``bb_per_100`` is the
  raw chip result and is dominated by luck -- reading it as the score puts
  Roman_SL at +0.51 where the leaderboard has them at -9.76.

Amounts are their chips throughout; converting to the units our tree was
trained in belongs to :class:`~src.interfaces.gtowizard.adapter.TableScale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from src.core.game.state import Card

# Their vocabulary. There is no all-in and no separate raise: a bet and a raise
# are both 'b' with a cumulative amount, and a shove is one at raise_range.max.
FOLD = "f"
CHECK = "k"
CALL = "c"
BET = "b"

STREETS = ("preflop", "flop", "turn", "river")
# Their action_history separates streets with this, in band with the actions.
STREET_BREAK = "_"

# The only game the API serves. `NewHandRequest.game_name` is a const in their
# schema, so a second one would be a schema change, not a parameter.
GAME_NAME = "HUNL 200BB"


class ProtocolError(ValueError):
    """A payload does not match the published schema.

    Raised rather than defaulted because every caller is about to bet chips on
    the answer, and a field we misread is indistinguishable from one we invented.
    """


def parse_cards(text: str) -> tuple[Card, ...]:
    """Their concatenated notation (``"AhKd7c"``) as cards.

    One deck, so a card appears once; a repeat means we are about to bucket a
    state that cannot exist rather than notice we misread the string.
    """
    text = text.strip()
    if len(text) % 2:
        raise ProtocolError(f"'{text}' is not a whole number of two-character cards.")
    cards = []
    for index in range(0, len(text), 2):
        token = text[index : index + 2]
        try:
            cards.append(Card.new(token))
        except (ValueError, KeyError, IndexError) as exc:
            raise ProtocolError(f"'{token}' is not a card.") from exc
    if len({card.mask for card in cards}) != len(cards):
        raise ProtocolError(f"A card repeats in '{text}'.")
    return tuple(cards)


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ProtocolError(f"Payload is missing the required field '{key}'.")
    return payload[key]


@dataclass(frozen=True)
class Game:
    """``GameModel`` -- the rules, re-read from every response rather than cached.

    ``stack_reset_per_hand`` is true for HUNL 200BB, which is what makes every
    hand the same depth and this a single-blueprint problem rather than the
    depth ladder chipzen needs.
    """

    game_id: int
    game_name: str
    game_format: str
    starting_stack: int
    small_blind: int
    big_blind: int
    stack_reset_per_hand: bool

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        blinds = _require(payload, "blinds")
        if len(blinds) != 2:
            raise ProtocolError(f"Expected two blinds, got {blinds!r}.")
        small, big = (int(blind) for blind in blinds)
        if big < small:
            raise ProtocolError(f"big blind {big} is below small blind {small}.")
        return cls(
            game_id=int(_require(payload, "game_id")),
            game_name=str(_require(payload, "game_name")),
            game_format=str(_require(payload, "game_format")),
            starting_stack=int(_require(payload, "starting_stack")),
            small_blind=small,
            big_blind=big,
            stack_reset_per_hand=bool(_require(payload, "stack_reset_per_hand")),
        )

    @property
    def depth_in_blinds(self) -> float:
        """Starting stack in big blinds -- the axis our tree is cut on. 200 here."""
        return self.starting_stack / self.big_blind


@dataclass(frozen=True)
class Player:
    name: str
    stack: int
    position: str
    hole_cards: tuple[Card, ...]

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        cards = payload.get("hole_cards")
        return cls(
            name=str(_require(payload, "name")),
            stack=int(_require(payload, "stack")),
            position=str(_require(payload, "position")),
            hole_cards=parse_cards(cards) if cards else (),
        )

    @property
    def is_hero(self) -> bool:
        """Only our own cards are dealt to us; the villain's arrive as null."""
        return bool(self.hole_cards)


@dataclass(frozen=True)
class Turn:
    """``GameState`` -- everything the server will accept an answer against.

    Every number here is authoritative. Our reconstruction of the hand exists to
    name an infoset, not to price a bet; where the two disagree, this is the one
    the server validates against.
    """

    street: str
    common_pot: int
    total_pot: int
    board: tuple[Card, ...]
    is_hand_over: bool
    players: tuple[Player, ...]
    legal_actions: tuple[str, ...]
    raise_min: int
    raise_max: int
    action_history: tuple[str, ...]
    has_villain_folded: bool
    winnings: float | None
    aivat_score: float | None

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        street = str(_require(payload, "street"))
        if street not in STREETS:
            raise ProtocolError(f"'{street}' is not a betting street.")
        players = tuple(Player.parse(entry) for entry in _require(payload, "players"))
        if len(players) != 2:
            raise ProtocolError(f"This blueprint plays heads-up; the table seats {len(players)}.")
        if sum(player.is_hero for player in players) != 1:
            raise ProtocolError("Exactly one seat should hold hole cards; the frame shows another.")
        board = parse_cards(str(payload.get("board_cards") or ""))
        seen = [*board, *players[0].hole_cards, *players[1].hole_cards]
        if len({card.mask for card in seen}) != len(seen):
            raise ProtocolError("A card repeats across the board and a holding.")
        raise_range = payload.get("raise_range") or {}
        return cls(
            street=street,
            common_pot=int(_require(payload, "common_pot")),
            total_pot=int(_require(payload, "total_pot")),
            board=board,
            is_hand_over=bool(_require(payload, "is_hand_over")),
            players=players,
            legal_actions=tuple(str(action) for action in payload.get("legal_actions", ())),
            raise_min=int(raise_range.get("min", 0)),
            raise_max=int(raise_range.get("max", 0)),
            action_history=tuple(str(token) for token in payload.get("action_history", ())),
            has_villain_folded=bool(payload.get("has_gto_wizard_folded", False)),
            winnings=None if payload.get("winnings") is None else float(payload["winnings"]),
            aivat_score=(
                None if payload.get("aivat_score") is None else float(payload["aivat_score"])
            ),
        )

    @property
    def hero_seat(self) -> int:
        """Our index into ``players`` -- the seat holding cards we can see."""
        return next(index for index, player in enumerate(self.players) if player.is_hero)

    def allows(self, action: str) -> bool:
        """Whether the server will accept this base action now.

        True when the frame carried no list at all, so a recorded fixture that
        predates the field is answered rather than refused.
        """
        return not self.legal_actions or action in self.legal_actions

    @property
    def rounds(self) -> tuple[tuple[str, ...], ...]:
        """``action_history`` split on the street breaks, oldest street first.

        A trailing break -- the street closed but nobody has acted on the next
        one -- yields a final empty round, which is what :meth:`button_seat`
        needs to see.
        """
        rounds: list[list[str]] = [[]]
        for token in self.action_history:
            if token == STREET_BREAK:
                rounds.append([])
            else:
                rounds[-1].append(token)
        return tuple(tuple(entries) for entries in rounds)

    def button_seat(self) -> int:
        """The seat on the button, derived from WHOSE TURN IT IS.

        Their frame names a ``position`` per player, but we have never seen
        their vocabulary -- no key-free endpoint returns one, and guessing
        between BTN/BU/SB/D would be a silent mis-seat rather than an error.
        This needs no vocabulary at all.

        Heads-up, seats strictly alternate within a betting round: every action
        passes the turn until the round closes. The button acts first preflop
        and second on every later street. So with ``k`` actions already in the
        current round and US to act, we are the button preflop exactly when
        ``k`` is even, and postflop exactly when it is odd.

        Only meaningful on a frame where it is our turn, which is every frame we
        are asked to answer.
        """
        acted = len(self.rounds[-1])
        hero_is_button = (acted % 2 == 0) if self.street == "preflop" else (acted % 2 == 1)
        return self.hero_seat if hero_is_button else 1 - self.hero_seat

    def street_index(self) -> int:
        """Streets completed, counted from the breaks rather than from ``street``.

        The two must agree: a board ahead of the history would have the replay
        deal cards nobody has bet into.
        """
        index = len(self.rounds) - 1
        if index >= len(STREETS) or STREETS[index] != self.street:
            raise ProtocolError(
                f"History shows {index} street break(s) but the frame says {self.street}."
            )
        return index


@dataclass(frozen=True)
class Frame:
    """``GameServiceResponse`` -- one hand id, its rules and its current state."""

    hand_id: int
    game: Game
    turn: Turn

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> Self:
        return cls(
            hand_id=int(_require(payload, "hand_id")),
            game=Game.parse(_require(payload, "game")),
            turn=Turn.parse(_require(payload, "game_state")),
        )
