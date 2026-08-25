"""Playing hands and adding them up.

The score is ``aivat_score`` summed over hands, in their chips, divided by the
big blind: **AIVAT bb/100 = 100 * sum(aivat_score) / big_blind / hands**.
Verified against the public board -- Bitcrumbs' -138,938.32 over 44,717 hands at
a 100 blind is the -3.11 they publish, where the raw result reads -9.15.

The error bar is the same statistic they report: the per-hand standard deviation
in big blinds, over sqrt(n), times 100. At their numbers that is ~2 bb per hand,
so **~2,000 hands buys +/-4.7 and ~50,000 buys +/-0.9**.

Hands are played concurrently because each is mostly waiting on their engine.
Their client caps concurrency at 20 and recommends fewer; a failed hand is
dropped rather than retried, since a hand abandoned mid-way is not scored.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from src.interfaces.gtowizard.protocol import BET, GAME_NAME, Frame

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.interfaces.gtowizard.agents import Agent


class Server(Protocol):
    """The two calls a hand needs.

    A protocol rather than :class:`BenchmarkClient`, because the hand loop is
    the part that runs tens of thousands of times and the API key that would
    exercise it live is granted by hand -- so the ONLY way to test it is to
    substitute a scripted server, and a concrete parameter type would make that
    a type error rather than the test it is.
    """

    def new_hand(self, game_name: str = ...) -> Frame: ...

    def act(self, hand_id: int, action: str, amount: int | None = ...) -> Frame: ...


logger = logging.getLogger(__name__)

# Their client documents 20 as the ceiling and suggests fewer so one stuck hand
# does not stall the run. MEASURED against their live engine: 2,000 hands at 5
# lost 851 to 503s that outlived 8 jittered retries, while 60 at 3 lost none.
# Their engine, not ours -- the failures are on `POST /hands`, before any card
# is dealt, so a lost hand costs throughput and biases nothing.
MAX_CONCURRENT = 20
DEFAULT_CONCURRENT = 3

# A heads-up hand cannot need this many of OUR decisions; a loop that reaches it
# is a protocol misread, and one that never terminates would burn the run.
MAX_DECISIONS_PER_HAND = 60


@dataclass(frozen=True)
class HandRecord:
    """One completed hand, as the score reads it."""

    hand_id: int
    big_blind: int
    winnings: float
    aivat: float
    decisions: int
    off_tree: int = 0
    truncated: bool = False

    def as_json(self) -> str:
        return json.dumps(
            {
                "hand_id": self.hand_id,
                "big_blind": self.big_blind,
                "winnings": self.winnings,
                "aivat": self.aivat,
                "decisions": self.decisions,
                "off_tree": self.off_tree,
                "truncated": self.truncated,
            }
        )


@dataclass
class Tally:
    """Everything a run's score is derived from, in one place.

    Each hand carries the big blind it was played at, read off its own frame, so
    nothing here caches a table's rules or can go stale against them.
    """

    hands: list[HandRecord] = field(default_factory=list)
    failed: int = 0

    def add(self, record: HandRecord) -> None:
        self.hands.append(record)

    @property
    def played(self) -> int:
        return len(self.hands)

    def _per_hand_bb(self, attribute: str) -> list[float]:
        return [getattr(hand, attribute) / hand.big_blind for hand in self.hands]

    @property
    def aivat_bb_per_100(self) -> float:
        if not self.hands:
            return 0.0
        return 100.0 * statistics.fmean(self._per_hand_bb("aivat"))

    @property
    def aivat_std_bb_per_100(self) -> float:
        """The standard ERROR of the mean, which is the column they publish."""
        if len(self.hands) < 2:
            return math.inf
        return 100.0 * statistics.stdev(self._per_hand_bb("aivat")) / math.sqrt(len(self.hands))

    @property
    def raw_bb_per_100(self) -> float:
        """The unadjusted chip result. Reported for contrast; never the score."""
        if not self.hands:
            return 0.0
        return 100.0 * statistics.fmean(self._per_hand_bb("winnings"))

    @property
    def off_tree_rate(self) -> float:
        """Opponent actions per hand that had to be snapped onto our tree."""
        if not self.hands:
            return 0.0
        return statistics.fmean(hand.off_tree for hand in self.hands)

    @property
    def truncated_hands(self) -> int:
        return sum(hand.truncated for hand in self.hands)


def play_hand(client: Server, agent: Agent, *, game_name: str = GAME_NAME) -> HandRecord:
    """One hand, start to finish.

    Their server acts for the villain between our turns, so every response is
    either the hand's end or our next decision -- there is nothing to poll.
    """
    frame = client.new_hand(game_name)
    decisions = 0
    off_tree = 0
    truncated = False
    while not frame.turn.is_hand_over:
        if decisions >= MAX_DECISIONS_PER_HAND:
            raise RuntimeError(
                f"Hand {frame.hand_id} passed {MAX_DECISIONS_PER_HAND} decisions; "
                "the loop is not reading the protocol."
            )
        if not frame.turn.legal_actions:
            # Not over, and nothing offered. `allows` is strict, so every agent
            # would fall through to its last resort and send an action the
            # server rejects; a 4xx mid-hand abandons a hand that would have
            # been scored. Say what happened instead.
            raise RuntimeError(
                f"Hand {frame.hand_id} is not over but offers no legal action "
                f"on the {frame.turn.street}; the protocol is being misread."
            )
        move = agent.decide(frame)
        amount = int(move.amount) if move.action == BET and move.amount is not None else None
        frame = client.act(frame.hand_id, move.action, amount)
        decisions += 1
        off_tree += move.off_tree
        truncated = truncated or move.truncated
    turn = frame.turn
    if turn.winnings is None or turn.aivat_score is None:
        raise RuntimeError(f"Hand {frame.hand_id} ended without a score; the frame carried none.")
    return HandRecord(
        hand_id=frame.hand_id,
        big_blind=frame.game.big_blind,
        winnings=turn.winnings,
        aivat=turn.aivat_score,
        decisions=decisions,
        off_tree=off_tree,
        truncated=truncated,
    )


def run(
    client: Server,
    agent: Agent,
    *,
    num_hands: int,
    concurrency: int = DEFAULT_CONCURRENT,
    log_path: Path | None = None,
    on_hand: Callable[[Tally], None] | None = None,
) -> Tally:
    """Play ``num_hands`` and add them up, ``concurrency`` in flight at a time.

    ``log_path`` gets one JSON object per hand as it lands, so a run that dies
    at hour three is still worth the hands it finished.
    """
    if not 1 <= concurrency <= MAX_CONCURRENT:
        raise ValueError(f"Concurrency must be 1..{MAX_CONCURRENT}; got {concurrency}.")
    tally = Tally()
    handle = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(play_hand, client, agent) for _ in range(num_hands)]
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception:
                    tally.failed += 1
                    logger.exception("Hand failed; dropping it from the score.")
                    continue
                tally.add(record)
                if handle:
                    handle.write(record.as_json() + "\n")
                    handle.flush()
                if on_hand is not None:
                    on_hand(tally)
    finally:
        if handle:
            handle.close()
    return tally
