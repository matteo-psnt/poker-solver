"""Which of their refusals are answers, and which are back pressure.

MEASURED against their live engine: a 600-hand run played 423 and lost 177 to
`POST /hands -> 409`, their twenty-hand concurrency cap. Nothing had been dealt
on any of them -- the cap is a queue signal, and reading it as a failure threw
away 30% of a run.
"""

from __future__ import annotations

import httpx
import pytest

from src.interfaces.errors import CommandError
from src.interfaces.gtowizard.client import BenchmarkClient, Retry

GAME = {
    "game_id": 1,
    "game_name": "HUNL 200BB",
    "game_format": "heads-up",
    "starting_stack": 20000,
    "blinds": [100, 50],
    "stack_reset_per_hand": True,
}
STATE = {
    "street": "preflop",
    "common_pot": 0,
    "total_pot": 150,
    "board_cards": "",
    "is_hand_over": False,
    "players": [
        {"name": "villain", "stack": 1, "position": "?", "hole_cards": None},
        {"name": "hero", "stack": 1, "position": "?", "hole_cards": "AhKd"},
    ],
    "legal_actions": ["f", "c", "b"],
    "raise_range": {"min": 200, "max": 20000},
    "action_history": [],
    "has_gto_wizard_folded": False,
    "winnings": None,
    "aivat_score": None,
}
FRAME = {"hand_id": 1, "game": GAME, "game_state": STATE}
CAP_BODY = {
    "detail": "Cannot start a new hand. The maximum number of concurrent hands allowed is 20."
}

# Full jitter over a zero window: the retry SCHEDULE is tested in the live run,
# what is tested here is which codes reach it.
NO_WAIT = Retry(attempts=4, base_seconds=0.0, max_seconds=0.0)


def client_over(handler) -> BenchmarkClient:
    transport = httpx.MockTransport(handler)
    return BenchmarkClient(
        "key",
        retry=NO_WAIT,
        client=httpx.Client(base_url="https://example.invalid", transport=transport),
    )


class TestTheConcurrencyCap:
    def test_a_capped_new_hand_waits_for_a_slot_instead_of_failing(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            if len(seen) < 3:
                return httpx.Response(409, json=CAP_BODY)
            return httpx.Response(200, json=FRAME)

        with client_over(handler) as client:
            assert client.new_hand().hand_id == 1
        assert len(seen) == 3

    def test_a_cap_that_never_clears_names_the_command_that_clears_it(self) -> None:
        # A wedged cap is the one case a human must act on, so the refusal has
        # to say what to do rather than only what happened.
        with (
            client_over(lambda _: httpx.Response(409, json=CAP_BODY)) as client,
            pytest.raises(CommandError, match="benchmark-drain"),
        ):
            client.new_hand()

    def test_a_conflict_mid_hand_is_an_answer_and_is_not_retried(self) -> None:
        # A 409 on `act` says the hand is already over. Repeating it cannot
        # change that, and retrying would burn the clock on every stale hand.
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(409, json={"detail": "hand is over"})

        with client_over(handler) as client, pytest.raises(CommandError):
            client.act(1, "f")
        assert len(seen) == 1


class TestBusy:
    def test_a_busy_engine_is_retried(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(200, json=FRAME) if len(seen) > 2 else httpx.Response(503)

        with client_over(handler) as client:
            assert client.new_hand().hand_id == 1
        assert len(seen) == 3

    def test_a_rejected_key_says_so_rather_than_retrying(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(401, json={"detail": "nope"})

        with (
            client_over(handler) as client,
            pytest.raises(CommandError, match="rejected the key"),
        ):
            client.new_hand()
        assert len(seen) == 1
