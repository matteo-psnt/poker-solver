"""The hand loop, and the arithmetic that turns hands into the published metric.

The score is the one number this whole surface exists to produce, and it is
derived from two of their fields by a formula nothing else checks. So the
identity is pinned against a REAL row off their public board rather than
against itself: Bitcrumbs' -138,938.32 AIVAT chips over 44,717 hands at a 100
blind is the -3.11 bb/100 they publish.
"""

from __future__ import annotations

import pytest

from src.interfaces.gtowizard import session
from src.interfaces.gtowizard.agents import AllIn, AlwaysFold, CheckCall
from src.interfaces.gtowizard.protocol import Frame

BB = 100
GAME = {
    "game_id": 1,
    "game_name": "HUNL 200BB",
    "game_format": "heads-up",
    "starting_stack": 20000,
    "blinds": [BB, 50],  # THEIR order: big first
    "stack_reset_per_hand": True,
}


def state(
    *,
    history: list[str],
    over: bool = False,
    legal: list[str],
    winnings: float | None = None,
    aivat: float | None = None,
) -> dict:
    return {
        "street": "preflop",
        "common_pot": 0,
        "total_pot": 150,
        "board_cards": "",
        "is_hand_over": over,
        "players": [
            {"name": "villain", "stack": 1, "position": "?", "hole_cards": None},
            {"name": "hero", "stack": 1, "position": "?", "hole_cards": "AhKd"},
        ],
        "legal_actions": legal,
        "raise_range": {"min": 200, "max": 20000},
        "action_history": history,
        "has_gto_wizard_folded": False,
        "winnings": winnings,
        "aivat_score": aivat,
    }


class FakeServer:
    """Answers one scripted hand: we act once, then it is over.

    Records every request so the loop's wire encoding is inspectable -- the
    thing a live probe checks, checked here for the shape rather than the value.
    """

    def __init__(self, *, winnings: float, aivat: float, fail_after: int | None = None) -> None:
        self.winnings = winnings
        self.aivat = aivat
        self.fail_after = fail_after
        self.hands = 0
        self.sent: list[tuple[str, int | None]] = []

    def new_hand(self, game_name: str = "") -> Frame:
        self.hands += 1
        if self.fail_after is not None and self.hands > self.fail_after:
            raise RuntimeError("engine gave up")
        return Frame.parse(
            {
                "hand_id": self.hands,
                "game": GAME,
                "game_state": state(history=[], legal=["f", "c", "b"]),
            }
        )

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        self.sent.append((action, amount))
        return Frame.parse(
            {
                "hand_id": hand_id,
                "game": GAME,
                "game_state": state(
                    history=[action],
                    over=True,
                    legal=[],
                    winnings=self.winnings,
                    aivat=self.aivat,
                ),
            }
        )


class TestScore:
    def test_the_metric_reproduces_a_real_row_off_their_board(self) -> None:
        # Bitcrumbs, GTO Wizard AI v2, read 2026-08-31: -138,938.31867 AIVAT
        # chips over 44,717 hands at a 100 blind is the -3.11 they publish.
        tally = session.Tally()
        per_hand = -138_938.31867 / 44_717
        for index in range(44_717):
            tally.add(
                session.HandRecord(
                    hand_id=index, big_blind=BB, winnings=0.0, aivat=per_hand, decisions=1
                )
            )
        assert tally.aivat_bb_per_100 == pytest.approx(-3.107, abs=0.001)

    def test_the_raw_result_is_a_different_number_and_is_reported_apart(self) -> None:
        # Roman_SL is +0.51 raw and -9.76 adjusted. Quoting the wrong column is
        # the single easiest way to misreport this benchmark.
        tally = session.Tally()
        for index in range(100):
            tally.add(
                session.HandRecord(
                    hand_id=index, big_blind=BB, winnings=50.0, aivat=-900.0, decisions=1
                )
            )
        assert tally.raw_bb_per_100 == pytest.approx(50.0)
        assert tally.aivat_bb_per_100 == pytest.approx(-900.0)

    def test_one_hand_carries_no_error_bar_rather_than_a_flattering_zero(self) -> None:
        tally = session.Tally()
        tally.add(session.HandRecord(hand_id=1, big_blind=BB, winnings=0.0, aivat=0.0, decisions=1))
        assert tally.aivat_std_bb_per_100 == float("inf")


class TestLoop:
    def test_a_hand_is_played_to_its_end_and_scored(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        tally = session.run(server, CheckCall(), num_hands=3, concurrency=1)
        assert tally.played == 3
        assert tally.failed == 0
        assert tally.aivat_bb_per_100 == pytest.approx(-150.0)

    def test_check_call_sends_a_call_when_it_cannot_check(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert server.sent == [("c", None)]

    def test_always_fold_folds(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, AlwaysFold(), num_hands=1, concurrency=1)
        assert server.sent == [("f", None)]

    def test_a_bet_carries_its_amount_and_nothing_else_does(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, AllIn(), num_hands=1, concurrency=1)
        assert server.sent == [("b", 20000)]

    def test_a_failed_hand_is_dropped_from_the_score_not_counted_as_a_loss(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0, fail_after=2)
        tally = session.run(server, CheckCall(), num_hands=5, concurrency=1)
        assert tally.played == 2
        assert tally.failed == 3
        assert tally.aivat_bb_per_100 == pytest.approx(-150.0)

    def test_hands_run_concurrently_without_losing_any(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        tally = session.run(server, CheckCall(), num_hands=40, concurrency=8)
        assert tally.played == 40

    @pytest.mark.parametrize("concurrency", [0, session.MAX_CONCURRENT + 1])
    def test_their_concurrency_cap_is_refused_here_not_by_their_server(
        self, concurrency: int
    ) -> None:
        with pytest.raises(ValueError, match="Concurrency"):
            session.run(
                FakeServer(winnings=0.0, aivat=0.0),
                CheckCall(),
                num_hands=1,
                concurrency=concurrency,
            )

    def test_the_log_gets_one_object_per_hand_as_it_lands(self, tmp_path) -> None:
        path = tmp_path / "hands.jsonl"
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        session.run(server, CheckCall(), num_hands=4, concurrency=2, log_path=path)
        assert len(path.read_text().strip().splitlines()) == 4


class TestBadFrames:
    def test_a_live_frame_offering_nothing_is_named_not_answered(self) -> None:
        """The failure `allows`' old permissive-empty default would have caused.

        Not over, and no legal action: every agent falls through to its last
        resort and sends something the server rejects, and a 4xx mid-hand
        abandons a hand that would otherwise have been scored.
        """

        class OffersNothing(FakeServer):
            def new_hand(self, game_name: str = "") -> Frame:
                self.hands += 1
                return Frame.parse(
                    {
                        "hand_id": self.hands,
                        "game": GAME,
                        "game_state": state(history=[], legal=[]),
                    }
                )

        server = OffersNothing(winnings=0.0, aivat=0.0)
        tally = session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert tally.failed == 1
        assert server.sent == []


class TestRunaway:
    def test_a_hand_that_never_ends_is_cut_off_rather_than_burning_the_run(self) -> None:
        class NeverEnds(FakeServer):
            def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
                super().act(hand_id, action, amount)
                return Frame.parse(
                    {
                        "hand_id": hand_id,
                        "game": GAME,
                        "game_state": state(history=[], legal=["f", "c", "b"]),
                    }
                )

        server = NeverEnds(winnings=0.0, aivat=0.0)
        tally = session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert tally.played == 0
        assert tally.failed == 1
        assert len(server.sent) == session.MAX_DECISIONS_PER_HAND
